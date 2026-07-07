/****************************************************************************
 * Copyright (c) 2024, JStewart28                                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Tessera library. Tessera is distributed under a *
 * BSD 3-Clause license. For the licensing terms see the LICENSE file in   *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef TESSERA_MARKQUALITY_HPP
#define TESSERA_MARKQUALITY_HPP

#include "Tessera_Fields.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Tessera
{

// ============================================================================
// Quality-based refinement marking (Step 10)
// ============================================================================
//
// A QualityCriterion inspects mesh geometry and returns the owned-face mask
// `refine()` already consumes, so a caller can drive AMR from mesh quality
// instead of hand-authoring the mask:
//
//   markByQuality(mesh, crit) -> refine(mesh, halo, mask, policy)
//
// Concept (duck-typed, like RefinePolicy): a criterion is any struct with
//   template <class MeshT> std::vector<char> mark( const MeshT& mesh ) const;
// returning a mask sized mesh.numOwnedFaces(), 1 = refine this owned face. The
// criterion owns its own evaluation, including any communication it needs --
// this is why the interface is a whole-mesh mark() and not a per-face
// bool operator()(face): EdgeLengthCriterion (10a) is embarrassingly local, but
// a later curvature criterion (10b) needs a cross-rank gather, and a uniform
// mark(mesh) interface hides that difference from the caller.
//
// Both metrics recompute on demand from the existing VertexField::Position /
// FaceField::Verts -- no new stored fields, no halo/migrate/refine propagation
// to design for.

namespace detail
{

//! Device-kernel core of EdgeLengthCriterion::mark. Pure per-owned-face
//! geometry, no MPI: every owned face's 3 vertices are local (owned or ghost --
//! the 1-ring closure invariant) and a shared vertex's Position is
//! bit-identical across ranks (same gid => same position, established by the
//! builder + halo sync), so the marked set is rank-count independent with no
//! communication.
template <class MeshT>
std::vector<char> markEdgeLength( const MeshT& mesh,
                                  typename MeshT::scalar_type maxLen )
{
    using Scalar = typename MeshT::scalar_type;
    using memory_space = typename MeshT::memory_space;
    using execution_space = typename MeshT::execution_space;
    constexpr int Dim = MeshT::dim;

    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );
    if ( nOwnedF == 0 )
        return std::vector<char>();

    const int nv = static_cast<int>( mesh.numVertices() );

    // ---- host: vertex gid -> local index map (same pattern as refine()) ----
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "mark_quality_hv", nv );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    std::unordered_map<GlobalId, int> gid2lv;
    gid2lv.reserve( static_cast<std::size_t>( nv ) * 2 );
    for ( int i = 0; i < nv; ++i )
        gid2lv[v_gid( i )] = i;

    // ---- host: owned-face -> vertex-local-index view ------------------------
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "mark_quality_hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );

    Kokkos::View<int* [3], memory_space> faceVertLocal(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "mark_quality_face_vert_local" ),
        nOwnedF );
    auto h_fvl = Kokkos::create_mirror_view( faceVertLocal );
    for ( int f = 0; f < nOwnedF; ++f )
        for ( int k = 0; k < 3; ++k )
            h_fvl( f, k ) =
                gid2lv.at( static_cast<GlobalId>( f_verts( f, k ) ) );
    Kokkos::deep_copy( faceVertLocal, h_fvl );

    // ---- device: read the (already device-resident) Position slice ---------
    auto pos = Cabana::slice<VertexField::Position>( mesh.vertices() );

    Kokkos::View<char*, memory_space> markDev(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "mark_quality_mark" ),
        nOwnedF );

    const Scalar maxLenSq = maxLen * maxLen;
    Kokkos::parallel_for(
        "tessera_mark_edge_length",
        Kokkos::RangePolicy<execution_space>( 0, nOwnedF ),
        KOKKOS_LAMBDA( const int f ) {
            const int vl[3] = { faceVertLocal( f, 0 ), faceVertLocal( f, 1 ),
                                faceVertLocal( f, 2 ) };
            char m = 0;
            for ( int k = 0; k < 3; ++k )
            {
                const int a = vl[k];
                const int b = vl[( k + 1 ) % 3];
                Scalar lenSq = Scalar( 0 );
                for ( int d = 0; d < Dim; ++d )
                {
                    const Scalar diff = pos( a, d ) - pos( b, d );
                    lenSq += diff * diff;
                }
                if ( lenSq > maxLenSq )
                    m = 1;
            }
            markDev( f ) = m;
        } );
    Kokkos::fence();

    auto h_mark =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), markDev );
    std::vector<char> mask( nOwnedF );
    for ( int f = 0; f < nOwnedF; ++f )
        mask[f] = h_mark( f );
    return mask;
}

} // namespace detail

// --- 10a: local, no communication -------------------------------------------

//! Mark an owned face if any of its 3 edges exceeds `maxLen` (an absolute
//! target edge length -- the vortex-sheet / interface-tracking convention:
//! insert points when a segment exceeds epsilon). Evaluated with a device
//! Kokkos kernel (see detail::markEdgeLength); no communication.
template <class Scalar>
struct EdgeLengthCriterion
{
    Scalar maxLen;

    template <class MeshT>
    std::vector<char> mark( const MeshT& mesh ) const
    {
        return detail::markEdgeLength( mesh, maxLen );
    }
};

//! Uniform entry point: dispatches to the criterion's own mark(). Disabled for
//! arithmetic `Criterion` so a bare scalar threshold resolves to the
//! convenience overload below instead of an ambiguous call.
template <class MeshT, class Criterion,
          class = std::enable_if_t<!std::is_arithmetic<Criterion>::value>>
std::vector<char> markByQuality( const MeshT& mesh, const Criterion& crit )
{
    return crit.mark( mesh );
}

//! Scalar convenience overload: builds an EdgeLengthCriterion<Scalar>.
template <class MeshT>
std::vector<char> markByQuality( const MeshT& mesh,
                                 typename MeshT::scalar_type maxEdgeLength )
{
    return EdgeLengthCriterion<typename MeshT::scalar_type>{ maxEdgeLength }
        .mark( mesh );
}

} // namespace Tessera

#endif // TESSERA_MARKQUALITY_HPP
