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

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_RefineParallel.hpp" // detail::edgeCoordRank (shared coordinator)
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <map>
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
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_MARK_EDGE_LEN );
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
    {
        TESSERA_SCOPED_TIMER_VERBOSE(
            ::Tessera::Profiling::TIMER_MARK_EDGE_KERNEL );
        Kokkos::parallel_for(
            "tessera_mark_edge_length",
            Kokkos::RangePolicy<execution_space>( 0, nOwnedF ),
            KOKKOS_LAMBDA( const int f ) {
                const int vl[3] = { faceVertLocal( f, 0 ),
                                    faceVertLocal( f, 1 ),
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
    }

    auto h_mark =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), markDev );
    std::vector<char> mask( nOwnedF );
    for ( int f = 0; f < nOwnedF; ++f )
        mask[f] = h_mark( f );
    return mask;
}

//! One (edge, incident owned face) advertisement for the curvature coordinator:
//! the face's unit normal plus its gid/owner so a mark can be routed back.
//! Trivially copyable (POD) so it ships through allToAllV as raw bytes.
template <class Scalar>
struct NormalMsg
{
    EdgeKey key;
    Scalar n[3];
    GlobalId faceGid;
    Rank owner;
};

//! Coordinator core of CurvatureCriterion::mark. Dihedral bend across an edge
//! needs BOTH incident faces' normals, and the neighbour across a partition
//! boundary is NOT guaranteed present in the vertex-based 1-ring halo (Step 6b's
//! analysis: at a 3-way corner the edge's two vertices can both be ghosts owned
//! by a lower rank, so the neighbour is incident to no owned vertex). The gather
//! is therefore routed through EDGE COORDINATORS (detail::edgeCoordRank +
//! allToAllV), exactly the Step-6b Phase-1 idiom -- NOT the halo. Because the
//! verdict is computed at a single deterministic coordinator per edge from both
//! true incident normals, the marked set is rank-count independent (and
//! boundary-straddle independent) by construction.
//!
//! Winding/orientation assumption: the builder/refine maintain a consistent
//! CCW-seen-from-outside winding with the edge convention e[k]=(v[k],v[k+1]), so
//! n = normalize((p1-p0) x (p2-p0)) is the OUTWARD normal on every face and
//! adjacent normals are comparably oriented -- n0.n1 = cos(dihedral bend), = 1 on
//! a flat surface and decreasing as the fold sharpens. An edge is "sharp" iff
//! acos(n0.n1) > maxAngle, i.e. n0.n1 < cos(maxAngle) (the cos form avoids acos
//! round-off near 0).
template <class MeshT>
std::vector<char> markCurvature( const MeshT& mesh,
                                 typename MeshT::scalar_type maxAngle )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_MARK_CURVATURE );
    using Scalar = typename MeshT::scalar_type;
    using memory_space = typename MeshT::memory_space;
    using execution_space = typename MeshT::execution_space;
    constexpr int Dim = MeshT::dim;

    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );

    // Dim != 3: a surface embedded in the plane has no dihedral bend; the cross
    // product is 3D. Nothing to mark (Milestone 1 is always Dim == 3).
    if constexpr ( Dim != 3 )
    {
        return std::vector<char>( static_cast<std::size_t>( nOwnedF ), 0 );
    }
    else
    {
        const int R = mesh.rank();
        const int size = mesh.commSize();
        MPI_Comm comm = mesh.comm();

        if ( nOwnedF == 0 )
            return std::vector<char>();

        const int nv = static_cast<int>( mesh.numVertices() );

        // ---- host: vertex gid -> local index (same pattern as refine()) -----
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "mark_curv_hv", nv );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto v_gid = Cabana::slice<VertexField::Gid>( hv );
        std::unordered_map<GlobalId, int> gid2lv;
        gid2lv.reserve( static_cast<std::size_t>( nv ) * 2 );
        for ( int i = 0; i < nv; ++i )
            gid2lv[v_gid( i )] = i;

        // ---- host: owned faces (gids, verts) --------------------------------
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "mark_curv_hf", mesh.numFaces() );
        Cabana::deep_copy( hf, mesh.faces() );
        auto f_gid = Cabana::slice<FaceField::Gid>( hf );
        auto f_verts = Cabana::slice<FaceField::Verts>( hf );

        Kokkos::View<int* [3], memory_space> faceVertLocal(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "mark_curv_face_vert_local" ),
            nOwnedF );
        auto h_fvl = Kokkos::create_mirror_view( faceVertLocal );
        for ( int f = 0; f < nOwnedF; ++f )
            for ( int k = 0; k < 3; ++k )
                h_fvl( f, k ) =
                    gid2lv.at( static_cast<GlobalId>( f_verts( f, k ) ) );
        Kokkos::deep_copy( faceVertLocal, h_fvl );

        // ---- device: per-owned-face outward unit normal ---------------------
        auto pos = Cabana::slice<VertexField::Position>( mesh.vertices() );
        Kokkos::View<Scalar* [3], memory_space> normalDev(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "mark_curv_normal" ),
            nOwnedF );
        {
            TESSERA_SCOPED_TIMER_VERBOSE(
                ::Tessera::Profiling::TIMER_MARK_CURV_KERNEL );
            Kokkos::parallel_for(
                "tessera_mark_curvature_normals",
                Kokkos::RangePolicy<execution_space>( 0, nOwnedF ),
                KOKKOS_LAMBDA( const int f ) {
                    const int a = faceVertLocal( f, 0 );
                    const int b = faceVertLocal( f, 1 );
                    const int c = faceVertLocal( f, 2 );
                    Scalar e1[3], e2[3];
                    for ( int d = 0; d < 3; ++d )
                    {
                        e1[d] = pos( b, d ) - pos( a, d );
                        e2[d] = pos( c, d ) - pos( a, d );
                    }
                    Scalar nx = e1[1] * e2[2] - e1[2] * e2[1];
                    Scalar ny = e1[2] * e2[0] - e1[0] * e2[2];
                    Scalar nz = e1[0] * e2[1] - e1[1] * e2[0];
                    const Scalar len =
                        Kokkos::sqrt( nx * nx + ny * ny + nz * nz );
                    const Scalar inv =
                        len > Scalar( 0 ) ? Scalar( 1 ) / len : Scalar( 0 );
                    normalDev( f, 0 ) = nx * inv;
                    normalDev( f, 1 ) = ny * inv;
                    normalDev( f, 2 ) = nz * inv;
                } );
            Kokkos::fence();
        }
        auto h_norm = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           normalDev );

        // owned face gid -> owned index (mark-request target).
        std::unordered_map<GlobalId, int> gid2of;
        gid2of.reserve( static_cast<std::size_t>( nOwnedF ) * 2 );
        for ( int f = 0; f < nOwnedF; ++f )
            gid2of[f_gid( f )] = f;

        // ---- advertise each owned face's edges + normal to the coordinator --
        std::vector<std::vector<NormalMsg<Scalar>>> toCoord( size );
        for ( int f = 0; f < nOwnedF; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key =
                    makeEdgeKey( f_verts( f, k ), f_verts( f, ( k + 1 ) % 3 ) );
                NormalMsg<Scalar> m;
                m.key = key;
                for ( int d = 0; d < 3; ++d )
                    m.n[d] = h_norm( f, d );
                m.faceGid = f_gid( f );
                m.owner = static_cast<Rank>( R );
                toCoord[detail::edgeCoordRank( key, size )].push_back( m );
            }
        auto got = allToAllV( comm, toCoord );

        // ---- coordinator verdict: exactly two incident faces per edge on a
        //      closed surface; sharp iff n0.n1 < cos(maxAngle). Route a mark to
        //      BOTH incident face owners (both faces of a sharp fold refine). --
        std::map<EdgeKey, std::vector<NormalMsg<Scalar>>> byEdge;
        for ( const auto& m : got.data )
            byEdge[m.key].push_back( m );

        const Scalar cosThresh =
            static_cast<Scalar>( std::cos( static_cast<double>( maxAngle ) ) );
        std::vector<std::vector<GlobalId>> markReq( size );
        for ( auto& kv : byEdge )
        {
            auto& inc = kv.second;
            if ( inc.size() != 2 )
                continue; // closed surface: exactly two incident faces
            Scalar dot = Scalar( 0 );
            for ( int d = 0; d < 3; ++d )
                dot += inc[0].n[d] * inc[1].n[d];
            if ( dot > Scalar( 1 ) )
                dot = Scalar( 1 );
            if ( dot < Scalar( -1 ) )
                dot = Scalar( -1 );
            if ( dot < cosThresh )
            {
                markReq[inc[0].owner].push_back( inc[0].faceGid );
                markReq[inc[1].owner].push_back( inc[1].faceGid );
            }
        }
        auto reqs = allToAllV( comm, markReq );

        // ---- apply marks into the owned-face mask ---------------------------
        std::vector<char> mask( static_cast<std::size_t>( nOwnedF ), 0 );
        for ( const GlobalId g : reqs.data )
        {
            auto it = gid2of.find( g );
            if ( it != gid2of.end() )
                mask[it->second] = 1;
        }
        return mask;
    }
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

// --- 10b: cross-rank gather via edge coordinators ---------------------------

//! Mark BOTH faces incident to any edge whose dihedral bend exceeds `maxAngle`
//! (radians, an absolute bend threshold). Needs both incident faces' normals;
//! the neighbour across a partition boundary is not guaranteed in the 1-ring
//! halo, so the gather is routed through edge coordinators (see
//! detail::markCurvature) -- NOT the halo -- which makes the marked set
//! rank-count independent and boundary-straddle independent. `Dim==2` returns an
//! all-zero mask (a planar surface has no dihedral bend).
template <class Scalar>
struct CurvatureCriterion
{
    Scalar maxAngle;

    template <class MeshT>
    std::vector<char> mark( const MeshT& mesh ) const
    {
        return detail::markCurvature( mesh, maxAngle );
    }
};

//! Uniform entry point: dispatches to the criterion's own mark(). Disabled for
//! arithmetic `Criterion` so a bare scalar threshold resolves to the
//! convenience overload below instead of an ambiguous call.
template <class MeshT, class Criterion,
          class = std::enable_if_t<!std::is_arithmetic<Criterion>::value>>
std::vector<char> markByQuality( const MeshT& mesh, const Criterion& crit )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_MARK_QUALITY );
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
