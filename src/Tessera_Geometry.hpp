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

#ifndef TESSERA_GEOMETRY_HPP
#define TESSERA_GEOMETRY_HPP

#include "Tessera_Fields.hpp"
#include "Tessera_GenerationGuard.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <unordered_map>
#include <utility>

namespace Tessera
{

// ============================================================================
// MeshGeometry — a device-capturable geometry accessor
// ============================================================================
//
// The raw geometric primitives below (faceArea, faceNormalRaw, edgeVector,
// cotangentAtCorner) are pure geometry: no physics, no convention (no normal
// orientation, no area definition, no weight scheme). They are the inputs a
// caller's weight-builder / normal / area functor consumes.
//
// WHY AN ACCESSOR AND NOT `faceArea(mesh, f)`: a `Mesh` holds AoSoAs, an
// MPI_Comm, and a raw generation pointer and is NOT capturable into a
// KOKKOS_LAMBDA; and the face/edge connectivity stores vertex *global ids*
// (FaceField::Verts = GlobalId[3]) with no device-side gid->local map. So the
// primitives cannot take the mesh directly on device. Instead, build a small
// POD accessor once (buildMeshGeometry) that bundles the position slice plus
// per-face / per-edge *local* vertex indices; the accessor is trivially
// captured by value into a kernel and the primitives read it in O(1). This
// mirrors the existing local-index derivation in Tessera_MarkQuality.hpp.
//
// INVALIDATION: the local-index Views and the position slice held here dangle
// after any topology-changing op (distribute/migrate/refine) — exactly like a
// bare slice. MeshGeometry therefore integrates with the generation guard for
// free: it holds the position slice as the GenerationHandle that
// Mesh::vertexSlice() already hands out (Tessera_GenerationGuard.hpp), stamped
// with the mesh generation at build time. Copying the accessor — which is what
// capturing it by value into a KOKKOS_LAMBDA does — copies that handle, whose
// copy constructor re-checks the mesh's live generation host-side and aborts on
// mismatch. The faceVerts/edgeVerts Views are rebuilt together with pos by
// buildMeshGeometry(), so the single pos stamp covers the whole accessor.
// Rebuild after a topology op; haloExchange() is topology-preserving and does
// not invalidate it.
template <class MeshT>
struct MeshGeometry
{
    using memory_space = typename MeshT::memory_space;
    using scalar_type = typename MeshT::scalar_type;
    using raw_pos_slice = decltype( Cabana::slice<VertexField::Position>(
        std::declval<typename MeshT::vertex_aosoa_type&>() ) );
    using pos_handle_type = GenerationHandle<raw_pos_slice>;

    pos_handle_type pos; // generation-guarded position slice
    Kokkos::View<int* [3], memory_space> faceVerts; // local vertex idx per face
    Kokkos::View<int* [2], memory_space> edgeVerts; // local vertex idx per edge

    MeshGeometry() = default;

    KOKKOS_INLINE_FUNCTION
    MeshGeometry( pos_handle_type pos_in,
                  Kokkos::View<int* [3], memory_space> face_verts,
                  Kokkos::View<int* [2], memory_space> edge_verts )
        : pos( std::move( pos_in ) )
        , faceVerts( std::move( face_verts ) )
        , edgeVerts( std::move( edge_verts ) )
    {
    }

    // Explicit device-callable copy ops: copying pos (a GenerationHandle) is
    // what re-validates the accessor host-side (its copy constructor runs the
    // guard); marking these KOKKOS_INLINE_FUNCTION keeps the accessor usable as
    // a by-value kernel capture on device (where the guard compiles out).
    KOKKOS_INLINE_FUNCTION
    MeshGeometry( const MeshGeometry& other )
        : pos( other.pos )
        , faceVerts( other.faceVerts )
        , edgeVerts( other.edgeVerts )
    {
    }

    KOKKOS_INLINE_FUNCTION
    MeshGeometry& operator=( const MeshGeometry& other )
    {
        pos = other.pos;
        faceVerts = other.faceVerts;
        edgeVerts = other.edgeVerts;
        return *this;
    }

    //! Explicit host-side staleness check (copying the accessor also triggers
    //! it via the pos handle's copy constructor).
    void validate() const { pos.validate(); }
};

// ============================================================================
// buildMeshGeometry — derive the device accessor from a mesh
// ============================================================================
//
// Builds per-face and per-edge *local* vertex indices from the gid-based
// connectivity on the host (small, ordered map — the same approach as the
// coarse builder and MarkQuality), deep-copies them to the device, and stamps
// the current mesh generation. Covers ALL locally held entities (owned +
// ghost) so operators over owned vertices can read their full 1-ring, including
// ghost neighbours. A connectivity gid absent from the local vertex set (should
// not occur for the 1-ring closure a 1-deep halo guarantees) is stored as
// invalid_local so downstream primitives skip it rather than read out of range.
//
// The position field is a defaulted template parameter (PosField), so geometry
// can be evaluated on any position-like vertex field of the same type as
// VertexField::Position — e.g. a reference configuration — via
// buildMeshGeometry<userVertexField<K>()>(mesh). The default keeps the existing
// buildMeshGeometry(mesh) behaviour (VertexField::Position) unchanged; the
// chosen field must have the same slice type as Position (Scalar[Dim]) so it
// fits MeshGeometry<MeshT>::pos_handle_type.
template <std::size_t PosField = VertexField::Position, class MeshT>
MeshGeometry<MeshT> buildMeshGeometry( MeshT& mesh )
{
    using memory_space = typename MeshT::memory_space;

    const std::size_t nv = mesh.numVertices();
    const std::size_t ne = mesh.numEdges();
    const std::size_t nf = mesh.numFaces();

    // Host copies of the identity / connectivity needed to build local indices.
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "geom_hv", nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "geom_he", ne );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "geom_hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );

    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto e_v = Cabana::slice<EdgeField::Verts>( he );
    auto f_v = Cabana::slice<FaceField::Verts>( hf );

    std::unordered_map<GlobalId, int> gid2local;
    gid2local.reserve( nv * 2 );
    for ( std::size_t i = 0; i < nv; ++i )
        gid2local[v_gid( i )] = static_cast<int>( i );

    auto lookup = [&]( GlobalId g ) -> int
    {
        auto it = gid2local.find( g );
        return it == gid2local.end() ? invalid_local : it->second;
    };

    Kokkos::View<int* [3], memory_space> faceVerts(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "geom_face_verts" ),
        nf );
    Kokkos::View<int* [2], memory_space> edgeVerts(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "geom_edge_verts" ),
        ne );
    auto h_fv = Kokkos::create_mirror_view( faceVerts );
    auto h_ev = Kokkos::create_mirror_view( edgeVerts );
    for ( std::size_t f = 0; f < nf; ++f )
        for ( int k = 0; k < 3; ++k )
            h_fv( f, k ) = lookup( f_v( f, k ) );
    for ( std::size_t e = 0; e < ne; ++e )
        for ( int j = 0; j < 2; ++j )
            h_ev( e, j ) = lookup( e_v( e, j ) );
    Kokkos::deep_copy( faceVerts, h_fv );
    Kokkos::deep_copy( edgeVerts, h_ev );

    return MeshGeometry<MeshT>(
        mesh.template vertexSlice<PosField>(), faceVerts, edgeVerts );
}

// ============================================================================
// Raw geometric primitives (KOKKOS_INLINE_FUNCTION, no convention)
// ============================================================================
//
// Every primitive indexes the position slice through the per-face / per-edge
// LOCAL vertex indices held in the accessor. Those indices are `invalid_local`
// (-1) for any connectivity entry whose gid was absent from the local vertex
// set (see buildMeshGeometry). Reading the position slice at -1 is an
// out-of-range access, so each primitive guards its indices up front: passing a
// face/edge that carries an invalid corner to a geometric primitive is a
// programming error (a caller must skip such entries), not a runtime condition,
// so the guard aborts with a diagnostic rather than silently returning garbage.
// The check follows the codebase's debug-guard idiom (Tessera_GenerationGuard):
// it uses Kokkos::abort (device-callable, unlike std::abort) and compiles out
// entirely when TESSERA_ENABLE_DEBUG_CHECKS is off, so release builds pay
// nothing on this hot per-element path.
#if defined( TESSERA_ENABLE_DEBUG_CHECKS )
#define TESSERA_GEOM_ASSERT_LOCAL( idx, what )                                 \
    do                                                                         \
    {                                                                          \
        if ( ( idx ) == invalid_local )                                        \
            Kokkos::abort(                                                      \
                "Tessera geometry: " what                                      \
                " received a face/edge with an invalid_local vertex "          \
                "(a boundary/missing connectivity entry). Skip such "          \
                "entries before calling a geometric primitive." );             \
    } while ( 0 )
#else
#define TESSERA_GEOM_ASSERT_LOCAL( idx, what )                                 \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#endif

//! Unsigned triangle area of face f: 1/2 * || (p1 - p0) x (p2 - p0) ||.
template <class MeshT>
KOKKOS_INLINE_FUNCTION typename MeshT::scalar_type
faceArea( const MeshGeometry<MeshT>& g, LocalIndex f )
{
    using Scalar = typename MeshT::scalar_type;
    const int a = g.faceVerts( f, 0 );
    const int b = g.faceVerts( f, 1 );
    const int c = g.faceVerts( f, 2 );
    TESSERA_GEOM_ASSERT_LOCAL( a, "faceArea" );
    TESSERA_GEOM_ASSERT_LOCAL( b, "faceArea" );
    TESSERA_GEOM_ASSERT_LOCAL( c, "faceArea" );
    Scalar e1[3], e2[3];
    for ( int d = 0; d < 3; ++d )
    {
        e1[d] = g.pos( b, d ) - g.pos( a, d );
        e2[d] = g.pos( c, d ) - g.pos( a, d );
    }
    const Scalar nx = e1[1] * e2[2] - e1[2] * e2[1];
    const Scalar ny = e1[2] * e2[0] - e1[0] * e2[2];
    const Scalar nz = e1[0] * e2[1] - e1[1] * e2[0];
    return Scalar( 0.5 ) * Kokkos::sqrt( nx * nx + ny * ny + nz * nz );
}

//! Unnormalized face normal (p1 - p0) x (p2 - p0). Its magnitude is twice the
//! face area; its sign/orientation follows the stored corner order — the
//! OUTWARD convention is the caller's to impose.
template <class MeshT>
KOKKOS_INLINE_FUNCTION void faceNormalRaw( const MeshGeometry<MeshT>& g,
                                           LocalIndex f,
                                           typename MeshT::scalar_type out[3] )
{
    using Scalar = typename MeshT::scalar_type;
    const int a = g.faceVerts( f, 0 );
    const int b = g.faceVerts( f, 1 );
    const int c = g.faceVerts( f, 2 );
    TESSERA_GEOM_ASSERT_LOCAL( a, "faceNormalRaw" );
    TESSERA_GEOM_ASSERT_LOCAL( b, "faceNormalRaw" );
    TESSERA_GEOM_ASSERT_LOCAL( c, "faceNormalRaw" );
    Scalar e1[3], e2[3];
    for ( int d = 0; d < 3; ++d )
    {
        e1[d] = g.pos( b, d ) - g.pos( a, d );
        e2[d] = g.pos( c, d ) - g.pos( a, d );
    }
    out[0] = e1[1] * e2[2] - e1[2] * e2[1];
    out[1] = e1[2] * e2[0] - e1[0] * e2[2];
    out[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

//! Edge vector p[v1] - p[v0] for edge e (direction follows the stored endpoint
//! order; the caller owns any sign convention).
template <class MeshT>
KOKKOS_INLINE_FUNCTION void edgeVector( const MeshGeometry<MeshT>& g,
                                        LocalIndex e,
                                        typename MeshT::scalar_type out[3] )
{
    const int v0 = g.edgeVerts( e, 0 );
    const int v1 = g.edgeVerts( e, 1 );
    TESSERA_GEOM_ASSERT_LOCAL( v0, "edgeVector" );
    TESSERA_GEOM_ASSERT_LOCAL( v1, "edgeVector" );
    for ( int d = 0; d < 3; ++d )
        out[d] = g.pos( v1, d ) - g.pos( v0, d );
}

//! Cotangent of the interior angle at `corner` (0,1,2) of triangle face f.
//! cot(theta) = (u . v) / || u x v ||, where u,v are the two edges emanating
//! from the corner. Triangle geometry only.
template <class MeshT>
KOKKOS_INLINE_FUNCTION typename MeshT::scalar_type
cotangentAtCorner( const MeshGeometry<MeshT>& g, LocalIndex f, int corner )
{
    using Scalar = typename MeshT::scalar_type;
    const int a = g.faceVerts( f, corner );
    const int b = g.faceVerts( f, ( corner + 1 ) % 3 );
    const int c = g.faceVerts( f, ( corner + 2 ) % 3 );
    TESSERA_GEOM_ASSERT_LOCAL( a, "cotangentAtCorner" );
    TESSERA_GEOM_ASSERT_LOCAL( b, "cotangentAtCorner" );
    TESSERA_GEOM_ASSERT_LOCAL( c, "cotangentAtCorner" );
    Scalar u[3], v[3];
    for ( int d = 0; d < 3; ++d )
    {
        u[d] = g.pos( b, d ) - g.pos( a, d );
        v[d] = g.pos( c, d ) - g.pos( a, d );
    }
    const Scalar dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const Scalar cx = u[1] * v[2] - u[2] * v[1];
    const Scalar cy = u[2] * v[0] - u[0] * v[2];
    const Scalar cz = u[0] * v[1] - u[1] * v[0];
    const Scalar cross_norm = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    return cross_norm > Scalar( 0 ) ? dot / cross_norm : Scalar( 0 );
}

#undef TESSERA_GEOM_ASSERT_LOCAL

} // namespace Tessera

#endif // TESSERA_GEOMETRY_HPP
