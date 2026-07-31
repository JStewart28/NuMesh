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

#ifndef TESSERA_REFINE_HPP
#define TESSERA_REFINE_HPP

#include "Tessera_CsrAdjacency.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_RefineClosure.hpp"
#include "Tessera_RefinePolicy.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// Local single-rank red (1->4) refinement
// ============================================================================
//
// refineLocal() applies a red split to each face flagged in `refineFace`: the
// face's three edges are bisected at new midpoint vertices and the face is
// replaced by four child triangles. It runs on ONE rank (no communication);
// the parallel 2:1 conforming balance across partition boundaries is Step 6b,
// which drives refineLocal() after propagating refine flags. Preconditions: the
// mesh is a single-rank / replicated mesh with gid == local index (as produced
// by buildIcosphere); all entities owned. The invariant gid == index is
// preserved (new entities are appended with gid == their new index) -- except for
// FACES in RefinementMode::Conforming, where a closed red face's gid is retired
// into its children and so cannot be reissued; see
// detail::refineLocalConforming().
//
// Child convention (matches the icosphere subdivision so a uniform refineLocal
// reproduces one subdivision level's topology). For a parent face with corner
// vertices (a,b,c) and edge midpoints ab=mid(a,b), bc=mid(b,c), ca=mid(c,a):
//   child 0: (a,  ab, ca)   child 1: (b,  bc, ab)
//   child 2: (c,  ca, bc)   child 3: (ab, bc, ca)   [center]
// Each child inherits parent.level + 1 and a copy of the parent's face user
// fields. Kept (unrefined) faces retain their level and fields.
//
// Midpoint identity: a midpoint is shared by every refined face touching its
// edge and is deduplicated by the edge's structured EdgeKey (the order-invariant
// endpoint-gid pair) — the exact mechanism Step 6b reuses to make a boundary
// edge's midpoint bit-identical on both ranks. In RefinementMode::HangingNode2to1
// a partial mask leaves hanging nodes (a split edge whose other face is
// unrefined); this is the accepted, bounded non-conforming state that Step 6b
// balances. Under a uniform mask the result is fully conforming and Euler V-E+F
// is preserved.
//
// In RefinementMode::Conforming the split above is applied to the persistent RED
// layer and a transient green/blue/red closure is laid over it, so a PARTIAL mask
// also leaves a conforming mesh (Euler V-E+F == 2, every edge with exactly two
// incident faces) -- see detail::refineLocalConforming() below and
// Tessera_RefineClosure.hpp.
//
// Edges, the vertex 1-ring CSR, and the edge/face key side tables are re-derived
// from the new face->vertex connectivity (as in the serial builder); per-edge
// user fields are re-initialized (Milestone 1 carries no edge user state through
// refinement). Vertex user fields on midpoints are interpolated by `policy`;
// face user fields are inherited from the parent.

namespace detail
{

// -- compile-time user-field helpers -----------------------------------------
//
// Iterate the USER members of an AoSoA (absolute indices [UserBegin, size)) at
// compile time. Members may be scalar (rank 0) or 1-D arrays Scalar[N] (rank 1);
// both are handled component-wise. An empty user pack expands to a no-op.

//! Copy one member (index Mabs) from src[si] to dst[di].
template <std::size_t Mabs, class Dst, class Src>
void copyMember( Dst& dst, int di, Src& src, int si )
{
    using MT = typename Dst::member_types;
    using FieldT = typename Cabana::MemberTypeAtIndex<Mabs, MT>::type;
    auto d = Cabana::slice<Mabs>( dst );
    auto s = Cabana::slice<Mabs>( src );
    if constexpr ( std::rank<FieldT>::value == 0 )
    {
        d( di ) = s( si );
    }
    else
    {
        constexpr int C = static_cast<int>( std::extent<FieldT, 0>::value );
        for ( int c = 0; c < C; ++c )
            d( di, c ) = s( si, c );
    }
}

template <std::size_t UserBegin, class Dst, class Src, std::size_t... Js>
void copyUserFieldsImpl( Dst& dst, int di, Src& src, int si,
                         std::index_sequence<Js...> )
{
    ( copyMember<UserBegin + Js>( dst, di, src, si ), ... );
}

//! Copy the N user fields starting at absolute index UserBegin, src[si] ->
//! dst[di]. Prefer this over copyUserFields() wherever the member list may carry
//! non-user members AFTER the user pack -- which is exactly the face AoSoA in
//! RefinementMode::Conforming, whose trailing closure members must not be
//! treated as user fields. Pass N = numFaceUserFields<MeshT::face_user_fields>().
template <std::size_t UserBegin, std::size_t N, class Dst, class Src>
void copyUserFieldsN( Dst& dst, int di, Src& src, int si )
{
    copyUserFieldsImpl<UserBegin>( dst, di, src, si,
                                   std::make_index_sequence<N>{} );
}

//! Copy every member from UserBegin to the end of the tuple, src[si] -> dst[di].
//! Correct only when the user pack is the tuple's suffix (vertices and edges,
//! and faces in RefinementMode::HangingNode2to1).
template <std::size_t UserBegin, class Dst, class Src>
void copyUserFields( Dst& dst, int di, Src& src, int si )
{
    constexpr std::size_t N = Dst::member_types::size - UserBegin;
    copyUserFieldsImpl<UserBegin>( dst, di, src, si,
                                   std::make_index_sequence<N>{} );
}

//! Blend one vertex user member (index Mabs) of the midpoint `mid` from
//! endpoints `a`,`b` via the policy hook (component-wise for array fields).
template <std::size_t Mabs, class VAoSoA, class Policy>
void blendVertexMember( VAoSoA& V, int mid, int a, int b, const Policy& policy )
{
    using MT = typename VAoSoA::member_types;
    using FieldT = typename Cabana::MemberTypeAtIndex<Mabs, MT>::type;
    auto s = Cabana::slice<Mabs>( V );
    if constexpr ( std::rank<FieldT>::value == 0 )
    {
        s( mid ) =
            policy.template interpolateVertexField<Mabs>( s( a ), s( b ) );
    }
    else
    {
        constexpr int C = static_cast<int>( std::extent<FieldT, 0>::value );
        for ( int c = 0; c < C; ++c )
            s( mid, c ) = policy.template interpolateVertexField<Mabs>(
                s( a, c ), s( b, c ) );
    }
}

template <class VAoSoA, class Policy, std::size_t... Js>
void blendVertexUserImpl( VAoSoA& V, int mid, int a, int b,
                          const Policy& policy, std::index_sequence<Js...> )
{
    ( blendVertexMember<VertexField::UserBegin + Js>( V, mid, a, b, policy ),
      ... );
}

//! Blend every vertex user field of midpoint `mid` from endpoints `a`,`b`.
template <class VAoSoA, class Policy>
void blendVertexUserFields( VAoSoA& V, int mid, int a, int b,
                            const Policy& policy )
{
    constexpr std::size_t N =
        VAoSoA::member_types::size - VertexField::UserBegin;
    blendVertexUserImpl( V, mid, a, b, policy, std::make_index_sequence<N>{} );
}

// -- cross-AoSoA variant (distributed refine: endpoints live in the current
//    vertex AoSoA, the midpoint is written into a freshly-built one) -----------

//! Blend one vertex user member (index Mabs) of midpoint `Dst[di]` from
//! endpoints `Src[a]`,`Src[b]` via the policy hook (component-wise for arrays).
template <std::size_t Mabs, class Dst, class Src, class Policy>
void blendVertexMemberCross( Dst& dst, int di, Src& src, int a, int b,
                             const Policy& policy )
{
    using MT = typename Dst::member_types;
    using FieldT = typename Cabana::MemberTypeAtIndex<Mabs, MT>::type;
    auto d = Cabana::slice<Mabs>( dst );
    auto s = Cabana::slice<Mabs>( src );
    if constexpr ( std::rank<FieldT>::value == 0 )
    {
        d( di ) =
            policy.template interpolateVertexField<Mabs>( s( a ), s( b ) );
    }
    else
    {
        constexpr int C = static_cast<int>( std::extent<FieldT, 0>::value );
        for ( int c = 0; c < C; ++c )
            d( di, c ) = policy.template interpolateVertexField<Mabs>(
                s( a, c ), s( b, c ) );
    }
}

template <class Dst, class Src, class Policy, std::size_t... Js>
void blendVertexUserCrossImpl( Dst& dst, int di, Src& src, int a, int b,
                               const Policy& policy,
                               std::index_sequence<Js...> )
{
    ( blendVertexMemberCross<VertexField::UserBegin + Js>( dst, di, src, a, b,
                                                           policy ),
      ... );
}

//! Blend every vertex user field of midpoint `dst[di]` from endpoints
//! `src[a]`,`src[b]` (dst and src may be distinct AoSoAs).
template <class Dst, class Src, class Policy>
void blendVertexUserCross( Dst& dst, int di, Src& src, int a, int b,
                           const Policy& policy )
{
    constexpr std::size_t N = Dst::member_types::size - VertexField::UserBegin;
    blendVertexUserCrossImpl( dst, di, src, a, b, policy,
                              std::make_index_sequence<N>{} );
}

//! RefinementMode::HangingNode2to1 implementation of refineLocal(). Called
//! through the refineLocal() dispatcher below; see the header comment for the
//! conventions and preconditions.
template <class MeshT, class Policy>
void refineLocalHangingNode( MeshT& mesh, const std::vector<char>& refineFace,
                             const Policy& policy )
{
    using memory_space = typename MeshT::memory_space;
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;

    const int nv = static_cast<int>( mesh.numVertices() );
    const int nf = static_cast<int>( mesh.numFaces() );
    const Rank owner = static_cast<Rank>( mesh.rank() );

    // ---- host copies of the current vertices and faces ---------------------
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );
    auto f_v = Cabana::slice<FaceField::Verts>( hf );
    auto f_lev = Cabana::slice<FaceField::Level>( hf );

    // ---- assign a midpoint vertex per split edge (deduped by EdgeKey) ------
    // A split edge is any edge of a refined face. Its midpoint gid == its new
    // local index, appended after the existing vertices.
    std::map<EdgeKey, int> midpointOf;      // edge key -> new vertex index
    std::vector<std::array<int, 2>> midEnd; // new vertex -> (endpoint a, b)
    auto midpoint = [&]( int a, int b ) -> int
    {
        const EdgeKey key = makeEdgeKey( static_cast<GlobalId>( a ),
                                         static_cast<GlobalId>( b ) );
        auto it = midpointOf.find( key );
        if ( it != midpointOf.end() )
            return it->second;
        const int idx = nv + static_cast<int>( midEnd.size() );
        midpointOf.emplace( key, idx );
        midEnd.push_back( { a, b } );
        return idx;
    };

    // New face -> corner vertices, parent face index, and level. Kept faces and
    // the four children are appended in a single pass over the parent faces.
    std::vector<std::array<int, 3>> newFaceV;
    std::vector<int> newFaceParent;
    std::vector<Level> newFaceLevel;
    newFaceV.reserve( nf );
    for ( int f = 0; f < nf; ++f )
    {
        const int a = static_cast<int>( f_v( f, 0 ) );
        const int b = static_cast<int>( f_v( f, 1 ) );
        const int c = static_cast<int>( f_v( f, 2 ) );
        if ( !refineFace[f] )
        {
            newFaceV.push_back( { a, b, c } );
            newFaceParent.push_back( f );
            newFaceLevel.push_back( f_lev( f ) );
            continue;
        }
        const int ab = midpoint( a, b );
        const int bc = midpoint( b, c );
        const int ca = midpoint( c, a );
        const std::array<int, 3> children[4] = {
            { a, ab, ca }, { b, bc, ab }, { c, ca, bc }, { ab, bc, ca } };
        const Level clev = static_cast<Level>( f_lev( f ) + 1 );
        for ( const auto& ch : children )
        {
            newFaceV.push_back( ch );
            newFaceParent.push_back( f );
            newFaceLevel.push_back( clev );
        }
    }
    const int newNv = nv + static_cast<int>( midEnd.size() );
    const int newNf = static_cast<int>( newFaceV.size() );

    // ---- build the new vertex AoSoA (old verts copied, midpoints blended) --
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> lv(
        "lv", newNv );
    {
        auto gid = Cabana::slice<VertexField::Gid>( lv );
        auto own = Cabana::slice<VertexField::Owner>( lv );
        auto flg = Cabana::slice<VertexField::Flags>( lv );
        auto pos = Cabana::slice<VertexField::Position>( lv );
        auto o_flg = Cabana::slice<VertexField::Flags>( hv );
        for ( int i = 0; i < nv; ++i ) // carry existing vertices unchanged
        {
            gid( i ) = static_cast<GlobalId>( i );
            own( i ) = owner;
            flg( i ) = o_flg( i );
            for ( int d = 0; d < Dim; ++d )
                pos( i, d ) = v_pos( i, d );
            detail::copyUserFields<VertexField::UserBegin>( lv, i, hv, i );
        }
        for ( int m = 0; m < static_cast<int>( midEnd.size() ); ++m )
        {
            const int idx = nv + m;
            const int a = midEnd[m][0];
            const int b = midEnd[m][1];
            gid( idx ) = static_cast<GlobalId>( idx );
            own( idx ) = owner;
            flg( idx ) = 0;
            Scalar pa[Dim], pb[Dim], pm[Dim];
            for ( int d = 0; d < Dim; ++d )
            {
                pa[d] = v_pos( a, d );
                pb[d] = v_pos( b, d );
            }
            policy.interpolatePosition( pm, pa, pb, Dim );
            for ( int d = 0; d < Dim; ++d )
                pos( idx, d ) = pm[d];
            detail::blendVertexUserFields( lv, idx, a, b, policy );
        }
    }
    mesh.resizeVertices( newNv );
    Cabana::deep_copy( mesh.vertices(), lv );

    // ---- re-derive unique edges + connectivity from new faces --------------
    // Same derivation as the serial builder: dedup edges by EdgeKey, record
    // face->edge (convention e[k] = edge(v[k], v[(k+1)%3])) and edge->face.
    std::map<EdgeKey, int> edge_of;
    std::vector<std::array<GlobalId, 2>> ep; // edge endpoint gids (sorted)
    std::vector<std::array<GlobalId, 2>> ef; // edge incident face gids
    std::vector<std::array<int, 3>> faceEdge( newNf );
    for ( int f = 0; f < newNf; ++f )
    {
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId a = static_cast<GlobalId>( newFaceV[f][k] );
            const GlobalId b =
                static_cast<GlobalId>( newFaceV[f][( k + 1 ) % 3] );
            const EdgeKey key = makeEdgeKey( a, b );
            int idx;
            auto it = edge_of.find( key );
            if ( it == edge_of.end() )
            {
                idx = static_cast<int>( ep.size() );
                edge_of.emplace( key, idx );
                ep.push_back( { key.id[0], key.id[1] } );
                ef.push_back( { invalid_gid, invalid_gid } );
            }
            else
            {
                idx = it->second;
            }
            faceEdge[f][k] = idx;
            if ( ef[idx][0] == invalid_gid )
                ef[idx][0] = static_cast<GlobalId>( f );
            else
                ef[idx][1] = static_cast<GlobalId>( f );
        }
    }
    const int newNe = static_cast<int>( ep.size() );

    // ---- new edge AoSoA (level = min incident face level) ------------------
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> le(
            "le", newNe );
        auto gid = Cabana::slice<EdgeField::Gid>( le );
        auto own = Cabana::slice<EdgeField::Owner>( le );
        auto lev = Cabana::slice<EdgeField::Level>( le );
        auto verts = Cabana::slice<EdgeField::Verts>( le );
        auto faces = Cabana::slice<EdgeField::Faces>( le );
        for ( int e = 0; e < newNe; ++e )
        {
            gid( e ) = static_cast<GlobalId>( e );
            own( e ) = owner;
            const int f0 = static_cast<int>( ef[e][0] );
            Level lv0 = newFaceLevel[f0];
            if ( ef[e][1] != invalid_gid )
            {
                const Level lv1 = newFaceLevel[static_cast<int>( ef[e][1] )];
                if ( lv1 < lv0 )
                    lv0 = lv1;
            }
            lev( e ) = lv0;
            verts( e, 0 ) = ep[e][0];
            verts( e, 1 ) = ep[e][1];
            faces( e, 0 ) = ef[e][0];
            faces( e, 1 ) = ef[e][1];
        }
        mesh.resizeEdges( newNe );
        Cabana::deep_copy( mesh.edges(), le );
    }

    // ---- new face AoSoA (children inherit parent user fields) --------------
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> lf(
            "lf", newNf );
        auto gid = Cabana::slice<FaceField::Gid>( lf );
        auto own = Cabana::slice<FaceField::Owner>( lf );
        auto lev = Cabana::slice<FaceField::Level>( lf );
        auto verts = Cabana::slice<FaceField::Verts>( lf );
        auto edges = Cabana::slice<FaceField::Edges>( lf );
        for ( int f = 0; f < newNf; ++f )
        {
            gid( f ) = static_cast<GlobalId>( f );
            own( f ) = owner;
            lev( f ) = newFaceLevel[f];
            for ( int k = 0; k < 3; ++k )
            {
                verts( f, k ) = static_cast<GlobalId>( newFaceV[f][k] );
                edges( f, k ) = static_cast<GlobalId>( faceEdge[f][k] );
            }
            copyUserFieldsN<
                FaceField::UserBegin,
                numFaceUserFields<typename MeshT::face_user_fields>()>(
                lf, f, hf, newFaceParent[f] );
        }
        mesh.resizeFaces( newNf );
        Cabana::deep_copy( mesh.faces(), lf );
    }

    // INVALIDATION: setOwnedCounts() above and the key-View/CSR rebuild below
    // reallocate and reassign this mesh's storage, invalidating every
    // slice/CSR/key-View handed out before this call. Re-slice from the mesh
    // after refine() returns.
    mesh.setOwnedCounts( newNv, newNe, newNf );

    // ---- rebuild key side tables -------------------------------------------
    {
        Kokkos::View<EdgeKey*, memory_space> ek(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
            newNe );
        auto h_ek = Kokkos::create_mirror_view( ek );
        for ( int e = 0; e < newNe; ++e )
            h_ek( e ) = makeEdgeKey( ep[e][0], ep[e][1] );
        Kokkos::deep_copy( ek, h_ek );
        mesh.setEdgeKeys( ek );

        Kokkos::View<FaceKey*, memory_space> fk(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
            newNf );
        auto h_fk = Kokkos::create_mirror_view( fk );
        for ( int f = 0; f < newNf; ++f )
            h_fk( f ) = makeFaceKey( static_cast<GlobalId>( newFaceV[f][0] ),
                                     static_cast<GlobalId>( newFaceV[f][1] ),
                                     static_cast<GlobalId>( newFaceV[f][2] ) );
        Kokkos::deep_copy( fk, h_fk );
        mesh.setFaceKeys( fk );
    }

    // ---- rebuild vertex 1-ring CSR (vertex -> faces, vertex -> edges) ------
    {
        std::vector<int> off( newNv + 1, 0 );
        for ( int f = 0; f < newNf; ++f )
            for ( int k = 0; k < 3; ++k )
                ++off[newFaceV[f][k] + 1];
        for ( int i = 0; i < newNv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int f = 0; f < newNf; ++f )
            for ( int k = 0; k < 3; ++k )
                nbr[cur[newFaceV[f][k]]++] = static_cast<LocalIndex>( f );
        mesh.rebuildVertexFaces( off, nbr, "vertex_faces" );
    }
    {
        std::vector<int> off( newNv + 1, 0 );
        for ( int e = 0; e < newNe; ++e )
        {
            ++off[static_cast<int>( ep[e][0] ) + 1];
            ++off[static_cast<int>( ep[e][1] ) + 1];
        }
        for ( int i = 0; i < newNv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int e = 0; e < newNe; ++e )
            for ( int j = 0; j < 2; ++j )
                nbr[cur[static_cast<int>( ep[e][j] )]++] =
                    static_cast<LocalIndex>( e );
        mesh.rebuildVertexEdges( off, nbr, "vertex_edges" );
    }
}

//! RefinementMode::Conforming implementation of refineLocal(): the same red
//! 1->4 split as above, wrapped in the transient red-green-blue closure of
//! Tessera_RefineClosure.hpp. Called through the refineLocal() dispatcher below.
//!
//! Sequence (all local, no communication -- closure never needs any):
//!   0.  UN-CLOSE  the visible faces to the persistent red layer.
//!   0b. TRANSLATE `refineFace` (indexed by VISIBLE faces, as every caller and
//!       markByQuality produce it) to a red-face mask: a red parent is marked iff
//!       any of its closure children was.
//!   1.  RED SPLIT the red layer 1->4, exactly as refineLocalHangingNode() does.
//!   3b. CLOSE every kept red face whose edges a neighbour just bisected.
//! Then edges, keys, and the vertex 1-ring CSR are re-derived from the VISIBLE
//! face list, which is what mesh.faces() and every consumer see.
//!
//! Differences from the hanging-node path, both deliberate:
//!
//!   * VERTEX and EDGE gids remain == their local index; FACE gids do NOT. A
//!     closed red face's gid is retired into its children's ClosureParent field,
//!     so it must not be reissued to a live face -- red gids therefore persist
//!     across the call and closure children are allocated above the current max.
//!     Live face gids are consequently sparse (already true of the distributed
//!     refine(); see Tessera_RefineParallel.hpp step 3b).
//!
//!   * refineLocal() enforces NO 2:1 level balance (neither mode does -- that is
//!     refine()'s Phase 1). One call from a balanced mesh can bisect each edge of
//!     a kept face at most once, so the closure patterns apply; a SEQUENCE of
//!     adaptive refineLocal() calls can drive a >2:1 jump, at which point an edge
//!     carries more than one midpoint and no fixed pattern applies. closeFaces()
//!     detects exactly that and aborts loudly rather than emitting a mesh that
//!     still has hanging nodes. Use refine() for repeated adaptive rounds.
//!
//! Preconditions are otherwise refineLocalHangingNode()'s: a single-rank mesh,
//! all entities owned, vertex gid == local index, and the closure bookkeeping
//! members initialized (initClosureFaceMembers(), which the builder calls).
template <class MeshT, class Policy>
void refineLocalConforming( MeshT& mesh, const std::vector<char>& refineFace,
                            const Policy& policy )
{
    using memory_space = typename MeshT::memory_space;
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    constexpr std::size_t kNumFaceUser =
        numFaceUserFields<typename MeshT::face_user_fields>();

    const int nv = static_cast<int>( mesh.numVertices() );
    const int nf = static_cast<int>( mesh.numFaces() );
    const Rank owner = static_cast<Rank>( mesh.rank() );

    // ---- host copies of the current vertices and visible faces -------------
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );

    // ---- step 0: un-close to the red layer; 0b: translate the mask ----------
    const std::vector<VisibleFace> visible = readVisibleFaces<MeshT>( hf, nf );
    const UncloseResult un = unclose( visible );
    const std::vector<char> redMask = translateMask( refineFace, un );
    const int nRed = static_cast<int>( un.red.size() );

    // ---- step 1: red 1->4 split of the red layer ---------------------------
    // A midpoint is deduplicated by the bisected edge's EdgeKey and, since
    // vertex gid == local index here, its gid is its new local index.
    std::map<EdgeKey, GlobalId> midpointOf;
    std::vector<std::array<int, 2>> midEnd; // new vertex -> (endpoint a, b)
    auto midpoint = [&]( GlobalId a, GlobalId b ) -> GlobalId
    {
        const EdgeKey key = makeEdgeKey( a, b );
        auto it = midpointOf.find( key );
        if ( it != midpointOf.end() )
            return it->second;
        const GlobalId g =
            static_cast<GlobalId>( nv + static_cast<int>( midEnd.size() ) );
        midpointOf.emplace( key, g );
        midEnd.push_back(
            { static_cast<int>( a ), static_cast<int>( b ) } ); // gid == index
        return g;
    };

    // Fresh red children are numbered above the current max red gid so a retired
    // parent gid (still referenced by its former children) can never collide.
    GlobalId maxRedGid = 0;
    for ( const RedFace& p : un.red )
        maxRedGid = std::max( maxRedGid, p.gid );
    GlobalId nextRedGid = maxRedGid + 1;

    std::vector<RedFace> newRed;  // the post-split red layer
    std::vector<char> freshChild; // parallel: created by THIS round's split
    std::vector<int> newRedSrcHf; // parallel: hf row holding the user fields
    newRed.reserve( static_cast<std::size_t>( nRed ) );
    freshChild.reserve( static_cast<std::size_t>( nRed ) );
    newRedSrcHf.reserve( static_cast<std::size_t>( nRed ) );
    for ( int r = 0; r < nRed; ++r )
    {
        const RedFace& p = un.red[r];
        const int src = un.sourceVisible[r];
        if ( !redMask[r] )
        {
            newRed.push_back( p );
            freshChild.push_back( 0 );
            newRedSrcHf.push_back( src );
            continue;
        }
        const GlobalId ab = midpoint( p.v[0], p.v[1] );
        const GlobalId bc = midpoint( p.v[1], p.v[2] );
        const GlobalId ca = midpoint( p.v[2], p.v[0] );
        const std::array<GlobalId, 3> ch[4] = { { p.v[0], ab, ca },
                                                { p.v[1], bc, ab },
                                                { p.v[2], ca, bc },
                                                { ab, bc, ca } };
        const Level clev = static_cast<Level>( p.level + 1 );
        for ( const auto& q : ch )
        {
            RedFace f;
            for ( int k = 0; k < 3; ++k )
                f.v[k] = q[k];
            f.gid = nextRedGid++;
            f.level = clev;
            newRed.push_back( f );
            freshChild.push_back( 1 );
            newRedSrcHf.push_back( src );
        }
    }

    // ---- step 3b: close every kept red face with a bisected edge -----------
    const CloseResult cl =
        closeFaces( newRed, midpointOf, nextRedGid, freshChild );
    const std::vector<VisibleFace>& newVis = cl.visible;
    const int newNv = nv + static_cast<int>( midEnd.size() );
    const int newNf = static_cast<int>( newVis.size() );

    // Visible-face corner LOCAL indices (vertex gid == index throughout).
    std::vector<std::array<int, 3>> newFaceV( newNf );
    for ( int f = 0; f < newNf; ++f )
        for ( int k = 0; k < 3; ++k )
            newFaceV[f][k] = static_cast<int>( newVis[f].v[k] );

    // ---- new vertex AoSoA (old verts copied, midpoints blended) ------------
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            lv( "lv", newNv );
        auto gid = Cabana::slice<VertexField::Gid>( lv );
        auto own = Cabana::slice<VertexField::Owner>( lv );
        auto flg = Cabana::slice<VertexField::Flags>( lv );
        auto pos = Cabana::slice<VertexField::Position>( lv );
        auto o_flg = Cabana::slice<VertexField::Flags>( hv );
        for ( int i = 0; i < nv; ++i ) // carry existing vertices unchanged
        {
            gid( i ) = static_cast<GlobalId>( i );
            own( i ) = owner;
            flg( i ) = o_flg( i );
            for ( int d = 0; d < Dim; ++d )
                pos( i, d ) = v_pos( i, d );
            detail::copyUserFields<VertexField::UserBegin>( lv, i, hv, i );
        }
        for ( int m = 0; m < static_cast<int>( midEnd.size() ); ++m )
        {
            const int idx = nv + m;
            const int a = midEnd[m][0];
            const int b = midEnd[m][1];
            gid( idx ) = static_cast<GlobalId>( idx );
            own( idx ) = owner;
            flg( idx ) = 0;
            Scalar pa[Dim], pb[Dim], pm[Dim];
            for ( int d = 0; d < Dim; ++d )
            {
                pa[d] = v_pos( a, d );
                pb[d] = v_pos( b, d );
            }
            policy.interpolatePosition( pm, pa, pb, Dim );
            for ( int d = 0; d < Dim; ++d )
                pos( idx, d ) = pm[d];
            detail::blendVertexUserFields( lv, idx, a, b, policy );
        }
        mesh.resizeVertices( newNv );
        Cabana::deep_copy( mesh.vertices(), lv );
    }

    // ---- re-derive unique edges + connectivity from the VISIBLE faces ------
    // Same derivation as the serial builder, except the incident-face field
    // stores face GIDS (which are no longer local indices; see above).
    std::map<EdgeKey, int> edge_of;
    std::vector<std::array<GlobalId, 2>> ep; // edge endpoint gids (sorted)
    std::vector<std::array<int, 2>> efLocal; // incident local face indices
    std::vector<std::array<int, 3>> faceEdge( newNf );
    for ( int f = 0; f < newNf; ++f )
    {
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId a = newVis[f].v[k];
            const GlobalId b = newVis[f].v[( k + 1 ) % 3];
            const EdgeKey key = makeEdgeKey( a, b );
            int idx;
            auto it = edge_of.find( key );
            if ( it == edge_of.end() )
            {
                idx = static_cast<int>( ep.size() );
                edge_of.emplace( key, idx );
                ep.push_back( { key.id[0], key.id[1] } );
                efLocal.push_back( { -1, -1 } );
            }
            else
            {
                idx = it->second;
            }
            faceEdge[f][k] = idx;
            if ( efLocal[idx][0] < 0 )
                efLocal[idx][0] = f;
            else
                efLocal[idx][1] = f;
        }
    }
    const int newNe = static_cast<int>( ep.size() );

    // ---- new edge AoSoA (level = min incident visible face level) ----------
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> le(
            "le", newNe );
        auto gid = Cabana::slice<EdgeField::Gid>( le );
        auto own = Cabana::slice<EdgeField::Owner>( le );
        auto lev = Cabana::slice<EdgeField::Level>( le );
        auto verts = Cabana::slice<EdgeField::Verts>( le );
        auto faces = Cabana::slice<EdgeField::Faces>( le );
        for ( int e = 0; e < newNe; ++e )
        {
            gid( e ) = static_cast<GlobalId>( e );
            own( e ) = owner;
            const int f0 = efLocal[e][0];
            Level lv0 = newVis[f0].level;
            if ( efLocal[e][1] >= 0 )
                lv0 = std::min( lv0, newVis[efLocal[e][1]].level );
            lev( e ) = lv0;
            verts( e, 0 ) = ep[e][0];
            verts( e, 1 ) = ep[e][1];
            faces( e, 0 ) = newVis[f0].gid;
            faces( e, 1 ) =
                efLocal[e][1] >= 0 ? newVis[efLocal[e][1]].gid : invalid_gid;
        }
        mesh.resizeEdges( newNe );
        Cabana::deep_copy( mesh.edges(), le );
    }

    // ---- new face AoSoA (children inherit the parent's user fields) --------
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> lf(
            "lf", newNf );
        auto gid = Cabana::slice<FaceField::Gid>( lf );
        auto own = Cabana::slice<FaceField::Owner>( lf );
        auto lev = Cabana::slice<FaceField::Level>( lf );
        auto verts = Cabana::slice<FaceField::Verts>( lf );
        auto edges = Cabana::slice<FaceField::Edges>( lf );
        for ( int f = 0; f < newNf; ++f )
        {
            gid( f ) = newVis[f].gid;
            own( f ) = owner;
            lev( f ) = newVis[f].level;
            for ( int k = 0; k < 3; ++k )
            {
                verts( f, k ) = newVis[f].v[k];
                edges( f, k ) = static_cast<GlobalId>( faceEdge[f][k] );
            }
            // User fields chase the two-hop provenance: visible face -> its red
            // face (cl.sourceRed) -> the pre-call visible row that red face's
            // fields came from (newRedSrcHf).
            detail::copyUserFieldsN<FaceField::UserBegin, kNumFaceUser>(
                lf, f, hf, newRedSrcHf[cl.sourceRed[f]] );
            writeClosureFace<MeshT>( lf, f, newVis[f] );
        }
        mesh.resizeFaces( newNf );
        Cabana::deep_copy( mesh.faces(), lf );
    }

    // INVALIDATION: setOwnedCounts() below and the key-View/CSR rebuild after it
    // reallocate and reassign this mesh's storage, invalidating every
    // slice/CSR/key-View handed out before this call. Re-slice from the mesh
    // after refineLocal() returns.
    mesh.setOwnedCounts( newNv, newNe, newNf );

    // ---- rebuild key side tables -------------------------------------------
    {
        Kokkos::View<EdgeKey*, memory_space> ek(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
            newNe );
        auto h_ek = Kokkos::create_mirror_view( ek );
        for ( int e = 0; e < newNe; ++e )
            h_ek( e ) = makeEdgeKey( ep[e][0], ep[e][1] );
        Kokkos::deep_copy( ek, h_ek );
        mesh.setEdgeKeys( ek );

        Kokkos::View<FaceKey*, memory_space> fk(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
            newNf );
        auto h_fk = Kokkos::create_mirror_view( fk );
        for ( int f = 0; f < newNf; ++f )
            h_fk( f ) =
                makeFaceKey( newVis[f].v[0], newVis[f].v[1], newVis[f].v[2] );
        Kokkos::deep_copy( fk, h_fk );
        mesh.setFaceKeys( fk );
    }

    // ---- rebuild vertex 1-ring CSR (vertex -> faces, vertex -> edges) ------
    {
        std::vector<int> off( newNv + 1, 0 );
        for ( int f = 0; f < newNf; ++f )
            for ( int k = 0; k < 3; ++k )
                ++off[newFaceV[f][k] + 1];
        for ( int i = 0; i < newNv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int f = 0; f < newNf; ++f )
            for ( int k = 0; k < 3; ++k )
                nbr[cur[newFaceV[f][k]]++] = static_cast<LocalIndex>( f );
        mesh.rebuildVertexFaces( off, nbr, "vertex_faces" );
    }
    {
        std::vector<int> off( newNv + 1, 0 );
        for ( int e = 0; e < newNe; ++e )
        {
            ++off[static_cast<int>( ep[e][0] ) + 1];
            ++off[static_cast<int>( ep[e][1] ) + 1];
        }
        for ( int i = 0; i < newNv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int e = 0; e < newNe; ++e )
            for ( int j = 0; j < 2; ++j )
                nbr[cur[static_cast<int>( ep[e][j] )]++] =
                    static_cast<LocalIndex>( e );
        mesh.rebuildVertexEdges( off, nbr, "vertex_edges" );
    }
}

} // namespace detail

//! Red (1->4) refine every face flagged in `refineFace` (indexed by local face
//! index) on a single rank. See the header comment for conventions/preconditions.
//!
//! Dispatches on MeshT::refinement_mode: RefinementMode::Conforming un-closes the
//! transient closure layer before the red split and re-closes afterwards, so the
//! resulting visible mesh has no hanging nodes -- see
//! detail::refineLocalConforming() for the sequence and the two contract
//! differences (face gids are no longer local indices; a *sequence* of adaptive
//! refineLocal() calls can violate the 2:1 precondition the closure needs, and
//! is detected loudly).
template <class MeshT,
          class Policy = DefaultRefinePolicy<typename MeshT::scalar_type>>
void refineLocal( MeshT& mesh, const std::vector<char>& refineFace,
                  const Policy& policy = Policy{} )
{
    if constexpr ( MeshT::refinement_mode == RefinementMode::Conforming )
    {
        detail::refineLocalConforming( mesh, refineFace, policy );
    }
    else
    {
        detail::refineLocalHangingNode( mesh, refineFace, policy );
    }
}

} // namespace Tessera

#endif // TESSERA_REFINE_HPP
