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

#ifndef TESSERA_MESH_BUILDER_HPP
#define TESSERA_MESH_BUILDER_HPP

#include "Tessera_Icosphere.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <map>
#include <vector>

namespace Tessera
{

// ============================================================================
// Serial connectivity builder
// ============================================================================
//
// buildFromTriangleSoup() turns a triangle soup (vertex positions + per-face
// vertex indices) into a fully-connected single-rank mesh:
//   - vertices  : gid (= soup index here), owner, flags, position
//   - edges     : gid, owner, level, endpoint vertex gids v[2], incident face
//                 gids f[2] (invalid_gid on a boundary; a closed sphere has none)
//   - faces     : gid, owner, level, corner vertex gids v[3], edge gids e[3]
//                 with the convention e[k] = edge (v[k], v[(k+1)%3])
//   - CSR 1-ring: vertex -> incident faces, vertex -> incident edges
//   - key side tables: edgeKeys, faceKeys
//
// Derivation runs on the host with ordered maps (the coarse mesh is small and is
// generated identically/replicated on every rank), then the results are
// deep-copied into the mesh's (possibly device) AoSoAs and Views. In this serial
// step gid == local index; Step 5 replaces the direct gid-as-index use with a
// gid->local map once entities are distributed.

//! Build full mesh connectivity from a triangle soup. Overwrites any existing
//! mesh contents. Single rank (gid == local index).
template <class MeshT, class Scalar>
void buildFromTriangleSoup( MeshT& mesh, const TriangleSoup<Scalar>& soup )
{
    static_assert( MeshT::dim == 3,
                   "buildFromTriangleSoup expects a 3D embedding (Dim == 3)" );
    using memory_space = typename MeshT::memory_space;

    const std::size_t nv = soup.numVertices();
    const std::size_t nf = soup.numFaces();
    const Rank owner = static_cast<Rank>( mesh.rank() );

    // ---- vertices ----------------------------------------------------------
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", nv );
        auto gid = Cabana::slice<VertexField::Gid>( hv );
        auto own = Cabana::slice<VertexField::Owner>( hv );
        auto flg = Cabana::slice<VertexField::Flags>( hv );
        auto pos = Cabana::slice<VertexField::Position>( hv );
        for ( std::size_t i = 0; i < nv; ++i )
        {
            gid( i ) = static_cast<GlobalId>( i );
            own( i ) = owner;
            flg( i ) = 0;
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) = soup.positions[3 * i + d];
        }
        mesh.resizeVertices( nv );
        Cabana::deep_copy( mesh.vertices(), hv );
    }

    // ---- derive unique edges + face->edge and edge->face connectivity ------
    std::map<EdgeKey, int> edge_of;
    std::vector<std::array<GlobalId, 2>> ep; // edge endpoint gids (sorted)
    std::vector<std::array<GlobalId, 2>> ef; // edge incident face gids
    std::vector<std::array<int, 3>> face_edges( nf );

    for ( std::size_t f = 0; f < nf; ++f )
    {
        const int tv[3] = { soup.triangles[3 * f + 0],
                            soup.triangles[3 * f + 1],
                            soup.triangles[3 * f + 2] };
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId a = static_cast<GlobalId>( tv[k] );
            const GlobalId b = static_cast<GlobalId>( tv[( k + 1 ) % 3] );
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
            face_edges[f][k] = idx;
            if ( ef[idx][0] == invalid_gid )
                ef[idx][0] = static_cast<GlobalId>( f );
            else
                ef[idx][1] = static_cast<GlobalId>( f );
        }
    }
    const std::size_t ne = ep.size();

    // ---- edges -------------------------------------------------------------
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
            "he", ne );
        auto gid = Cabana::slice<EdgeField::Gid>( he );
        auto own = Cabana::slice<EdgeField::Owner>( he );
        auto lev = Cabana::slice<EdgeField::Level>( he );
        auto verts = Cabana::slice<EdgeField::Verts>( he );
        auto faces = Cabana::slice<EdgeField::Faces>( he );
        for ( std::size_t e = 0; e < ne; ++e )
        {
            gid( e ) = static_cast<GlobalId>( e );
            own( e ) = owner;
            lev( e ) = 0;
            verts( e, 0 ) = ep[e][0];
            verts( e, 1 ) = ep[e][1];
            faces( e, 0 ) = ef[e][0];
            faces( e, 1 ) = ef[e][1];
        }
        mesh.resizeEdges( ne );
        Cabana::deep_copy( mesh.edges(), he );
    }

    // ---- faces -------------------------------------------------------------
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf", nf );
        auto gid = Cabana::slice<FaceField::Gid>( hf );
        auto own = Cabana::slice<FaceField::Owner>( hf );
        auto lev = Cabana::slice<FaceField::Level>( hf );
        auto verts = Cabana::slice<FaceField::Verts>( hf );
        auto edges = Cabana::slice<FaceField::Edges>( hf );
        for ( std::size_t f = 0; f < nf; ++f )
        {
            gid( f ) = static_cast<GlobalId>( f );
            own( f ) = owner;
            lev( f ) = 0;
            for ( int k = 0; k < 3; ++k )
            {
                verts( f, k ) =
                    static_cast<GlobalId>( soup.triangles[3 * f + k] );
                edges( f, k ) = static_cast<GlobalId>( face_edges[f][k] );
            }
        }
        mesh.resizeFaces( nf );
        Cabana::deep_copy( mesh.faces(), hf );
    }

    // ---- canonical-key side tables -----------------------------------------
    {
        Kokkos::View<EdgeKey*, memory_space> ek(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
            ne );
        auto h_ek = Kokkos::create_mirror_view( ek );
        for ( std::size_t e = 0; e < ne; ++e )
            h_ek( e ) = makeEdgeKey( ep[e][0], ep[e][1] );
        Kokkos::deep_copy( ek, h_ek );
        mesh.setEdgeKeys( ek );

        Kokkos::View<FaceKey*, memory_space> fk(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
            nf );
        auto h_fk = Kokkos::create_mirror_view( fk );
        for ( std::size_t f = 0; f < nf; ++f )
            h_fk( f ) = makeFaceKey(
                static_cast<GlobalId>( soup.triangles[3 * f + 0] ),
                static_cast<GlobalId>( soup.triangles[3 * f + 1] ),
                static_cast<GlobalId>( soup.triangles[3 * f + 2] ) );
        Kokkos::deep_copy( fk, h_fk );
        mesh.setFaceKeys( fk );
    }

    // ---- CSR: vertex -> incident faces -------------------------------------
    {
        std::vector<int> off( nv + 1, 0 );
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
                ++off[soup.triangles[3 * f + k] + 1];
        for ( std::size_t i = 0; i < nv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cursor( off.begin(), off.end() );
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const int v = soup.triangles[3 * f + k];
                nbr[cursor[v]++] = static_cast<LocalIndex>( f );
            }
        mesh.rebuildVertexFaces( off, nbr, "vertex_faces" );
    }

    // ---- CSR: vertex -> incident edges -------------------------------------
    {
        std::vector<int> off( nv + 1, 0 );
        for ( std::size_t e = 0; e < ne; ++e )
        {
            ++off[static_cast<int>( ep[e][0] ) + 1];
            ++off[static_cast<int>( ep[e][1] ) + 1];
        }
        for ( std::size_t i = 0; i < nv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cursor( off.begin(), off.end() );
        for ( std::size_t e = 0; e < ne; ++e )
            for ( int j = 0; j < 2; ++j )
            {
                const int v = static_cast<int>( ep[e][j] );
                nbr[cursor[v]++] = static_cast<LocalIndex>( e );
            }
        mesh.rebuildVertexEdges( off, nbr, "vertex_edges" );
    }

    // Replicated / serial mesh: every entity is owned by this rank. Nothing
    // can be stale yet (this is the initial build), but setOwnedCounts() bumps
    // generation() here too for uniformity with every other count-changing op.
    mesh.setOwnedCounts( nv, ne, nf );
}

//! Convenience: generate an icosphere of the given subdivision level and build
//! full connectivity into `mesh`.
template <class MeshT>
void buildIcosphere( MeshT& mesh, int subdivisions )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_BUILD_ICOSPHERE );
    static_assert( MeshT::dim == 3, "buildIcosphere requires Dim == 3" );
    auto soup = generateIcosphere<typename MeshT::scalar_type>( subdivisions );
    buildFromTriangleSoup( mesh, soup );
}

} // namespace Tessera

#endif // TESSERA_MESH_BUILDER_HPP
