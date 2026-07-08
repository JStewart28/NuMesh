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

#ifndef TESSERA_DISTRIBUTE_HPP
#define TESSERA_DISTRIBUTE_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_HaloExchange.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// MeshHalo — the three per-entity-kind halo plans for a distributed mesh
// ============================================================================
template <class MemorySpace>
struct MeshHalo
{
    HaloExchangePlan<MemorySpace> vplan;
    HaloExchangePlan<MemorySpace> eplan;
    HaloExchangePlan<MemorySpace> fplan;
};

namespace detail
{

// Build one entity kind's halo plan. `ghosts` are this rank's ghost entities as
// (gid, owner) pairs in ascending-gid order; `gid2local[g]` maps a global gid to
// this rank's local index (-1 if absent). The recv side reads directly from the
// ghost list; the send side is discovered by telling each owner which of its
// entities we ghost (allToAllV) — the owner maps those gids back to its local
// indices. Both sides order a peer-pair's entities by the ghoster's gid order, so
// pack and unpack align without extra metadata.
template <class MemorySpace>
HaloExchangePlan<MemorySpace>
buildKindPlan( MPI_Comm comm, int self_rank, int comm_size,
               const std::vector<std::pair<GlobalId, Rank>>& ghosts,
               const std::vector<LocalIndex>& gid2local )
{
    std::map<int, std::vector<LocalIndex>> recv_by_peer, send_by_peer;
    std::vector<std::vector<GlobalId>> send_gids( comm_size );
    for ( const auto& go : ghosts )
    {
        const GlobalId gid = go.first;
        const Rank owner = go.second;
        recv_by_peer[owner].push_back( gid2local[gid] );
        send_gids[owner].push_back( gid );
    }

    // Each owner learns which of its entities we ghost, and replies implicitly by
    // building its own send list from the requests it receives.
    auto req = allToAllV( comm, send_gids );
    for ( int s = 0; s < comm_size; ++s )
    {
        const GlobalId* p = req.from( s );
        const int c = req.count( s );
        for ( int k = 0; k < c; ++k )
            send_by_peer[s].push_back(
                gid2local[static_cast<std::size_t>( p[k] )] );
    }

    HaloExchangePlan<MemorySpace> plan;
    plan.setFromHost( self_rank, send_by_peer, recv_by_peer );
    return plan;
}

} // namespace detail

// ============================================================================
// facePartitionByAxis — deterministic geometric block partition of faces
// ============================================================================
//
// Sorts faces by their centroid coordinate along `axis` (tie-broken by gid) and
// blocks the sorted order into comm_size contiguous groups. Because it runs on
// the replicated coarse mesh with identical input on every rank, every rank
// computes the same assignment with no communication. Sorting by one axis yields
// latitude-band partitions whose halos are just the band-boundary rings — small
// and predictable. Returns owner rank indexed by face gid (== face index on the
// replicated mesh). No Zoltan2 yet; Step 7 adds the geometric MultiJagged path.
template <class MeshT>
std::vector<Rank> facePartitionByAxis( const MeshT& mesh, int axis = 2 )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_PARTITION );
    using Scalar = typename MeshT::scalar_type;
    const int Nv = static_cast<int>( mesh.numVertices() );
    const int Nf = static_cast<int>( mesh.numFaces() );
    const int comm_size = mesh.commSize();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", Nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", Nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    // Face centroid coordinate along `axis` (gid == index on replicated mesh).
    std::vector<std::pair<double, int>> keyed( Nf );
    for ( int f = 0; f < Nf; ++f )
    {
        double c = 0.0;
        for ( int k = 0; k < 3; ++k )
            c += static_cast<double>(
                pos( static_cast<int>( fv( f, k ) ), axis ) );
        keyed[f] = { c / 3.0, f };
    }
    std::sort(
        keyed.begin(), keyed.end(),
        []( const std::pair<double, int>& a, const std::pair<double, int>& b )
        {
            if ( a.first != b.first )
                return a.first < b.first;
            return a.second < b.second; // gid tie-break -> deterministic
        } );

    std::vector<Rank> owner( Nf, 0 );
    for ( int p = 0; p < Nf; ++p )
    {
        const int f = keyed[p].second;
        owner[f] =
            static_cast<Rank>( static_cast<long long>( p ) * comm_size / Nf );
    }
    (void)sizeof( Scalar );
    return owner;
}

// ============================================================================
// distribute — replicated mesh -> distributed mesh with a 1-deep halo
// ============================================================================
//
// Preconditions: `mesh` is the full replicated coarse mesh (every rank identical,
// gid == index), as produced by buildIcosphere(); `faceOwner` is the deterministic
// face partition (indexed by face gid). Postconditions: the mesh holds only this
// rank's local entities, ordered owned-first then ghost; connectivity fields keep
// global gids; ownership follows the lowest-rank rule; the vertex 1-ring CSR is
// rebuilt in local indices; owned counts are set; and `halo` holds the three plans
// so a subsequent haloExchange() fills every ghost from its owner.
//
// Ownership (lowest-rank rule), computable locally because the mesh is replicated:
//   face f   -> faceOwner[f]
//   vertex v -> min faceOwner over faces incident to v
//   edge e   -> min faceOwner over the (<=2) faces incident to e
// Local set (1-deep closure of owned vertices' 1-rings):
//   local faces = owned faces + faces incident to an owned vertex
//   local verts = union of vertices over local faces (gives each owned vertex its
//                 full 1-ring), local edges likewise.
template <class MeshT>
void distribute( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                 const std::vector<Rank>& faceOwner )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_DISTRIBUTE );
    using memory_space = typename MeshT::memory_space;
    constexpr int Dim = MeshT::dim;
    const int R = mesh.rank();
    const int comm_size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int Nv = static_cast<int>( mesh.numVertices() );
    const int Ne = static_cast<int>( mesh.numEdges() );
    const int Nf = static_cast<int>( mesh.numFaces() );

    // ---- host copies of the replicated mesh --------------------------------
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", Nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", Ne );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", Nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );
    auto e_v = Cabana::slice<EdgeField::Verts>( he );
    auto e_f = Cabana::slice<EdgeField::Faces>( he );
    auto e_lev = Cabana::slice<EdgeField::Level>( he );
    auto f_v = Cabana::slice<FaceField::Verts>( hf );
    auto f_e = Cabana::slice<FaceField::Edges>( hf );
    auto f_lev = Cabana::slice<FaceField::Level>( hf );

    auto vf_off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().offsets );
    auto vf_nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().neighbors );

    // ---- ownership (lowest-rank rule) --------------------------------------
    std::vector<Rank> vOwner( Nv, 0 ), eOwner( Ne, 0 );
    for ( int v = 0; v < Nv; ++v )
    {
        Rank m = comm_size;
        for ( int p = vf_off( v ); p < vf_off( v + 1 ); ++p )
            m = std::min( m, faceOwner[vf_nbr( p )] );
        vOwner[v] = ( m == comm_size ) ? 0 : m;
    }
    for ( int e = 0; e < Ne; ++e )
    {
        Rank m = faceOwner[static_cast<int>( e_f( e, 0 ) )];
        if ( e_f( e, 1 ) != invalid_gid )
            m = std::min( m, faceOwner[static_cast<int>( e_f( e, 1 ) )] );
        eOwner[e] = m;
    }

    // ---- local entity sets --------------------------------------------------
    std::vector<char> inFace( Nf, 0 ), inV( Nv, 0 ), inE( Ne, 0 );
    for ( int f = 0; f < Nf; ++f )
        if ( faceOwner[f] == R )
            inFace[f] = 1;
    for ( int v = 0; v < Nv; ++v )
        if ( vOwner[v] == R )
            for ( int p = vf_off( v ); p < vf_off( v + 1 ); ++p )
                inFace[vf_nbr( p )] = 1;
    for ( int f = 0; f < Nf; ++f )
        if ( inFace[f] )
            for ( int k = 0; k < 3; ++k )
            {
                inV[static_cast<int>( f_v( f, k ) )] = 1;
                inE[static_cast<int>( f_e( f, k ) )] = 1;
            }

    // ---- owned-first ordering (indices ascending == gid ascending) ---------
    auto build_order = [&]( int N, const std::vector<char>& in,
                            const std::vector<Rank>& owner,
                            std::vector<int>& order,
                            std::vector<LocalIndex>& g2l, int& n_owned )
    {
        std::vector<int> owned, ghost;
        for ( int i = 0; i < N; ++i )
            if ( in[i] )
                ( owner[i] == R ? owned : ghost ).push_back( i );
        n_owned = static_cast<int>( owned.size() );
        order.clear();
        order.insert( order.end(), owned.begin(), owned.end() );
        order.insert( order.end(), ghost.begin(), ghost.end() );
        g2l.assign( N, invalid_local );
        for ( int li = 0; li < static_cast<int>( order.size() ); ++li )
            g2l[order[li]] = li;
    };
    std::vector<int> vorder, eorder, forder;
    std::vector<LocalIndex> v2l, e2l, f2l;
    int nov, noe, nof;
    build_order( Nv, inV, vOwner, vorder, v2l, nov );
    build_order( Ne, inE, eOwner, eorder, e2l, noe );
    build_order( Nf, inFace, faceOwner, forder, f2l, nof );
    const int nlv = static_cast<int>( vorder.size() );
    const int nle = static_cast<int>( eorder.size() );
    const int nlf = static_cast<int>( forder.size() );

    // ---- build compact local AoSoAs, then hand them to the mesh ------------
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            lv( "lv", nlv );
        auto gid = Cabana::slice<VertexField::Gid>( lv );
        auto own = Cabana::slice<VertexField::Owner>( lv );
        auto flg = Cabana::slice<VertexField::Flags>( lv );
        auto pos = Cabana::slice<VertexField::Position>( lv );
        for ( int li = 0; li < nlv; ++li )
        {
            const int g = vorder[li];
            gid( li ) = static_cast<GlobalId>( g );
            own( li ) = vOwner[g];
            flg( li ) = 0;
            for ( int d = 0; d < Dim; ++d )
                pos( li, d ) = v_pos( g, d );
        }
        mesh.resizeVertices( nlv );
        Cabana::deep_copy( mesh.vertices(), lv );
    }
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> le(
            "le", nle );
        auto gid = Cabana::slice<EdgeField::Gid>( le );
        auto own = Cabana::slice<EdgeField::Owner>( le );
        auto lev = Cabana::slice<EdgeField::Level>( le );
        auto verts = Cabana::slice<EdgeField::Verts>( le );
        auto faces = Cabana::slice<EdgeField::Faces>( le );
        for ( int li = 0; li < nle; ++li )
        {
            const int g = eorder[li];
            gid( li ) = static_cast<GlobalId>( g );
            own( li ) = eOwner[g];
            lev( li ) = e_lev( g );
            verts( li, 0 ) = e_v( g, 0 );
            verts( li, 1 ) = e_v( g, 1 );
            faces( li, 0 ) = e_f( g, 0 );
            faces( li, 1 ) = e_f( g, 1 );
        }
        mesh.resizeEdges( nle );
        Cabana::deep_copy( mesh.edges(), le );
    }
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> lf(
            "lf", nlf );
        auto gid = Cabana::slice<FaceField::Gid>( lf );
        auto own = Cabana::slice<FaceField::Owner>( lf );
        auto lev = Cabana::slice<FaceField::Level>( lf );
        auto verts = Cabana::slice<FaceField::Verts>( lf );
        auto edges = Cabana::slice<FaceField::Edges>( lf );
        for ( int li = 0; li < nlf; ++li )
        {
            const int g = forder[li];
            gid( li ) = static_cast<GlobalId>( g );
            own( li ) = faceOwner[g];
            lev( li ) = f_lev( g );
            for ( int k = 0; k < 3; ++k )
            {
                verts( li, k ) = f_v( g, k );
                edges( li, k ) = f_e( g, k );
            }
        }
        mesh.resizeFaces( nlf );
        Cabana::deep_copy( mesh.faces(), lf );
    }
    mesh.setOwnedCounts( nov, noe, nof );

    // ---- rebuild local CSR 1-ring (local indices) --------------------------
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DISTRIBUTE_CSR );
        std::vector<int> off( nlv + 1, 0 );
        for ( int li = 0; li < nlf; ++li )
        {
            const int g = forder[li];
            for ( int k = 0; k < 3; ++k )
                ++off[v2l[static_cast<int>( f_v( g, k ) )] + 1];
        }
        for ( int i = 0; i < nlv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int li = 0; li < nlf; ++li )
        {
            const int g = forder[li];
            for ( int k = 0; k < 3; ++k )
                nbr[cur[v2l[static_cast<int>( f_v( g, k ) )]]++] =
                    static_cast<LocalIndex>( li );
        }
        detail::fillCsr( mesh.vertexFaces(), off, nbr, "vertex_faces" );
    }
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DISTRIBUTE_CSR );
        std::vector<int> off( nlv + 1, 0 );
        for ( int li = 0; li < nle; ++li )
        {
            const int g = eorder[li];
            for ( int j = 0; j < 2; ++j )
                ++off[v2l[static_cast<int>( e_v( g, j ) )] + 1];
        }
        for ( int i = 0; i < nlv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int li = 0; li < nle; ++li )
        {
            const int g = eorder[li];
            for ( int j = 0; j < 2; ++j )
                nbr[cur[v2l[static_cast<int>( e_v( g, j ) )]]++] =
                    static_cast<LocalIndex>( li );
        }
        detail::fillCsr( mesh.vertexEdges(), off, nbr, "vertex_edges" );
    }

    // ---- halo plans ---------------------------------------------------------
    auto ghosts_of = [&]( const std::vector<int>& order, int n_owned,
                          const std::vector<Rank>& owner )
    {
        std::vector<std::pair<GlobalId, Rank>> g;
        for ( int li = n_owned; li < static_cast<int>( order.size() ); ++li )
        {
            const int gi = order[li];
            g.push_back( { static_cast<GlobalId>( gi ), owner[gi] } );
        }
        return g;
    };
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DISTRIBUTE_HALOPLAN );
        halo.vplan = detail::buildKindPlan<memory_space>(
            comm, R, comm_size, ghosts_of( vorder, nov, vOwner ), v2l );
        halo.eplan = detail::buildKindPlan<memory_space>(
            comm, R, comm_size, ghosts_of( eorder, noe, eOwner ), e2l );
        halo.fplan = detail::buildKindPlan<memory_space>(
            comm, R, comm_size, ghosts_of( forder, nof, faceOwner ), f2l );
    }
}

// ============================================================================
// haloExchange(mesh, halo) — sync all three entity kinds
// ============================================================================
template <class MeshT, class MemorySpace>
void haloExchange( MeshT& mesh, MeshHalo<MemorySpace>& halo )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_HALO_EXCHANGE );
    haloExchange( mesh.comm(), mesh.vertices(), halo.vplan );
    haloExchange( mesh.comm(), mesh.edges(), halo.eplan );
    haloExchange( mesh.comm(), mesh.faces(), halo.fplan );
}

} // namespace Tessera

#endif // TESSERA_DISTRIBUTE_HPP
