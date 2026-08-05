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

#ifndef TESSERA_HALO_REBUILD_HPP
#define TESSERA_HALO_REBUILD_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// General (non-replicated) 1-deep halo rebuild
// ============================================================================
//
// This is the halo half of what migrate() used to be: given the entities a rank
// OWNS, discover ownership and the 1-deep ghost layer by communication alone and
// assemble the local storage plus the three halo plans. Nothing here moves an
// entity between ranks, so it is shared verbatim by the two callers that need it:
//
//   migrate()      moves owned faces first (round S + round A), then finishes here.
//   rebuildHalo()  moves nothing; refine() calls it to restore the ghost layer it
//                  dropped, so a second refine()/haloExchange() is well-defined.
//
// Rounds (letters kept from migrate()'s original single-function form, so the
// profiling keys and the design notes still line up):
//
//   G  Gather referenced-but-non-held vertex/edge tuples. A freshly-refined mesh
//      can hold an owned face whose vertex/edge is owned elsewhere and held
//      nowhere locally (refine() ships a midpoint's gid, not its position, and
//      drops the ghost layer). Each rank advertises its OWNED vertex/edge tuples
//      to a gid coordinator (gid % size) and pulls any missing reference back. A
//      no-op when every reference is already held.
//   B  Ownership + ghost discovery via coordinators. Each owned face advertises
//      (vertex, faceGid, thisRank) to the vertex coordinator and (edge, thisRank)
//      to the edge coordinator. A coordinator sets owner = lowest advertising rank
//      and (vertices only) returns to that owner the list of incident faces owned
//      by OTHER ranks — the ghost faces the owner must pull.
//   C  Ghost fetch. Each vertex owner requests the remote incident faces from
//      their owners; the owner replies with the face tuple plus its 3 vertex and
//      3 edge tuples (owners stamped), completing every owned vertex's 1-ring.
//   D  Assemble owned-first local AoSoAs, rebuild the vertex 1-ring CSRs, the key
//      side tables, and the three halo plans (buildKindPlan).
//
// Round G's postcondition is stronger than "a 1-deep ghost layer exists": every
// vertex and edge referenced by an owned face is held locally, WITH its position
// and its whole field pack. Round D's is that local index order is a pure function
// of the owned/ghost gid sets — owned first, then ghost, each ascending by gid.

namespace detail
{

//! Fixed-size byte image of a Cabana tuple so whole-tuple payloads can travel over
//! allToAllV (Cabana::Tuple is not trivially copyable, but its byte image is — the
//! same assumption the device migrate primitive makes with MPI_Type_contiguous).
template <class Tup>
struct TupleBlob
{
    unsigned char bytes[sizeof( Tup )];
};
template <class Tup>
TupleBlob<Tup> toBlob( const Tup& t )
{
    TupleBlob<Tup> b;
    std::memcpy( b.bytes, &t, sizeof( Tup ) );
    return b;
}
template <class Tup>
Tup fromBlob( const TupleBlob<Tup>& b )
{
    Tup t;
    std::memcpy( &t, b.bytes, sizeof( Tup ) );
    return t;
}

//! Deterministic coordinator rank for a single global id (vertex or edge).
inline int gidCoordRank( GlobalId g, int comm_size )
{
    return static_cast<int>( g % static_cast<GlobalId>( comm_size ) );
}

//! (vertex gid, an incident face's gid, that face's owner) advertisement.
struct VtxInc
{
    GlobalId vg;
    GlobalId faceGid;
    Rank owner;
};
//! (vertex gid, resolved owner) ownership reply / (edge gid, resolved owner).
struct GidOwn
{
    GlobalId gid;
    Rank owner;
};
//! (edge gid, an incident face's owner) advertisement.
struct EdgeInc
{
    GlobalId eg;
    Rank owner;
};

//! Host-space tuple types of a mesh's three entity kinds. The gid-keyed maps that
//! form the interface between the move half and the halo half hold these.
template <class MeshT>
using HostVertexTuple = typename Cabana::AoSoA<
    typename MeshT::vertex_member_types, Kokkos::HostSpace>::tuple_type;
template <class MeshT>
using HostEdgeTuple =
    typename Cabana::AoSoA<typename MeshT::edge_member_types,
                           Kokkos::HostSpace>::tuple_type;
template <class MeshT>
using HostFaceTuple =
    typename Cabana::AoSoA<typename MeshT::face_member_types,
                           Kokkos::HostSpace>::tuple_type;

//! ROUND G. Add to `heldV`/`heldE` every gid in `refVerts`/`refEdges` that is not
//! already there, fetching the full tuple from its true owner via a gid
//! coordinator (gid % size): every rank advertises its OWNED tuples to the
//! coordinator, then any rank missing a referenced gid requests it and the
//! coordinator replies the whole tuple. Same coordinator idiom the cross-rank
//! refine()/invariant checks use (MeshInvariants.hpp, test_markquality_edge.cpp's
//! maxOwnedEdgeLength).
//!
//! Collective. Callers that already materialize every reference —
//! distribute()+haloExchange(), readMesh() — need nothing, and the single
//! MPI_Allreduce below lets every rank skip the gather together.
template <class VTuple, class ETuple>
void gatherReferencedTuples( MPI_Comm comm, int self_rank, int comm_size,
                             const std::vector<GlobalId>& refVerts,
                             const std::vector<GlobalId>& refEdges,
                             std::map<GlobalId, VTuple>& heldV,
                             std::map<GlobalId, ETuple>& heldE )
{
    TESSERA_SCOPED_TIMER_DETAILED( ::Tessera::Profiling::TIMER_HALO_GATHER );
    std::set<GlobalId> needV, needE;
    for ( GlobalId g : refVerts )
        if ( heldV.find( g ) == heldV.end() )
            needV.insert( g );
    for ( GlobalId g : refEdges )
        if ( heldE.find( g ) == heldE.end() )
            needE.insert( g );

    long long localNeed = static_cast<long long>( needV.size() + needE.size() );
    long long globalNeed = 0;
    MPI_Allreduce( &localNeed, &globalNeed, 1, MPI_LONG_LONG, MPI_SUM, comm );
    if ( globalNeed == 0 )
        return;

    auto gather = [&]( auto& held, const std::set<GlobalId>& need,
                       auto ownerOf )
    {
        using Held = typename std::decay<decltype( held )>::type;
        using Tup = typename Held::mapped_type;
        struct Rec
        {
            GlobalId gid;
            TupleBlob<Tup> blob;
        };
        // Advertise every locally-OWNED tuple to its coordinator.
        std::vector<std::vector<Rec>> adv( comm_size );
        for ( const auto& kv : held )
            if ( ownerOf( kv.second ) == static_cast<Rank>( self_rank ) )
                adv[gidCoordRank( kv.first, comm_size )].push_back(
                    { kv.first, toBlob( kv.second ) } );
        auto advGot = allToAllV( comm, adv );
        std::map<GlobalId, TupleBlob<Tup>> coord;
        for ( const auto& r : advGot.data )
            coord[r.gid] = r.blob;

        // Request each needed gid from its coordinator; it replies the tuple.
        std::vector<std::vector<GlobalId>> req( comm_size );
        for ( GlobalId g : need )
            req[gidCoordRank( g, comm_size )].push_back( g );
        auto reqGot = allToAllV( comm, req );
        std::vector<std::vector<Rec>> rep( comm_size );
        for ( int s = 0; s < comm_size; ++s )
        {
            const GlobalId* p = reqGot.from( s );
            const int c = reqGot.count( s );
            for ( int i = 0; i < c; ++i )
                rep[s].push_back( { p[i], coord.at( p[i] ) } );
        }
        auto repGot = allToAllV( comm, rep );
        for ( const auto& r : repGot.data )
            held[r.gid] = fromBlob( r.blob );
    };

    gather( heldV, needV,
            []( const VTuple& t ) { return Cabana::get<VertexField::Owner>( t ); } );
    gather( heldE, needE,
            []( const ETuple& t ) { return Cabana::get<EdgeField::Owner>( t ); } );
}

//! ROUNDS B, C, D. Given the tuples this rank now OWNS (`faceById`) and the
//! vertex/edge tuples its owned faces reference (`vById`/`eById`, complete by
//! round G's postcondition), resolve ownership, fetch the 1-deep ghost layer, and
//! overwrite `mesh`'s storage and `halo`'s three plans.
//!
//! `vById`/`eById` are grown in place with the ghost tuples that arrive in round
//! C, so they are taken by non-const reference.
//!
//! INVALIDATION: reallocates the AoSoAs, the key Views and the CSRs, and replaces
//! (not merely clears) the halo plans — every slice, CSR handle and key View taken
//! out before this call is dangling.
template <class MeshT>
void finishHaloAndAssemble(
    MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
    const std::map<GlobalId, HostFaceTuple<MeshT>>& faceById,
    std::map<GlobalId, HostVertexTuple<MeshT>>& vById,
    std::map<GlobalId, HostEdgeTuple<MeshT>>& eById )
{
    using memory_space = typename MeshT::memory_space;
    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;
    using VTuple = HostVertexTuple<MeshT>;
    using ETuple = HostEdgeTuple<MeshT>;
    using FTuple = HostFaceTuple<MeshT>;

    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    std::map<GlobalId, Rank> vOwner, eOwner; // resolved owners

    // ======================================================================
    // Round B — ownership (lowest-rank) + ghost discovery via coordinators.
    // ======================================================================
    // Advertise each owned face's vertex/edge incidences (advertiser == owner).
    std::vector<std::vector<VtxInc>> toVC( size );
    std::vector<std::vector<EdgeInc>> toEC( size );
    for ( const auto& fkv : faceById )
    {
        const GlobalId fg = fkv.first;
        const FTuple& t = fkv.second;
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId vg = Cabana::get<FaceField::Verts>( t, k );
            const GlobalId eg = Cabana::get<FaceField::Edges>( t, k );
            toVC[gidCoordRank( vg, size )].push_back(
                { vg, fg, static_cast<Rank>( R ) } );
            toEC[gidCoordRank( eg, size )].push_back(
                { eg, static_cast<Rank>( R ) } );
        }
    }
    auto vAdv = allToAllV( comm, toVC );
    auto eAdv = allToAllV( comm, toEC );

    // Vertex coordinator: owner = min advertiser; reply owner to each advertiser
    // and hand the owner the list of incident faces owned by OTHER ranks.
    std::map<GlobalId, std::vector<VtxInc>> byV;
    for ( const auto& m : vAdv.data )
        byV[m.vg].push_back( m );
    std::vector<std::vector<GidOwn>> vOwnReply( size );
    std::vector<std::vector<VtxInc>> ghostNeed( size );
    for ( const auto& kv : byV )
    {
        const GlobalId vg = kv.first;
        Rank owner = size;
        std::set<Rank> advertisers;
        for ( const auto& inc : kv.second )
        {
            owner = std::min( owner, inc.owner );
            advertisers.insert( inc.owner );
        }
        for ( Rank r : advertisers )
            vOwnReply[r].push_back( { vg, owner } );
        for ( const auto& inc : kv.second )
            if ( inc.owner != owner )
                ghostNeed[owner].push_back( inc );
    }
    auto vOwnGot = allToAllV( comm, vOwnReply );
    auto needGot = allToAllV( comm, ghostNeed );
    for ( const auto& m : vOwnGot.data )
        vOwner[m.gid] = m.owner;

    // Edge coordinator: owner = min advertiser; reply owner to each advertiser.
    std::map<GlobalId, std::vector<Rank>> byE;
    for ( const auto& m : eAdv.data )
        byE[m.eg].push_back( m.owner );
    std::vector<std::vector<GidOwn>> eOwnReply( size );
    for ( const auto& kv : byE )
    {
        Rank owner = size;
        std::set<Rank> advertisers;
        for ( Rank r : kv.second )
        {
            owner = std::min( owner, r );
            advertisers.insert( r );
        }
        for ( Rank r : advertisers )
            eOwnReply[r].push_back( { kv.first, owner } );
    }
    auto eOwnGot = allToAllV( comm, eOwnReply );
    for ( const auto& m : eOwnGot.data )
        eOwner[m.gid] = m.owner;

    // ======================================================================
    // Round C — fetch ghost faces (+ their vertices/edges) from face owners.
    // ======================================================================
    std::map<GlobalId, Rank> neededFace; // ghost faceGid -> its owner
    for ( const auto& m : needGot.data )
        if ( faceById.find( m.faceGid ) == faceById.end() )
            neededFace[m.faceGid] = m.owner;
    std::vector<std::vector<GlobalId>> reqFace( size );
    for ( const auto& kv : neededFace )
        reqFace[kv.second].push_back( kv.first );
    auto reqGot = allToAllV( comm, reqFace );

    // Serve requests: reply the face tuple + its 3 vertex/edge tuples (owners
    // stamped from this rank's resolved maps) to the requesting rank.
    std::vector<std::vector<TupleBlob<FTuple>>> repF( size );
    std::vector<std::vector<TupleBlob<VTuple>>> repV( size );
    std::vector<std::vector<TupleBlob<ETuple>>> repE( size );
    for ( int s = 0; s < size; ++s )
    {
        const GlobalId* p = reqGot.from( s );
        const int c = reqGot.count( s );
        for ( int i = 0; i < c; ++i )
        {
            FTuple ft = faceById.at( p[i] );
            Cabana::get<FaceField::Owner>( ft ) = static_cast<Rank>( R );
            repF[s].push_back( toBlob( ft ) );
            for ( int k = 0; k < 3; ++k )
            {
                const GlobalId vg = Cabana::get<FaceField::Verts>( ft, k );
                const GlobalId eg = Cabana::get<FaceField::Edges>( ft, k );
                VTuple vt = vById.at( vg );
                Cabana::get<VertexField::Owner>( vt ) = vOwner.at( vg );
                repV[s].push_back( toBlob( vt ) );
                ETuple et = eById.at( eg );
                Cabana::get<EdgeField::Owner>( et ) = eOwner.at( eg );
                repE[s].push_back( toBlob( et ) );
            }
        }
    }
    auto ghF = allToAllV( comm, repF );
    auto ghV = allToAllV( comm, repV );
    auto ghE = allToAllV( comm, repE );

    // Ingest ghosts. Ghost faces are new; ghost vertices/edges may repeat locally
    // held ones (dedup by gid). Owners for ghosts come stamped in their tuples.
    std::map<GlobalId, FTuple> ghostFaceById;
    for ( const auto& b : ghF.data )
    {
        FTuple t = fromBlob( b );
        ghostFaceById[Cabana::get<FaceField::Gid>( t )] = t;
    }
    for ( const auto& b : ghV.data )
    {
        VTuple t = fromBlob( b );
        const GlobalId vg = Cabana::get<VertexField::Gid>( t );
        if ( vById.find( vg ) == vById.end() )
        {
            vById.emplace( vg, t );
            vOwner[vg] = Cabana::get<VertexField::Owner>( t );
        }
    }
    for ( const auto& b : ghE.data )
    {
        ETuple t = fromBlob( b );
        const GlobalId eg = Cabana::get<EdgeField::Gid>( t );
        if ( eById.find( eg ) == eById.end() )
        {
            eById.emplace( eg, t );
            eOwner[eg] = Cabana::get<EdgeField::Owner>( t );
        }
    }

    // ======================================================================
    // Round D — assemble owned-first local AoSoAs + CSR + keys + halo plans.
    // ======================================================================
    // Owned-first ordering (owned then ghost, each ascending gid) per kind. This
    // is the canonical layout: local index order is a pure function of the two
    // gid sets, so a mesh's layout does not depend on how it was reached.
    auto order_of = [&]( const std::vector<GlobalId>& all,
                         const std::map<GlobalId, Rank>& owner, int& n_owned )
    {
        std::vector<GlobalId> owned, ghost;
        for ( GlobalId g : all )
            ( owner.at( g ) == R ? owned : ghost ).push_back( g );
        std::sort( owned.begin(), owned.end() );
        std::sort( ghost.begin(), ghost.end() );
        n_owned = static_cast<int>( owned.size() );
        std::vector<GlobalId> ord = owned;
        ord.insert( ord.end(), ghost.begin(), ghost.end() );
        return ord;
    };

    std::vector<GlobalId> vAll, eAll;
    for ( const auto& kv : vById )
        vAll.push_back( kv.first );
    for ( const auto& kv : eById )
        eAll.push_back( kv.first );
    int nOwnedV = 0, nOwnedE = 0;
    std::vector<GlobalId> vord = order_of( vAll, vOwner, nOwnedV );
    std::vector<GlobalId> eord = order_of( eAll, eOwner, nOwnedE );

    // Faces: owned (received) first, then ghosts (fetched), each ascending gid.
    std::vector<GlobalId> fOwned, fGhost;
    for ( const auto& kv : faceById )
        fOwned.push_back( kv.first );
    for ( const auto& kv : ghostFaceById )
        fGhost.push_back( kv.first );
    std::sort( fOwned.begin(), fOwned.end() );
    std::sort( fGhost.begin(), fGhost.end() );
    const int nOwnedF = static_cast<int>( fOwned.size() );
    std::vector<GlobalId> ford = fOwned;
    ford.insert( ford.end(), fGhost.begin(), fGhost.end() );

    const int nlv = static_cast<int>( vord.size() );
    const int nle = static_cast<int>( eord.size() );
    const int nlf = static_cast<int>( ford.size() );

    // gid -> final local index (dense vectors sized to the local max gid; every
    // gid indexed here is held locally, so it is within range).
    auto make_g2l = [&]( const std::vector<GlobalId>& ord )
    {
        GlobalId mx = 0;
        for ( GlobalId g : ord )
            mx = std::max( mx, g );
        std::vector<LocalIndex> g2l( static_cast<std::size_t>( mx ) + 1,
                                     invalid_local );
        for ( int li = 0; li < static_cast<int>( ord.size() ); ++li )
            g2l[ord[li]] = li;
        return g2l;
    };
    std::vector<LocalIndex> v2l = make_g2l( vord );
    std::vector<LocalIndex> e2l = make_g2l( eord );
    std::vector<LocalIndex> f2l = make_g2l( ford );

    // INVALIDATION: the resize/deep_copy calls below, the key-View reassignment,
    // and the CSR rebuild reallocate and reassign this mesh's storage,
    // invalidating every slice/CSR/key-View handed out before this call. The halo
    // plans are replaced (not merely cleared) further down for the same reason.
    //
    // Vertex AoSoA (tuple carries position + user fields; owner set explicitly).
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_HALO_ASSEMBLE );
        Cabana::AoSoA<VMT, Kokkos::HostSpace> lv( "lv", nlv );
        auto own = Cabana::slice<VertexField::Owner>( lv );
        for ( int li = 0; li < nlv; ++li )
        {
            lv.setTuple( li, vById.at( vord[li] ) );
            own( li ) = vOwner.at( vord[li] );
        }
        mesh.resizeVertices( nlv );
        Cabana::deep_copy( mesh.vertices(), lv );
    }
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_HALO_ASSEMBLE );
        Cabana::AoSoA<EMT, Kokkos::HostSpace> le( "le", nle );
        auto own = Cabana::slice<EdgeField::Owner>( le );
        for ( int li = 0; li < nle; ++li )
        {
            le.setTuple( li, eById.at( eord[li] ) );
            own( li ) = eOwner.at( eord[li] );
        }
        mesh.resizeEdges( nle );
        Cabana::deep_copy( mesh.edges(), le );
    }
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_HALO_ASSEMBLE );
        Cabana::AoSoA<FMT, Kokkos::HostSpace> lf( "lf", nlf );
        auto own = Cabana::slice<FaceField::Owner>( lf );
        for ( int li = 0; li < nlf; ++li )
        {
            const GlobalId g = ford[li];
            lf.setTuple( li, li < nOwnedF ? faceById.at( g )
                                          : ghostFaceById.at( g ) );
            // Owned faces belong to this rank; ghost faces keep the owner their
            // sender stamped into the tuple.
            if ( li < nOwnedF )
                own( li ) = static_cast<Rank>( R );
        }
        mesh.resizeFaces( nlf );
        Cabana::deep_copy( mesh.faces(), lf );
    }
    mesh.setOwnedCounts( nOwnedV, nOwnedE, nOwnedF );

    // Rebuild vertex 1-ring CSR (local indices) over ALL local faces/edges.
    {
        std::vector<int> off( nlv + 1, 0 );
        for ( int li = 0; li < nlf; ++li )
        {
            const FTuple& t = ( li < nOwnedF ) ? faceById.at( ford[li] )
                                               : ghostFaceById.at( ford[li] );
            for ( int k = 0; k < 3; ++k )
                ++off[v2l[Cabana::get<FaceField::Verts>( t, k )] + 1];
        }
        for ( int i = 0; i < nlv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int li = 0; li < nlf; ++li )
        {
            const FTuple& t = ( li < nOwnedF ) ? faceById.at( ford[li] )
                                               : ghostFaceById.at( ford[li] );
            for ( int k = 0; k < 3; ++k )
                nbr[cur[v2l[Cabana::get<FaceField::Verts>( t, k )]]++] =
                    static_cast<LocalIndex>( li );
        }
        mesh.rebuildVertexFaces( off, nbr, "vertex_faces" );
    }
    {
        std::vector<int> off( nlv + 1, 0 );
        for ( int li = 0; li < nle; ++li )
        {
            const ETuple& t = eById.at( eord[li] );
            for ( int j = 0; j < 2; ++j )
                ++off[v2l[Cabana::get<EdgeField::Verts>( t, j )] + 1];
        }
        for ( int i = 0; i < nlv; ++i )
            off[i + 1] += off[i];
        std::vector<LocalIndex> nbr( off.back() );
        std::vector<int> cur( off.begin(), off.end() );
        for ( int li = 0; li < nle; ++li )
        {
            const ETuple& t = eById.at( eord[li] );
            for ( int j = 0; j < 2; ++j )
                nbr[cur[v2l[Cabana::get<EdgeField::Verts>( t, j )]]++] =
                    static_cast<LocalIndex>( li );
        }
        mesh.rebuildVertexEdges( off, nbr, "vertex_edges" );
    }

    // Rebuild key side tables.
    {
        Kokkos::View<EdgeKey*, memory_space> ek(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
            nle );
        auto h_ek = Kokkos::create_mirror_view( ek );
        for ( int li = 0; li < nle; ++li )
        {
            const ETuple& t = eById.at( eord[li] );
            h_ek( li ) = makeEdgeKey( Cabana::get<EdgeField::Verts>( t, 0 ),
                                      Cabana::get<EdgeField::Verts>( t, 1 ) );
        }
        Kokkos::deep_copy( ek, h_ek );
        mesh.setEdgeKeys( ek );

        Kokkos::View<FaceKey*, memory_space> fk(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
            nlf );
        auto h_fk = Kokkos::create_mirror_view( fk );
        for ( int li = 0; li < nlf; ++li )
        {
            const FTuple& t = ( li < nOwnedF ) ? faceById.at( ford[li] )
                                               : ghostFaceById.at( ford[li] );
            h_fk( li ) = makeFaceKey( Cabana::get<FaceField::Verts>( t, 0 ),
                                      Cabana::get<FaceField::Verts>( t, 1 ),
                                      Cabana::get<FaceField::Verts>( t, 2 ) );
        }
        Kokkos::deep_copy( fk, h_fk );
        mesh.setFaceKeys( fk );
    }

    // Halo plans: ghosts are the trailing (owner != R) entries, already ascending
    // by gid, matching the buildKindPlan alignment contract.
    auto ghosts_of = [&]( const std::vector<GlobalId>& ord, int n_owned,
                          const std::map<GlobalId, Rank>& owner )
    {
        std::vector<std::pair<GlobalId, Rank>> g;
        for ( int li = n_owned; li < static_cast<int>( ord.size() ); ++li )
            g.push_back( { ord[li], owner.at( ord[li] ) } );
        return g;
    };
    std::map<GlobalId, Rank> fOwnerMap;
    for ( GlobalId g : fOwned )
        fOwnerMap[g] = static_cast<Rank>( R );
    for ( const auto& kv : ghostFaceById )
        fOwnerMap[kv.first] = Cabana::get<FaceField::Owner>( kv.second );

    // INVALIDATION: the local entity count and ghost set changed above, so the
    // previous halo plans are stale; replace (not merely clear) them here.
    halo.vplan = buildKindPlan<memory_space>(
        comm, R, size, ghosts_of( vord, nOwnedV, vOwner ), v2l );
    halo.eplan = buildKindPlan<memory_space>(
        comm, R, size, ghosts_of( eord, nOwnedE, eOwner ), e2l );
    halo.fplan = buildKindPlan<memory_space>(
        comm, R, size, ghosts_of( ford, nOwnedF, fOwnerMap ), f2l );
}

} // namespace detail

//! Rebuild the 1-deep ghost layer and the three halo plans in place, from the
//! mesh's current OWNED entities. No entity changes rank, and nothing is
//! communicated beyond ownership/ghost discovery. After this returns,
//! haloExchange() is meaningful and refine() may be called again.
//!
//! refine() calls this itself, so a caller normally never needs to: it exists as
//! public API for the case where a mesh's owned set was changed by something other
//! than refine()/migrate(), and because naming the operation is what let refine()
//! stop leaving a mesh whose halo was silently empty.
//!
//! Postconditions:
//!   * Every vertex and edge referenced by an owned face is held locally (owned or
//!     ghost) WITH its position and whole field pack — round G's guarantee, which
//!     is stronger than "a 1-deep ghost layer exists".
//!   * Every face sharing a vertex with an owned face is held as a ghost.
//!   * halo.{v,e,f}plan are consistent with the new local indices; ghost values
//!     are the owners' values, so a following haloExchange() is a re-sync.
//!   * Local ordering is CANONICAL: owned first then ghost, each kind ascending by
//!     gid. Callers can observe this through gid-derived face masks.
//!
//! Collective. `mesh` must be owned-first with global gids on every entity; a
//! valid halo on entry is NOT required (that is the point).
//!
//! INVALIDATION: reallocates the AoSoAs, key Views and CSRs, and replaces the
//! halo plans — every slice, CSR handle and key View taken out before this call is
//! dangling. Re-slice from the mesh afterwards.
template <class MeshT>
void rebuildHalo( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_HALO_REBUILD );
    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;
    using VTuple = detail::HostVertexTuple<MeshT>;
    using ETuple = detail::HostEdgeTuple<MeshT>;
    using FTuple = detail::HostFaceTuple<MeshT>;

    const int nv = static_cast<int>( mesh.numVertices() );
    const int ne = static_cast<int>( mesh.numEdges() );
    const int nof = static_cast<int>( mesh.numOwnedFaces() );

    // Host copies of everything currently held (owned + any ghost).
    Cabana::AoSoA<VMT, Kokkos::HostSpace> hv( "hv", nv );
    Cabana::AoSoA<EMT, Kokkos::HostSpace> he( "he", ne );
    Cabana::AoSoA<FMT, Kokkos::HostSpace> hf( "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto e_gid = Cabana::slice<EdgeField::Gid>( he );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_edges = Cabana::slice<FaceField::Edges>( hf );

    std::map<GlobalId, VTuple> heldV;
    std::map<GlobalId, ETuple> heldE;
    for ( int i = 0; i < nv; ++i )
        heldV[v_gid( i )] = hv.getTuple( i );
    for ( int i = 0; i < ne; ++i )
        heldE[e_gid( i )] = he.getTuple( i );

    // Round G: recover any vertex/edge an owned face references but this rank does
    // not hold. After refine() this is the common case, not the exception — a
    // closure child's midpoint corner is owned by the refining neighbour.
    std::vector<GlobalId> refV, refE;
    refV.reserve( static_cast<std::size_t>( nof ) * 3 );
    refE.reserve( static_cast<std::size_t>( nof ) * 3 );
    for ( int f = 0; f < nof; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            refV.push_back( f_verts( f, k ) );
            refE.push_back( f_edges( f, k ) );
        }
    detail::gatherReferencedTuples( mesh.comm(), mesh.rank(), mesh.commSize(),
                                   refV, refE, heldV, heldE );

    // The three gid-keyed maps rounds B/C/D consume. migrate() fills them from the
    // faces it RECEIVED; here nothing moved, so they come from the owned faces
    // already in hand — a purely local fill, no communication.
    std::map<GlobalId, FTuple> faceById;
    std::map<GlobalId, VTuple> vById;
    std::map<GlobalId, ETuple> eById;
    for ( int f = 0; f < nof; ++f )
    {
        faceById[f_gid( f )] = hf.getTuple( f );
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId vg = f_verts( f, k );
            const GlobalId eg = f_edges( f, k );
            vById.emplace( vg, heldV.at( vg ) );
            eById.emplace( eg, heldE.at( eg ) );
        }
    }

    detail::finishHaloAndAssemble( mesh, halo, faceById, vById, eById );
}

} // namespace Tessera

#endif // TESSERA_HALO_REBUILD_HPP
