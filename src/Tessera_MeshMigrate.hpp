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

#ifndef TESSERA_MESH_MIGRATE_HPP
#define TESSERA_MESH_MIGRATE_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_CsrAdjacency.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_HaloExchange.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

namespace Tessera
{

// ============================================================================
// Mesh migration + general (non-replicated) 1-deep halo rebuild (Step 7)
// ============================================================================
//
// migrate() applies an externally-computed face assignment: each OWNED face is
// moved to dest[localFaceIndex], its vertices and edges (with their whole field
// pack) follow, ownership is recomputed under the lowest-rank rule, and a fresh
// 1-deep ghost layer + halo plans are built. Unlike distribute() (Step 5) it does
// NOT assume a replicated mesh — every rank holds only its own entities on entry,
// so all cross-rank knowledge is discovered by communication. This is the general
// ghost builder that Step 6b's distributed refine() deferred; it is the public
// contract Canopy drives with its FMM-tree assignment (via ownedFace* accessors),
// and the substrate loadBalance() (Step 7b, Zoltan2) reuses unchanged.
//
// The move and rebuild are orchestrated host-side over allToAllV (like
// distribute()): entity payloads travel as whole Cabana tuples (generic over the
// field pack), and ownership / ghost discovery run through coordinators
// (vertex/edge coordinator rank = gid % size) rather than replicated lookups. The
// device migrate primitive (Step 4a) stays the tested building block for pure
// single-destination AoSoA moves; a mesh migrate additionally needs multi-
// destination vertex/edge follow and a ghost rebuild, which the host orchestration
// expresses directly.
//
// Rounds:
//   G  Gather referenced-but-non-held tuples. A freshly-refined mesh can hold an
//      owned face whose vertex/edge is owned elsewhere and held nowhere locally
//      (refine() ships a midpoint's gid, not its position, and drops the ghost
//      layer). Each rank advertises its OWNED vertex/edge tuples to a gid
//      coordinator (gid % size) and pulls any missing reference back, so round A
//      sees a self-contained 1-ring. A no-op when every reference is already held.
//   A  Move each owned face + its 3 vertices + 3 edges to dest (three allToAllV).
//      The receiver's owned faces are the faces it received; its candidate
//      vertices/edges are their (deduped) endpoints.
//   B  Ownership + ghost discovery via coordinators. Each owned face advertises
//      (vertex, faceGid, thisRank) to the vertex coordinator and (edge, thisRank)
//      to the edge coordinator. A coordinator sets owner = lowest advertising rank
//      and (vertices only) returns to that owner the list of incident faces owned
//      by OTHER ranks — the ghost faces the owner must pull.
//   C  Ghost fetch. Each vertex owner requests the remote incident faces from
//      their owners; the owner replies with the face tuple plus its 3 vertex and
//      3 edge tuples (owners stamped), completing every owned vertex's 1-ring.
//   D  Assemble owned-first local AoSoAs, rebuild the vertex 1-ring CSR, the key
//      side tables, and the three halo plans (buildKindPlan). The mesh is left
//      halo-consistent (ghost values are the owners' values); a subsequent
//      haloExchange() re-syncs the field pack over the new plans.
//
// Preconditions: `mesh` is distributed and owned-first, entities carry global
// gids, and dest.size() == numOwnedFaces() with every entry in [0, commSize()).
// A valid 1-deep halo is NOT required: round G recovers any vertex/edge an owned
// face references but does not hold locally (the post-refine case), so both a
// halo-consistent mesh (distribute()+haloExchange, readMesh) and a freshly-refined
// owned-only mesh are accepted. Edge/face connectivity fields store global gids and
// are carried verbatim (an edge's incident-face gids may reference a face now on
// another rank; that metadata is not used by the 1-ring invariants and is left
// as-is).

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

} // namespace detail

//! Migrate owned faces to `dest` (indexed by owned face local index), moving their
//! vertices/edges + whole field pack, recomputing lowest-rank ownership, and
//! rebuilding the 1-deep halo. See the header comment for the algorithm.
template <class MeshT>
void migrate( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
              const std::vector<Rank>& dest )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_MIGRATE );
    using memory_space = typename MeshT::memory_space;
    constexpr int Dim = MeshT::dim;
    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;
    using VTuple = typename Cabana::AoSoA<VMT, Kokkos::HostSpace>::tuple_type;
    using ETuple = typename Cabana::AoSoA<EMT, Kokkos::HostSpace>::tuple_type;
    using FTuple = typename Cabana::AoSoA<FMT, Kokkos::HostSpace>::tuple_type;

    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int nv = static_cast<int>( mesh.numVertices() );
    const int ne = static_cast<int>( mesh.numEdges() );
    const int nof = static_cast<int>( mesh.numOwnedFaces() );

    // ---- host copies of everything currently held (owned + ghost) ----------
    Cabana::AoSoA<VMT, Kokkos::HostSpace> hv( "hv", nv );
    Cabana::AoSoA<EMT, Kokkos::HostSpace> he( "he", ne );
    Cabana::AoSoA<FMT, Kokkos::HostSpace> hf( "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto e_gid = Cabana::slice<EdgeField::Gid>( he );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_edges = Cabana::slice<FaceField::Edges>( hf );

    // gid -> currently-held tuple (owned + any ghost). After refine() a rank can
    // own a face whose vertex/edge is owned by another rank and held nowhere
    // locally -- refine() ships a new midpoint's gid to its co-sharers, never its
    // position (Tessera_RefineParallel.hpp Phase-2), and drops the pre-refine
    // ghost layer. migrate() is the deferred post-refine ghost builder (README
    // "Load balancing"), so it must recover those referenced tuples before round A
    // can move a face with its full vertex/edge pack.
    std::map<GlobalId, VTuple> heldV;
    std::map<GlobalId, ETuple> heldE;
    for ( int i = 0; i < nv; ++i )
        heldV[v_gid( i )] = hv.getTuple( i );
    for ( int i = 0; i < ne; ++i )
        heldE[e_gid( i )] = he.getTuple( i );

    // Gather referenced-but-non-held vertex/edge tuples from their true owner via
    // a gid coordinator (gid % size): every rank advertises its OWNED tuples to
    // the coordinator, then any rank missing a gid its owned faces reference
    // requests it and the coordinator replies the full tuple. Same coordinator
    // idiom the cross-rank refine()/invariant checks use (MeshInvariants.hpp,
    // test_markquality_edge.cpp's maxOwnedEdgeLength). Callers that already
    // materialize every reference -- distribute()+haloExchange(), readMesh() --
    // need nothing and skip the gather entirely.
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_MIGRATE_GATHER );
        std::set<GlobalId> needV, needE;
        for ( int f = 0; f < nof; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                if ( heldV.find( f_verts( f, k ) ) == heldV.end() )
                    needV.insert( f_verts( f, k ) );
                if ( heldE.find( f_edges( f, k ) ) == heldE.end() )
                    needE.insert( f_edges( f, k ) );
            }
        long long localNeed =
            static_cast<long long>( needV.size() + needE.size() );
        long long globalNeed = 0;
        MPI_Allreduce( &localNeed, &globalNeed, 1, MPI_LONG_LONG, MPI_SUM,
                       comm );

        auto gather =
            [&]( auto& held, const std::set<GlobalId>& need, auto ownerOf )
        {
            using Held = typename std::decay<decltype( held )>::type;
            using Tup = typename Held::mapped_type;
            struct Rec
            {
                GlobalId gid;
                detail::TupleBlob<Tup> blob;
            };
            // Advertise every locally-OWNED tuple to its coordinator.
            std::vector<std::vector<Rec>> adv( size );
            for ( const auto& kv : held )
                if ( ownerOf( kv.second ) == static_cast<Rank>( R ) )
                    adv[detail::gidCoordRank( kv.first, size )].push_back(
                        { kv.first, detail::toBlob( kv.second ) } );
            auto advGot = allToAllV( comm, adv );
            std::map<GlobalId, detail::TupleBlob<Tup>> coord;
            for ( const auto& r : advGot.data )
                coord[r.gid] = r.blob;

            // Request each needed gid from its coordinator; it replies the tuple.
            std::vector<std::vector<GlobalId>> req( size );
            for ( GlobalId g : need )
                req[detail::gidCoordRank( g, size )].push_back( g );
            auto reqGot = allToAllV( comm, req );
            std::vector<std::vector<Rec>> rep( size );
            for ( int s = 0; s < size; ++s )
            {
                const GlobalId* p = reqGot.from( s );
                const int c = reqGot.count( s );
                for ( int i = 0; i < c; ++i )
                    rep[s].push_back( { p[i], coord.at( p[i] ) } );
            }
            auto repGot = allToAllV( comm, rep );
            for ( const auto& r : repGot.data )
                held[r.gid] = detail::fromBlob( r.blob );
        };

        if ( globalNeed > 0 )
        {
            gather( heldV, needV, []( const VTuple& t )
                    { return Cabana::get<VertexField::Owner>( t ); } );
            gather( heldE, needE, []( const ETuple& t )
                    { return Cabana::get<EdgeField::Owner>( t ); } );
        }
    }

    // ======================================================================
    // Round A — move owned faces + their vertices/edges to destinations.
    // ======================================================================
    // Owned faces (unique) and candidate vertices/edges (deduped by gid).
    std::map<GlobalId, FTuple> faceById;     // this rank's new owned faces
    std::map<GlobalId, VTuple> vById;        // vertices referenced locally
    std::map<GlobalId, ETuple> eById;        // edges referenced locally
    std::map<GlobalId, Rank> vOwner, eOwner; // resolved owners (filled below)
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_MIGRATE_MOVE );
        std::vector<std::vector<detail::TupleBlob<FTuple>>> sendF( size );
        std::vector<std::vector<detail::TupleBlob<VTuple>>> sendV( size );
        std::vector<std::vector<detail::TupleBlob<ETuple>>> sendE( size );
        for ( int f = 0; f < nof; ++f )
        {
            const Rank d = dest[f];
            sendF[d].push_back( detail::toBlob( hf.getTuple( f ) ) );
            for ( int k = 0; k < 3; ++k )
            {
                sendV[d].push_back(
                    detail::toBlob( heldV.at( f_verts( f, k ) ) ) );
                sendE[d].push_back(
                    detail::toBlob( heldE.at( f_edges( f, k ) ) ) );
            }
        }
        auto gotF = allToAllV( comm, sendF );
        auto gotV = allToAllV( comm, sendV );
        auto gotE = allToAllV( comm, sendE );

        for ( const auto& b : gotF.data )
        {
            FTuple t = detail::fromBlob( b );
            faceById[Cabana::get<FaceField::Gid>( t )] = t;
        }
        for ( const auto& b : gotV.data )
        {
            VTuple t = detail::fromBlob( b );
            vById.emplace( Cabana::get<VertexField::Gid>( t ), t );
        }
        for ( const auto& b : gotE.data )
        {
            ETuple t = detail::fromBlob( b );
            eById.emplace( Cabana::get<EdgeField::Gid>( t ), t );
        }
    }

    // ======================================================================
    // Round B — ownership (lowest-rank) + ghost discovery via coordinators.
    // ======================================================================
    // Advertise each owned face's vertex/edge incidences (advertiser == owner).
    std::vector<std::vector<detail::VtxInc>> toVC( size );
    std::vector<std::vector<detail::EdgeInc>> toEC( size );
    for ( const auto& fkv : faceById )
    {
        const GlobalId fg = fkv.first;
        const FTuple& t = fkv.second;
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId vg = Cabana::get<FaceField::Verts>( t, k );
            const GlobalId eg = Cabana::get<FaceField::Edges>( t, k );
            toVC[detail::gidCoordRank( vg, size )].push_back(
                { vg, fg, static_cast<Rank>( R ) } );
            toEC[detail::gidCoordRank( eg, size )].push_back(
                { eg, static_cast<Rank>( R ) } );
        }
    }
    auto vAdv = allToAllV( comm, toVC );
    auto eAdv = allToAllV( comm, toEC );

    // Vertex coordinator: owner = min advertiser; reply owner to each advertiser
    // and hand the owner the list of incident faces owned by OTHER ranks.
    std::map<GlobalId, std::vector<detail::VtxInc>> byV;
    for ( const auto& m : vAdv.data )
        byV[m.vg].push_back( m );
    std::vector<std::vector<detail::GidOwn>> vOwnReply( size );
    std::vector<std::vector<detail::VtxInc>> ghostNeed( size );
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
    std::vector<std::vector<detail::GidOwn>> eOwnReply( size );
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
    std::vector<std::vector<detail::TupleBlob<FTuple>>> repF( size );
    std::vector<std::vector<detail::TupleBlob<VTuple>>> repV( size );
    std::vector<std::vector<detail::TupleBlob<ETuple>>> repE( size );
    for ( int s = 0; s < size; ++s )
    {
        const GlobalId* p = reqGot.from( s );
        const int c = reqGot.count( s );
        for ( int i = 0; i < c; ++i )
        {
            FTuple ft = faceById.at( p[i] );
            Cabana::get<FaceField::Owner>( ft ) = static_cast<Rank>( R );
            repF[s].push_back( detail::toBlob( ft ) );
            for ( int k = 0; k < 3; ++k )
            {
                const GlobalId vg = Cabana::get<FaceField::Verts>( ft, k );
                const GlobalId eg = Cabana::get<FaceField::Edges>( ft, k );
                VTuple vt = vById.at( vg );
                Cabana::get<VertexField::Owner>( vt ) = vOwner.at( vg );
                repV[s].push_back( detail::toBlob( vt ) );
                ETuple et = eById.at( eg );
                Cabana::get<EdgeField::Owner>( et ) = eOwner.at( eg );
                repE[s].push_back( detail::toBlob( et ) );
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
        FTuple t = detail::fromBlob( b );
        ghostFaceById[Cabana::get<FaceField::Gid>( t )] = t;
    }
    for ( const auto& b : ghV.data )
    {
        VTuple t = detail::fromBlob( b );
        const GlobalId vg = Cabana::get<VertexField::Gid>( t );
        if ( vById.find( vg ) == vById.end() )
        {
            vById.emplace( vg, t );
            vOwner[vg] = Cabana::get<VertexField::Owner>( t );
        }
    }
    for ( const auto& b : ghE.data )
    {
        ETuple t = detail::fromBlob( b );
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
    // Owned-first ordering (owned then ghost, each ascending gid) per kind.
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

    // Vertex AoSoA (tuple carries position + user fields; owner set explicitly).
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_MIGRATE_ASSEMBLE );
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
            ::Tessera::Profiling::TIMER_MIGRATE_ASSEMBLE );
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
            ::Tessera::Profiling::TIMER_MIGRATE_ASSEMBLE );
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
        detail::fillCsr( mesh.vertexFaces(), off, nbr, "vertex_faces" );
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
        detail::fillCsr( mesh.vertexEdges(), off, nbr, "vertex_edges" );
    }

    // Rebuild key side tables.
    {
        mesh.edgeKeys() = Kokkos::View<EdgeKey*, memory_space>(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
            nle );
        auto h_ek = Kokkos::create_mirror_view( mesh.edgeKeys() );
        for ( int li = 0; li < nle; ++li )
        {
            const ETuple& t = eById.at( eord[li] );
            h_ek( li ) = makeEdgeKey( Cabana::get<EdgeField::Verts>( t, 0 ),
                                      Cabana::get<EdgeField::Verts>( t, 1 ) );
        }
        Kokkos::deep_copy( mesh.edgeKeys(), h_ek );

        mesh.faceKeys() = Kokkos::View<FaceKey*, memory_space>(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
            nlf );
        auto h_fk = Kokkos::create_mirror_view( mesh.faceKeys() );
        for ( int li = 0; li < nlf; ++li )
        {
            const FTuple& t = ( li < nOwnedF ) ? faceById.at( ford[li] )
                                               : ghostFaceById.at( ford[li] );
            h_fk( li ) = makeFaceKey( Cabana::get<FaceField::Verts>( t, 0 ),
                                      Cabana::get<FaceField::Verts>( t, 1 ),
                                      Cabana::get<FaceField::Verts>( t, 2 ) );
        }
        Kokkos::deep_copy( mesh.faceKeys(), h_fk );
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

    halo.vplan = detail::buildKindPlan<memory_space>(
        comm, R, size, ghosts_of( vord, nOwnedV, vOwner ), v2l );
    halo.eplan = detail::buildKindPlan<memory_space>(
        comm, R, size, ghosts_of( eord, nOwnedE, eOwner ), e2l );
    halo.fplan = detail::buildKindPlan<memory_space>(
        comm, R, size, ghosts_of( ford, nOwnedF, fOwnerMap ), f2l );
}

// ============================================================================
// Read accessors for an external partitioner (Canopy / Zoltan2)
// ============================================================================
//
// These expose this rank's OWNED faces as the geometric primitives a partitioner
// needs. loadBalance() (Step 7b) feeds ownedFaceCentroids/Weights into Zoltan2 and
// drives the resulting assignment back through migrate(); Canopy calls the same
// accessors to compute a destination array from its FMM tree.

//! Centroid of each owned face, row-major [f*Dim + d] (host).
//!
//! An owned face can reference a vertex this rank does not hold locally after
//! refine() (see migrate()'s round G / the README post-refine gotcha), so any
//! missing endpoint position is gathered from its true owner via a gid
//! coordinator (gid % size) -- the same idiom migrate()/maxOwnedEdgeLength use.
//! A no-op when every reference is already held (the loadBalance() call on a
//! halo-consistent mesh, Canopy's accessors).
template <class MeshT>
std::vector<typename MeshT::scalar_type> ownedFaceCentroids( const MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    const int nv = static_cast<int>( mesh.numVertices() );
    const int nov = static_cast<int>( mesh.numOwnedVertices() );
    const int nof = static_cast<int>( mesh.numOwnedFaces() );
    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    // gid -> position for every locally-held vertex.
    std::map<GlobalId, std::array<Scalar, Dim>> posByGid;
    for ( int i = 0; i < nv; ++i )
    {
        std::array<Scalar, Dim> p;
        for ( int d = 0; d < Dim; ++d )
            p[d] = vp( i, d );
        posByGid[vg( i )] = p;
    }

    // Positions of owned-face endpoints not held locally, gathered from their
    // owner via a gid coordinator (advertise owned positions -> request misses).
    std::set<GlobalId> need;
    for ( int f = 0; f < nof; ++f )
        for ( int k = 0; k < 3; ++k )
            if ( posByGid.find( fv( f, k ) ) == posByGid.end() )
                need.insert( fv( f, k ) );
    long long localNeed = static_cast<long long>( need.size() );
    long long globalNeed = 0;
    MPI_Allreduce( &localNeed, &globalNeed, 1, MPI_LONG_LONG, MPI_SUM, comm );
    if ( globalNeed > 0 )
    {
        struct PosMsg
        {
            GlobalId gid;
            Scalar pos[Dim];
        };
        std::vector<std::vector<PosMsg>> adv( size );
        for ( int i = 0; i < nov; ++i ) // owned-first: [0, nov) are owned
        {
            PosMsg m;
            m.gid = vg( i );
            for ( int d = 0; d < Dim; ++d )
                m.pos[d] = vp( i, d );
            adv[detail::gidCoordRank( m.gid, size )].push_back( m );
        }
        auto advGot = allToAllV( comm, adv );
        std::map<GlobalId, std::array<Scalar, Dim>> coord;
        for ( const auto& m : advGot.data )
        {
            std::array<Scalar, Dim> p;
            for ( int d = 0; d < Dim; ++d )
                p[d] = m.pos[d];
            coord[m.gid] = p;
        }
        std::vector<std::vector<GlobalId>> req( size );
        for ( GlobalId g : need )
            req[detail::gidCoordRank( g, size )].push_back( g );
        auto reqGot = allToAllV( comm, req );
        std::vector<std::vector<PosMsg>> rep( size );
        for ( int s = 0; s < size; ++s )
        {
            const GlobalId* p = reqGot.from( s );
            const int c = reqGot.count( s );
            for ( int i = 0; i < c; ++i )
            {
                PosMsg m;
                m.gid = p[i];
                const auto& pp = coord.at( p[i] );
                for ( int d = 0; d < Dim; ++d )
                    m.pos[d] = pp[d];
                rep[s].push_back( m );
            }
        }
        auto repGot = allToAllV( comm, rep );
        for ( const auto& m : repGot.data )
        {
            std::array<Scalar, Dim> p;
            for ( int d = 0; d < Dim; ++d )
                p[d] = m.pos[d];
            posByGid[m.gid] = p;
        }
    }

    std::vector<Scalar> c( static_cast<std::size_t>( nof ) * Dim, Scalar( 0 ) );
    for ( int f = 0; f < nof; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const auto& p = posByGid.at( fv( f, k ) );
            for ( int d = 0; d < Dim; ++d )
                c[static_cast<std::size_t>( f ) * Dim + d] +=
                    p[d] / Scalar( 3 );
        }
    return c;
}

//! Global id of each owned face (host).
template <class MeshT>
std::vector<GlobalId> ownedFaceGids( const MeshT& mesh )
{
    const int nof = static_cast<int>( mesh.numOwnedFaces() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    std::vector<GlobalId> out( nof );
    for ( int f = 0; f < nof; ++f )
        out[f] = g( f );
    return out;
}

//! Per-owned-face work weight for a partitioner. Milestone 1 treats every leaf
//! face as one unit of work; a refinement-descendant-aware weight can be layered
//! in later without changing the migrate()/loadBalance() contract.
template <class MeshT>
std::vector<double> ownedFaceWeights( const MeshT& mesh )
{
    return std::vector<double>( mesh.numOwnedFaces(), 1.0 );
}

} // namespace Tessera

#endif // TESSERA_MESH_MIGRATE_HPP
