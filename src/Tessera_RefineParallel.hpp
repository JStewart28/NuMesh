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

#ifndef TESSERA_REFINE_PARALLEL_HPP
#define TESSERA_REFINE_PARALLEL_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Refine.hpp"
#include "Tessera_RefinePolicy.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// Distributed 2:1-balanced red refinement (Step 6b)
// ============================================================================
//
// refine() refines a distributed mesh's OWNED faces (mask indexed by owned face
// local index) with a conforming 2:1 level balance enforced across partition
// boundaries and — the central guarantee — a midpoint vertex whose global id is
// BIT-IDENTICAL on every rank that shares the bisected edge.
//
// It relies on neither the vertex-based 1-ring halo (which does not guarantee a
// face sees its edge-neighbour across a boundary) nor replicated knowledge.
// Instead every cross-rank decision is routed through an EDGE COORDINATOR: the
// rank `edgeCoordRank(EdgeKey)` gathers both incident faces of an edge (each
// owned by some rank and advertised) and drives the decision. Three coordinator
// phases run:
//   1. 2:1 mark-propagation fixpoint: pure function of (level, mark) per face;
//      coordinator flags the coarser face of any edge whose incident final
//      levels would differ by >1; marks only ever turn on (monotone) so the
//      fixpoint terminates, guarded by MPI_Allreduce(changed) and a hard cap.
//   2. Midpoint-gid assignment: for each split edge the midpoint owner is the
//      lowest incident refining-face owner; owners count their midpoints,
//      MPI_Exscan a global contiguous block onto the pre-refinement global vertex
//      count, assign, and SEND the gid to co-sharers — so both sides agree with
//      no reliance on identical local ordering.
//   3. Edge ownership: each refined edge is owned by the lowest incident (child)
//      face owner, so owned counts remain a global partition (for Euler).
//
// Local topology is then rebuilt exactly as the serial engine (Tessera_Refine),
// reusing the interpolation policy. Because the halo rebuild for a distributed
// (non-replicated) mesh is shared with migration, it is deferred to Step 7: this
// routine leaves each rank holding only its refined OWNED entities (owned counts
// == local counts) and CLEARS the passed halo. A haloExchange() must not be
// called until the halo is rebuilt (Step 7). The acceptance invariants (midpoint
// agreement, 2:1 balance, owned-only Euler) are all owned-entity properties and
// need no halo.
//
// Preconditions: `mesh` is distributed (post-distribute), owned-first, entities
// carry global gids, and is 2:1-balanced on entry (true initially — uniform
// level 0 — and maintained by each call). `mask.size() == numOwnedFaces()`.

struct RefineResult
{
    //! 2:1 mark-propagation rounds executed to reach the fixpoint.
    int iterations = 0;
    //! (bisected edge, midpoint gid) for every split edge this rank participates
    //! in — owned midpoints it assigned and shared ones it received. Used to
    //! verify cross-rank gid agreement.
    std::vector<std::pair<EdgeKey, GlobalId>> midpoints;
};

namespace detail
{

//! Deterministic coordinator rank for an edge (routing only; disambiguated by
//! the full EdgeKey at the coordinator).
inline int edgeCoordRank( const EdgeKey& k, int comm_size )
{
    unsigned long long h = k.id[0] * 1099511628211ULL ^ k.id[1];
    return static_cast<int>( h % static_cast<unsigned long long>( comm_size ) );
}

//! One (edge, incident owned face) advertisement for the 2:1 fixpoint.
struct PropMsg
{
    EdgeKey key;
    GlobalId faceGid;
    Rank owner;
    Level level;
    unsigned char mark;
};
//! (edge, incident refining-face owner) advertisement for midpoint ownership.
struct OwnerMsg
{
    EdgeKey key;
    Rank owner;
};
//! (edge, its midpoint owner) reply delivered to each participant.
struct OwnerReply
{
    EdgeKey key;
    Rank owner;
};
//! (owned edge, a co-sharer rank) delivered to the midpoint owner.
struct CosharerMsg
{
    EdgeKey key;
    Rank cosharer;
};
//! (edge, assigned midpoint gid) delivered from owner to co-sharer.
struct KeyGid
{
    EdgeKey key;
    GlobalId gid;
};
//! (edge, incident child-face owner, incident child-face level) for edge
//! ownership + level in the refined mesh.
struct EdgeOwnMsg
{
    EdgeKey key;
    Rank owner;
    Level level;
};

} // namespace detail

//! Distributed 2:1-balanced red refinement of the owned faces flagged in `mask`.
//! See the header comment for the algorithm, guarantees, and the (deferred) halo.
template <class MeshT,
          class Policy = DefaultRefinePolicy<typename MeshT::scalar_type>>
RefineResult refine( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                     const std::vector<char>& mask,
                     const Policy& policy = Policy{} )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_REFINE );
    using memory_space = typename MeshT::memory_space;
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int nv = static_cast<int>( mesh.numVertices() ); // owned + ghost
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );
    const int nOwnedV = static_cast<int>( mesh.numOwnedVertices() );

    RefineResult result;

    // ---- host copies: all vertices (endpoint lookup) + owned faces ----------
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_lev = Cabana::slice<FaceField::Level>( hf );

    std::unordered_map<GlobalId, int> gid2lv; // vertex gid -> local index
    gid2lv.reserve( nv * 2 );
    for ( int i = 0; i < nv; ++i )
        gid2lv[v_gid( i )] = i;

    // owned-face topology snapshot
    std::vector<std::array<GlobalId, 3>> fV( nOwnedF );
    std::vector<GlobalId> fG( nOwnedF );
    std::vector<Level> fL( nOwnedF );
    for ( int f = 0; f < nOwnedF; ++f )
    {
        for ( int k = 0; k < 3; ++k )
            fV[f][k] = f_verts( f, k );
        fG[f] = f_gid( f );
        fL[f] = f_lev( f );
    }
    auto keyOf = []( GlobalId a, GlobalId b ) { return makeEdgeKey( a, b ); };

    // ---- Phase 1: 2:1 mark-propagation fixpoint -----------------------------
    std::vector<char> mark( nOwnedF, 0 );
    for ( int f = 0; f < nOwnedF && f < static_cast<int>( mask.size() ); ++f )
        mark[f] = mask[f] ? 1 : 0;
    std::unordered_map<GlobalId, int> gid2of; // owned face gid -> index
    gid2of.reserve( nOwnedF * 2 );
    for ( int f = 0; f < nOwnedF; ++f )
        gid2of[fG[f]] = f;

    const int kCap = 256;
    int iter = 0;
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_REFINE_BALANCE );
        while ( true )
        {
            std::map<EdgeKey, std::vector<detail::PropMsg>> byEdge;
            {
                TESSERA_SCOPED_TIMER_VERBOSE(
                    ::Tessera::Profiling::TIMER_REFINE_ADVERTISE );
                std::vector<std::vector<detail::PropMsg>> toCoord( size );
                for ( int f = 0; f < nOwnedF; ++f )
                    for ( int k = 0; k < 3; ++k )
                    {
                        const EdgeKey key =
                            keyOf( fV[f][k], fV[f][( k + 1 ) % 3] );
                        toCoord[detail::edgeCoordRank( key, size )].push_back(
                            { key, fG[f], static_cast<Rank>( R ), fL[f],
                              static_cast<unsigned char>( mark[f] ) } );
                    }
                auto got = allToAllV( comm, toCoord );
                for ( const auto& m : got.data )
                    byEdge[m.key].push_back( m );
            }

            int changed = 0;
            {
                TESSERA_SCOPED_TIMER_VERBOSE(
                    ::Tessera::Profiling::TIMER_REFINE_MARKREQ );
                std::vector<std::vector<GlobalId>> markReq( size );
                for ( auto& kv : byEdge )
                {
                    auto& inc = kv.second;
                    if ( inc.size() != 2 )
                        continue; // closed surface: exactly two incident faces
                    const int fa = inc[0].level + inc[0].mark;
                    const int fb = inc[1].level + inc[1].mark;
                    if ( fa - fb >= 2 )
                        markReq[inc[1].owner].push_back( inc[1].faceGid );
                    else if ( fb - fa >= 2 )
                        markReq[inc[0].owner].push_back( inc[0].faceGid );
                }
                auto reqs = allToAllV( comm, markReq );

                for ( const GlobalId g : reqs.data )
                {
                    auto it = gid2of.find( g );
                    if ( it != gid2of.end() && !mark[it->second] )
                    {
                        mark[it->second] = 1;
                        ++changed;
                    }
                }
            }
            int globalChanged = 0;
            MPI_Allreduce( &changed, &globalChanged, 1, MPI_INT, MPI_SUM,
                           comm );
            ++iter;
            if ( globalChanged == 0 || iter >= kCap )
                break;
        }
    }
    result.iterations = iter;

    // ---- Phase 2: midpoint-gid assignment with cross-rank agreement ---------
    // 2a. advertise each refining face's edges (owner) to the coordinator.
    {
        std::vector<std::vector<detail::OwnerMsg>> toCoord( size );
        for ( int f = 0; f < nOwnedF; ++f )
        {
            if ( !mark[f] )
                continue;
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key = keyOf( fV[f][k], fV[f][( k + 1 ) % 3] );
                toCoord[detail::edgeCoordRank( key, size )].push_back(
                    { key, static_cast<Rank>( R ) } );
            }
        }
        auto got = allToAllV( comm, toCoord );

        // 2b. coordinator: per split edge, owner = min incident refining owner;
        //     reply (key, owner) to every participant, and (key, cosharer) to the
        //     owner for each other participant.
        std::map<EdgeKey, std::set<Rank>> owners;
        for ( const auto& m : got.data )
            owners[m.key].insert( m.owner );

        std::vector<std::vector<detail::OwnerReply>> reply( size );
        std::vector<std::vector<detail::CosharerMsg>> coshare( size );
        for ( auto& kv : owners )
        {
            const EdgeKey& key = kv.first;
            const std::set<Rank>& os = kv.second;
            const Rank owner = *os.begin(); // std::set is ordered ascending
            for ( Rank r : os )
            {
                reply[r].push_back( { key, owner } );
                if ( r != owner )
                    coshare[owner].push_back( { key, r } );
            }
        }
        auto replies = allToAllV( comm, reply );
        auto coshares = allToAllV( comm, coshare );

        // 2c. this rank now knows the owner of every split edge it touches.
        std::map<EdgeKey, Rank> midOwner;
        for ( const auto& m : replies.data )
            midOwner[m.key] = m.owner;
        std::map<EdgeKey, std::vector<Rank>> myCosharers;
        for ( const auto& m : coshares.data )
            myCosharers[m.key].push_back( m.cosharer );

        // owned midpoints (deterministic order), assign global gids via exscan.
        std::vector<EdgeKey> myMid;
        for ( const auto& kv : midOwner )
            if ( kv.second == R )
                myMid.push_back( kv.first );
        std::sort( myMid.begin(), myMid.end() );

        long long localOwnedV = nOwnedV;
        long long globalV = 0;
        MPI_Allreduce( &localOwnedV, &globalV, 1, MPI_LONG_LONG, MPI_SUM,
                       comm );
        long long myCount = static_cast<long long>( myMid.size() );
        long long baseOff = 0;
        MPI_Exscan( &myCount, &baseOff, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            baseOff = 0;

        std::map<EdgeKey, GlobalId> midGid;
        for ( std::size_t i = 0; i < myMid.size(); ++i )
            midGid[myMid[i]] = static_cast<GlobalId>(
                globalV + baseOff + static_cast<long long>( i ) );

        // deliver owned midpoint gids to co-sharers.
        std::vector<std::vector<detail::KeyGid>> toShare( size );
        for ( const EdgeKey& key : myMid )
        {
            auto it = myCosharers.find( key );
            if ( it == myCosharers.end() )
                continue;
            for ( Rank c : it->second )
                toShare[c].push_back( { key, midGid[key] } );
        }
        auto shared = allToAllV( comm, toShare );
        for ( const auto& m : shared.data )
            midGid[m.key] = m.gid;

        // record for verification + hand to local reconstruction below.
        result.midpoints.reserve( midGid.size() );
        for ( const auto& kv : midGid )
            result.midpoints.push_back( { kv.first, kv.second } );

        // ---- Phase 3: local topology reconstruction (owned only) ------------
        auto midOf = [&]( GlobalId a, GlobalId b ) -> GlobalId
        { return midGid.at( keyOf( a, b ) ); };

        // 3a. new owned vertices: originals + owned midpoints.
        const int nNewV = nOwnedV + static_cast<int>( myMid.size() );
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            lv( "lv", nNewV );
        {
            auto gid = Cabana::slice<VertexField::Gid>( lv );
            auto own = Cabana::slice<VertexField::Owner>( lv );
            auto flg = Cabana::slice<VertexField::Flags>( lv );
            auto pos = Cabana::slice<VertexField::Position>( lv );
            auto o_flg = Cabana::slice<VertexField::Flags>( hv );
            for ( int i = 0; i < nOwnedV; ++i )
            {
                gid( i ) = v_gid( i );
                own( i ) = static_cast<Rank>( R );
                flg( i ) = o_flg( i );
                for ( int d = 0; d < Dim; ++d )
                    pos( i, d ) = v_pos( i, d );
                detail::copyUserFields<VertexField::UserBegin>( lv, i, hv, i );
            }
            for ( std::size_t m = 0; m < myMid.size(); ++m )
            {
                const int idx = nOwnedV + static_cast<int>( m );
                const GlobalId a = myMid[m].id[0];
                const GlobalId b = myMid[m].id[1];
                const int la = gid2lv.at( a );
                const int lb = gid2lv.at( b );
                gid( idx ) = midGid[myMid[m]];
                own( idx ) = static_cast<Rank>( R );
                flg( idx ) = 0;
                Scalar pa[Dim], pb[Dim], pm[Dim];
                for ( int d = 0; d < Dim; ++d )
                {
                    pa[d] = v_pos( la, d );
                    pb[d] = v_pos( lb, d );
                }
                policy.interpolatePosition( pm, pa, pb, Dim );
                for ( int d = 0; d < Dim; ++d )
                    pos( idx, d ) = pm[d];
                detail::blendVertexUserCross( lv, idx, hv, la, lb, policy );
            }
        }
        // INVALIDATION: the resize/deep_copy calls in this function, the key-
        // View reassignment, and the CSR rebuild below reallocate and reassign
        // this mesh's storage, invalidating every slice/CSR/key-View handed out
        // before this call. Re-slice from the mesh after refine() returns.
        mesh.resizeVertices( nNewV );
        Cabana::deep_copy( mesh.vertices(), lv );

        // 3b. new owned faces: kept (retain gid) + 4 children (new gids).
        //
        // Child gids are appended above the current global MAX face gid, not
        // above the face COUNT: a refined parent's gid is retired (it is replaced
        // by 4 children), so after any round the live gids are sparse and the max
        // exceeds the count. Basing new gids on the count would collide with the
        // previous round's high-numbered children on a later refine (surfacing as
        // duplicate face gids once migrate() brings two ranks' faces together).
        long long localMaxF = -1;
        for ( int f = 0; f < nOwnedF; ++f )
            localMaxF = std::max( localMaxF, static_cast<long long>( fG[f] ) );
        long long globalMaxF = -1;
        MPI_Allreduce( &localMaxF, &globalMaxF, 1, MPI_LONG_LONG, MPI_MAX,
                       comm );
        int nRefining = 0;
        for ( int f = 0; f < nOwnedF; ++f )
            nRefining += mark[f] ? 1 : 0;
        long long myChild = 4LL * nRefining;
        long long childBase = 0;
        MPI_Exscan( &myChild, &childBase, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            childBase = 0;
        GlobalId childGid = static_cast<GlobalId>( globalMaxF + 1 + childBase );

        std::vector<std::array<GlobalId, 3>> nFV;
        std::vector<GlobalId> nFG;
        std::vector<Level> nFL;
        std::vector<int> nFParent;
        for ( int f = 0; f < nOwnedF; ++f )
        {
            const GlobalId a = fV[f][0], b = fV[f][1], c = fV[f][2];
            if ( !mark[f] )
            {
                nFV.push_back( { a, b, c } );
                nFG.push_back( fG[f] );
                nFL.push_back( fL[f] );
                nFParent.push_back( f );
                continue;
            }
            const GlobalId ab = midOf( a, b );
            const GlobalId bc = midOf( b, c );
            const GlobalId ca = midOf( c, a );
            const std::array<GlobalId, 3> ch[4] = {
                { a, ab, ca }, { b, bc, ab }, { c, ca, bc }, { ab, bc, ca } };
            const Level clev = static_cast<Level>( fL[f] + 1 );
            for ( const auto& q : ch )
            {
                nFV.push_back( q );
                nFG.push_back( childGid++ );
                nFL.push_back( clev );
                nFParent.push_back( f );
            }
        }
        const int nNewF = static_cast<int>( nFV.size() );

        // 3c. re-derive edges from the new owned faces (dedup by EdgeKey).
        std::map<EdgeKey, int> edge_of;
        std::vector<std::array<GlobalId, 2>> ep;
        std::vector<std::array<int, 2>> efLocal; // incident local face indices
        std::vector<std::array<int, 3>> faceEdge( nNewF );
        for ( int f = 0; f < nNewF; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key = keyOf( nFV[f][k], nFV[f][( k + 1 ) % 3] );
                auto it = edge_of.find( key );
                int e;
                if ( it == edge_of.end() )
                {
                    e = static_cast<int>( ep.size() );
                    edge_of.emplace( key, e );
                    ep.push_back( { key.id[0], key.id[1] } );
                    efLocal.push_back( { -1, -1 } );
                }
                else
                    e = it->second;
                faceEdge[f][k] = e;
                if ( efLocal[e][0] < 0 )
                    efLocal[e][0] = f;
                else
                    efLocal[e][1] = f;
            }
        const int nLocalE = static_cast<int>( ep.size() );

        // 3d. edge ownership + level via coordinator (min incident child-face
        //     owner / level). Every edge is owned by exactly one rank globally.
        std::vector<std::vector<detail::EdgeOwnMsg>> toEC( size );
        for ( int e = 0; e < nLocalE; ++e )
        {
            const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
            Level lv0 = nFL[efLocal[e][0]];
            if ( efLocal[e][1] >= 0 )
                lv0 = std::min( lv0, nFL[efLocal[e][1]] );
            toEC[detail::edgeCoordRank( key, size )].push_back(
                { key, static_cast<Rank>( R ), lv0 } );
        }
        auto ecGot = allToAllV( comm, toEC );
        std::map<EdgeKey, std::pair<Rank, Level>> edgeAgg;
        for ( const auto& m : ecGot.data )
        {
            auto it = edgeAgg.find( m.key );
            if ( it == edgeAgg.end() )
                edgeAgg[m.key] = { m.owner, m.level };
            else
            {
                it->second.first = std::min( it->second.first, m.owner );
                it->second.second = std::min( it->second.second, m.level );
            }
        }
        // Reply to each participant, grouped by source rank via from()/count().
        std::vector<std::vector<detail::EdgeOwnMsg>> ecReply( size );
        for ( int s = 0; s < size; ++s )
        {
            const detail::EdgeOwnMsg* p = ecGot.from( s );
            const int cnt = ecGot.count( s );
            for ( int i = 0; i < cnt; ++i )
            {
                const auto& agg = edgeAgg[p[i].key];
                ecReply[s].push_back( { p[i].key, agg.first, agg.second } );
            }
        }
        auto ecRes = allToAllV( comm, ecReply );
        std::map<EdgeKey, std::pair<Rank, Level>> edgeOwner;
        for ( const auto& m : ecRes.data )
            edgeOwner[m.key] = { m.owner, m.level };

        // 3e. order edges owned-first, assign globally-unique gids (exscan over
        //     local edge count; boundary edges carry distinct gids per rank but
        //     are matched by EdgeKey and counted once via the owner field).
        std::vector<int> order; // local edge indices, owned first
        order.reserve( nLocalE );
        std::vector<char> isOwned( nLocalE, 0 );
        for ( int e = 0; e < nLocalE; ++e )
        {
            const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
            if ( edgeOwner[key].first == R )
            {
                isOwned[e] = 1;
                order.push_back( e );
            }
        }
        const int nOwnedE = static_cast<int>( order.size() );
        for ( int e = 0; e < nLocalE; ++e )
            if ( !isOwned[e] )
                order.push_back( e );

        long long localE = nLocalE;
        long long eBase = 0;
        MPI_Exscan( &localE, &eBase, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            eBase = 0;
        std::vector<GlobalId> edgeGid( nLocalE );
        std::vector<int> newIndexOf( nLocalE );
        for ( int li = 0; li < nLocalE; ++li )
        {
            const int e = order[li];
            edgeGid[e] = static_cast<GlobalId>( eBase + li );
            newIndexOf[e] = li;
        }

        // 3f. materialize the edge AoSoA (owned-first).
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_REBUILD );
            Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace>
                le( "le", nLocalE );
            auto gid = Cabana::slice<EdgeField::Gid>( le );
            auto own = Cabana::slice<EdgeField::Owner>( le );
            auto lev = Cabana::slice<EdgeField::Level>( le );
            auto verts = Cabana::slice<EdgeField::Verts>( le );
            auto faces = Cabana::slice<EdgeField::Faces>( le );
            for ( int e = 0; e < nLocalE; ++e )
            {
                const int li = newIndexOf[e];
                const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
                gid( li ) = edgeGid[e];
                own( li ) = edgeOwner[key].first;
                lev( li ) = edgeOwner[key].second;
                verts( li, 0 ) = ep[e][0];
                verts( li, 1 ) = ep[e][1];
                // Incident faces: local child gids; the cross-rank second face is
                // filled when the halo is rebuilt (Step 7).
                faces( li, 0 ) = nFG[efLocal[e][0]];
                faces( li, 1 ) =
                    efLocal[e][1] >= 0 ? nFG[efLocal[e][1]] : invalid_gid;
            }
            mesh.resizeEdges( nLocalE );
            Cabana::deep_copy( mesh.edges(), le );
        }

        // 3g. materialize the face AoSoA (edges now have gids).
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_REBUILD );
            Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace>
                lf( "lf", nNewF );
            auto gid = Cabana::slice<FaceField::Gid>( lf );
            auto own = Cabana::slice<FaceField::Owner>( lf );
            auto lev = Cabana::slice<FaceField::Level>( lf );
            auto verts = Cabana::slice<FaceField::Verts>( lf );
            auto edges = Cabana::slice<FaceField::Edges>( lf );
            for ( int f = 0; f < nNewF; ++f )
            {
                gid( f ) = nFG[f];
                own( f ) = static_cast<Rank>( R );
                lev( f ) = nFL[f];
                for ( int k = 0; k < 3; ++k )
                {
                    verts( f, k ) = nFV[f][k];
                    edges( f, k ) = edgeGid[faceEdge[f][k]];
                }
                detail::copyUserFields<FaceField::UserBegin>( lf, f, hf,
                                                              nFParent[f] );
            }
            mesh.resizeFaces( nNewF );
            Cabana::deep_copy( mesh.faces(), lf );
        }

        // owned-only mesh: every local entity is owned.
        mesh.setOwnedCounts( nNewV, nOwnedE, nNewF );

        // 3h. rebuild key side tables.
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_REBUILD );
            Kokkos::View<EdgeKey*, memory_space> ek(
                Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
                nLocalE );
            auto h_ek = Kokkos::create_mirror_view( ek );
            for ( int e = 0; e < nLocalE; ++e )
                h_ek( newIndexOf[e] ) = keyOf( ep[e][0], ep[e][1] );
            Kokkos::deep_copy( ek, h_ek );
            mesh.setEdgeKeys( ek );

            Kokkos::View<FaceKey*, memory_space> fk(
                Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
                nNewF );
            auto h_fk = Kokkos::create_mirror_view( fk );
            for ( int f = 0; f < nNewF; ++f )
                h_fk( f ) = makeFaceKey( nFV[f][0], nFV[f][1], nFV[f][2] );
            Kokkos::deep_copy( fk, h_fk );
            mesh.setFaceKeys( fk );
        }

        // 3i. best-effort owned-vertex 1-ring CSR over LOCAL faces/edges. It is
        //     incomplete at partition boundaries (ghost faces/edges are dropped
        //     until the Step-7 halo rebuild); provided so the container is not
        //     stale rather than as a complete 1-ring.
        std::unordered_map<GlobalId, int> gid2nv;
        gid2nv.reserve( nNewV * 2 );
        {
            auto gid = Cabana::slice<VertexField::Gid>( lv );
            for ( int i = 0; i < nNewV; ++i )
                gid2nv[gid( i )] = i;
        }
        {
            std::vector<int> off( nNewV + 1, 0 );
            for ( int f = 0; f < nNewF; ++f )
                for ( int k = 0; k < 3; ++k )
                {
                    auto it = gid2nv.find( nFV[f][k] );
                    if ( it != gid2nv.end() )
                        ++off[it->second + 1];
                }
            for ( int i = 0; i < nNewV; ++i )
                off[i + 1] += off[i];
            std::vector<LocalIndex> nbr( off.back() );
            std::vector<int> cur( off.begin(), off.end() );
            for ( int f = 0; f < nNewF; ++f )
                for ( int k = 0; k < 3; ++k )
                {
                    auto it = gid2nv.find( nFV[f][k] );
                    if ( it != gid2nv.end() )
                        nbr[cur[it->second]++] = static_cast<LocalIndex>( f );
                }
            mesh.rebuildVertexFaces( off, nbr, "vertex_faces" );
        }
        {
            std::vector<int> off( nNewV + 1, 0 );
            for ( int e = 0; e < nLocalE; ++e )
                for ( int j = 0; j < 2; ++j )
                {
                    auto it = gid2nv.find( ep[e][j] );
                    if ( it != gid2nv.end() )
                        ++off[it->second + 1];
                }
            for ( int i = 0; i < nNewV; ++i )
                off[i + 1] += off[i];
            std::vector<LocalIndex> nbr( off.back() );
            std::vector<int> cur( off.begin(), off.end() );
            for ( int e = 0; e < nLocalE; ++e )
                for ( int j = 0; j < 2; ++j )
                {
                    auto it = gid2nv.find( ep[e][j] );
                    if ( it != gid2nv.end() )
                        nbr[cur[it->second]++] =
                            static_cast<LocalIndex>( newIndexOf[e] );
                }
            mesh.rebuildVertexEdges( off, nbr, "vertex_edges" );
        }
    }

    // INVALIDATION: the 1-deep halo is now stale (topology changed, ghosts
    // dropped), as is every slice/CSR/key-View a caller took out before this
    // refine() call. Clear the halo plans; Step 7 provides the general
    // (non-replicated) halo rebuild. Callers must re-slice from the mesh.
    halo.vplan.clear();
    halo.eplan.clear();
    halo.fplan.clear();

    return result;
}

} // namespace Tessera

#endif // TESSERA_REFINE_PARALLEL_HPP
