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

#ifndef TESSERA_DISTRIBUTED_BUILDER_HPP
#define TESSERA_DISTRIBUTED_BUILDER_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"  // MeshHalo
#include "Tessera_HaloRebuild.hpp" // rebuildHalo
#include "Tessera_Icosphere.hpp"   // TriangleSoup, detail::normalize3
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_RefineClosure.hpp"  // initClosureFaceMembers
#include "Tessera_RefineParallel.hpp" // detail::edgeCoordRank
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// Distributed initial mesh construction
// ============================================================================
//
// buildFromTriangleSoup() is host-side, serial, and takes a REPLICATED soup;
// buildIcosphere() generates the whole soup on every rank and hands it over; and
// distribute() can then compute ownership locally precisely BECAUSE the mesh is
// replicated. So peak memory per rank is proportional to the GLOBAL mesh size,
// paid simultaneously on every rank, and the `std::map<EdgeKey,int>` inside the
// serial builder compounds it. That is a hard ceiling exactly when the initial
// mesh's resolution should be comparable to the running refined mesh — the normal
// case for a production run that does not want to spend its first hundred steps
// refining up from a coarse sphere — and it is the reason a consumer that builds
// its own surfaces in parallel (the Beatnik z-model) currently cannot hand them
// to Tessera.
//
// This header adds the two entry points that remove the replication requirement.
// distribute() is NOT on this path at all, and buildFromTriangleSoup() stays
// exactly as it is for the replicated/serial case — it is the right tool for a
// small initial mesh and for every pre-existing test.
//
//   buildFromTriangleSoupDistributed()  the general capability: PER-RANK LOCAL
//                                       triangle patches plus a canonical key per
//                                       local vertex.
//   buildIcosphereDistributed()         the concrete case that proves it, and the
//                                       drop-in replacement for
//                                       buildIcosphere() + distribute().
//
// THE CANONICAL KEY IS THE WHOLE DESIGN. Deduplicating a vertex two patches share
// is the one thing that genuinely needs global agreement. A position-based dedup
// would need a tolerance and would not be reproducible; a hash of the position
// would silently weld distinct vertices. Requiring the CALLER to supply a
// rank-independent key pushes the problem to where the answer is known for free:
// a generator always knows *why* two patches share a vertex. `VertexKey` reuses
// the existing EdgeKey machinery's shape (a sorted pair of GlobalId, 128 bits, no
// hashing), which covers the two cases that matter:
//
//   base vertex    makeVertexKey( i )        == { i, invalid_gid }
//   midpoint       makeVertexKey( a, b )     == { min(a,b), max(a,b) }
//
// and a caller with a different scheme hashes into it at its own risk. The
// contract is exactly three properties, and the third is CHECKED:
//
//   1. RANK-INDEPENDENT. Two ranks meaning the same vertex compute the same key
//      with no communication.
//   2. EQUAL IFF THE SAME VERTEX. Patches must COVER the surface with no gaps;
//      OVERLAP is allowed and is resolved by the key dedup, so a caller may
//      generate a patch plus a boundary ring.
//   3. COLLISIONS THROW. Two vertices with the same key but different positions
//      are a caller bug, not a weld: the key coordinator compares every
//      claimant's position BITWISE and every rank throws naming the key. Silently
//      welding them would produce a mesh that passes every structural invariant
//      and is geometrically wrong.
//
// The implementation is a reuse of machinery that already exists, in five steps:
//
//   1. VERTEX DEDUP + GID ASSIGNMENT. Route each local vertex's key to a
//      coordinator by key hash via allToAllV. The coordinator sees every rank
//      claiming that key, picks the owner by the lowest-rank rule, numbers its own
//      distinct keys densely from an MPI_Exscan over its key count, and replies
//      (key -> gid, owner) to every claimant. Exactly refine() Phase 2's pattern
//      with midpoint keys replaced by vertex keys.
//   2. FACE DEDUP. A duplicate triangle (two overlapping patches) is owned by the
//      lowest rank claiming its FaceKey — the existing key machinery already
//      defines that identity. One coordinator round, then face gids from an
//      MPI_Exscan over the surviving owned counts, so a rank's face gids are
//      CONTIGUOUS (the subdivision-tree partition below is locality-preserving,
//      and contiguous gids keep it observable).
//   3. EDGE DERIVATION, LOCALLY. Each rank derives the unique edges of ITS OWN
//      owned triangles with a local std::map<EdgeKey,int> — O(local) memory, which
//      is the entire point. Edge gids and ownership then go through
//      detail::edgeCoordRank exactly as in refine()'s Phase 3e, so an edge shared
//      across a patch boundary gets one gid agreed by both sides.
//   4. ASSEMBLE the owned entities into the mesh's AoSoAs.
//   5. GHOST LAYER AND HALO by calling rebuildHalo( mesh, halo, haloDepth ).
//
// STEP 5 IS THE STRUCTURAL REASON THIS IS FEASIBLE AT ALL. rebuildHalo()'s round B
// resolves ownership BY COMMUNICATION, so this builder needs none of the
// replicated-mesh ownership shortcut distribute() relies on; and its round G
// recovers any vertex/edge an owned face references but this rank does not hold,
// so steps 1-4 only ever have to produce the OWNED entities. Everything after step
// 4 is the code path refine() and migrate() already exercise on every gate run.

//! A vertex's CANONICAL KEY: a rank-independent 128-bit identifier that is equal
//! on two ranks exactly when they mean the same vertex. Same shape and same
//! guarantees as EdgeKey — a sorted pair of GlobalId, structured rather than
//! hashed, so it is order-invariant and collision-free by construction.
using VertexKey = Key<2>;

//! Canonical key of a BASE vertex (`makeVertexKey( i )`, second slot the invalid
//! sentinel) or of the MIDPOINT of two already-identified vertices
//! (`makeVertexKey( a, b )`, order-invariant). `invalid_gid` is the largest
//! GlobalId, so a base key always sorts as `{ i, invalid_gid }` and can never
//! collide with a midpoint key (no real vertex is identified by `invalid_gid`).
KOKKOS_INLINE_FUNCTION
VertexKey makeVertexKey( GlobalId a, GlobalId b = invalid_gid )
{
    return makeEdgeKey( a, b );
}

namespace detail
{

//! Deterministic coordinator rank for a vertex key. Routing only — the full
//! 128-bit key disambiguates at the coordinator, exactly as for an EdgeKey — so
//! sharing edgeCoordRank()'s mixing function is sound and keeps one hash in the
//! library rather than two.
inline int vertexKeyCoordRank( const VertexKey& k, int comm_size )
{
    return edgeCoordRank( k, comm_size );
}

//! Deterministic coordinator rank for a face key (routing only; disambiguated by
//! the full FaceKey at the coordinator).
inline int faceKeyCoordRank( const FaceKey& k, int comm_size )
{
    unsigned long long h = k.id[0] * 1099511628211ULL;
    h = ( h ^ k.id[1] ) * 1099511628211ULL;
    h ^= k.id[2];
    return static_cast<int>( h % static_cast<unsigned long long>( comm_size ) );
}

//! (vertex key, the claimant's position for it) advertisement. The position rides
//! along so the coordinator can CHECK the caller's key contract rather than trust
//! it; it is carried as `double` regardless of the mesh's Scalar, which is exact
//! for float and leaves the bitwise comparison equivalent either way.
struct VKeyClaim
{
    VertexKey key;
    double pos[3];
};
//! (vertex key, its assigned gid, its resolved owner) reply to every claimant.
struct VKeyGid
{
    VertexKey key;
    GlobalId gid;
    Rank owner;
};
//! (face key) claim / (face key, resolved owner) reply.
struct FKeyClaim
{
    FaceKey key;
};
struct FKeyOwn
{
    FaceKey key;
    Rank owner;
};
//! (edge key, its assigned gid, its resolved owner). `gid` is unused on the
//! outbound advertisement and carries the coordinator's dense numbering back.
struct DEdgeKeyGid
{
    EdgeKey key;
    GlobalId gid;
    Rank owner;
};

//! Bitwise equality of two doubles. Deliberately not `==`: two identical NaN bit
//! patterns are the SAME position (and must not be reported as a key collision),
//! while `==` would call them different, and two zeros of opposite sign are
//! different bit patterns that `==` would call equal.
inline bool bitwiseEqualD( double a, double b )
{
    std::uint64_t ba, bb;
    std::memcpy( &ba, &a, sizeof( double ) );
    std::memcpy( &bb, &b, sizeof( double ) );
    return ba == bb;
}

//! STEP 1 of the distributed build, and also the per-level round the icosphere
//! generator uses: assign every distinct key in `keys` a globally agreed gid in
//! `[base, base + globalCount)` and a globally agreed owner (the lowest claiming
//! rank), and verify that every claimant supplied a BITWISE IDENTICAL position for
//! it.
//!
//! `keys` is one entry per LOCAL vertex and may repeat — passing the un-deduped
//! list is deliberate, because it is what lets the coordinator catch a
//! WITHIN-RANK key collision as well as a cross-rank one. `pos` is 3 doubles per
//! local vertex. `gidOut`/`ownerOut` come back one per local vertex.
//!
//! Collective. A key collision throws `std::runtime_error` on EVERY rank (the
//! offending rank is agreed by MPI_Allreduce(MAX) and its key broadcast), because
//! a throw on the coordinator alone would deadlock every other rank in the reply
//! exchange — the same collective-failure idiom buildFaceAdjacency() uses for
//! non-manifold input.
inline void assignVertexKeyGids( MPI_Comm comm, int self_rank, int comm_size,
                                 const std::vector<VertexKey>& keys,
                                 const std::vector<double>& pos, GlobalId base,
                                 std::vector<GlobalId>& gidOut,
                                 std::vector<Rank>& ownerOut,
                                 long long& globalCount, const char* what )
{
    TESSERA_SCOPED_TIMER_DETAILED( ::Tessera::Profiling::TIMER_DBUILD_VKEYS );
    const std::size_t n = keys.size();

    // Advertise every local vertex to its key's coordinator.
    std::vector<std::vector<VKeyClaim>> adv( comm_size );
    for ( std::size_t i = 0; i < n; ++i )
    {
        VKeyClaim c;
        c.key = keys[i];
        for ( int d = 0; d < 3; ++d )
            c.pos[d] = pos[3 * i + d];
        adv[vertexKeyCoordRank( c.key, comm_size )].push_back( c );
    }
    auto got = allToAllV( comm, adv );

    // Coordinate: owner = lowest claiming rank; positions must agree bitwise.
    struct Agg
    {
        Rank owner;
        double pos[3];
        GlobalId gid;
    };
    std::map<VertexKey, Agg> agg;
    VertexKey badKey{};
    bool bad = false;
    for ( int s = 0; s < comm_size; ++s )
    {
        const VKeyClaim* p = got.from( s );
        const int c = got.count( s );
        for ( int i = 0; i < c; ++i )
        {
            auto it = agg.find( p[i].key );
            if ( it == agg.end() )
            {
                Agg a;
                a.owner = static_cast<Rank>( s );
                for ( int d = 0; d < 3; ++d )
                    a.pos[d] = p[i].pos[d];
                a.gid = invalid_gid;
                agg.emplace( p[i].key, a );
            }
            else
            {
                it->second.owner =
                    std::min( it->second.owner, static_cast<Rank>( s ) );
                for ( int d = 0; d < 3; ++d )
                    if ( !bitwiseEqualD( it->second.pos[d], p[i].pos[d] ) )
                    {
                        bad = true;
                        badKey = p[i].key;
                    }
            }
        }
    }

    // Collective failure: agree the offending rank, broadcast its key, all throw.
    {
        int myBad = bad ? self_rank : -1;
        int badRank = -1;
        MPI_Allreduce( &myBad, &badRank, 1, MPI_INT, MPI_MAX, comm );
        if ( badRank >= 0 )
        {
            GlobalId pack[2] = { badKey.id[0], badKey.id[1] };
            MPI_Bcast( pack, 2, MPI_UINT64_T, badRank, comm );
            throw std::runtime_error(
                std::string( "Tessera::" ) + what +
                ": CANONICAL KEY COLLISION. The VertexKey {" +
                std::to_string( pack[0] ) + ", " + std::to_string( pack[1] ) +
                "} was claimed for two vertices at DIFFERENT positions. A "
                "canonical key must be equal on two ranks exactly when they "
                "mean the same vertex; welding two distinct vertices would "
                "produce a mesh that passes every structural invariant and is "
                "geometrically wrong, so this is a hard failure. Fix the "
                "caller's key scheme (or, if the two positions differ only in "
                "their last bits, make the generator compute the shared vertex "
                "identically on both ranks)." );
        }
    }

    // Dense global numbering of this coordinator's keys, offset by `base`.
    // std::map iteration order is deterministic, so the numbering is too.
    {
        long long myKeys = static_cast<long long>( agg.size() );
        long long off = 0;
        MPI_Exscan( &myKeys, &off, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( self_rank == 0 )
            off = 0;
        long long k = 0;
        for ( auto& kv : agg )
            kv.second.gid = base + static_cast<GlobalId>( off + k++ );
        MPI_Allreduce( &myKeys, &globalCount, 1, MPI_LONG_LONG, MPI_SUM, comm );
    }

    // Reply (key, gid, owner) to every claimant, grouped by source rank.
    std::vector<std::vector<VKeyGid>> rep( comm_size );
    for ( int s = 0; s < comm_size; ++s )
    {
        const VKeyClaim* p = got.from( s );
        const int c = got.count( s );
        for ( int i = 0; i < c; ++i )
        {
            const Agg& a = agg.at( p[i].key );
            rep[s].push_back( { p[i].key, a.gid, a.owner } );
        }
    }
    auto res = allToAllV( comm, rep );
    std::map<VertexKey, VKeyGid> answer;
    for ( const auto& m : res.data )
        answer[m.key] = m;

    gidOut.resize( n );
    ownerOut.resize( n );
    for ( std::size_t i = 0; i < n; ++i )
    {
        const VKeyGid& a = answer.at( keys[i] );
        gidOut[i] = a.gid;
        ownerOut[i] = a.owner;
    }
}

} // namespace detail

//! Build a distributed mesh from PER-RANK LOCAL triangle patches, with no rank
//! ever holding the global mesh.
//!
//! The caller supplies its own patch of triangles and, for each of its local
//! vertices, a CANONICAL KEY: a rank-independent 128-bit identifier that is equal
//! on two ranks exactly when they mean the same vertex (see this header's
//! contract note, and `makeVertexKey`). Tessera dedups by key, assigns globally
//! unique contiguous gids, derives the unique edge set, resolves ownership, and
//! builds the ghost layer and halo plans.
//!
//! Patches must COVER the surface with no gaps; OVERLAP is allowed and is
//! resolved by the key dedup, so a caller may generate a patch plus a boundary
//! ring. A duplicated triangle is kept by the lowest rank claiming its FaceKey and
//! dropped by the others, so the caller need not coordinate the seams.
//!
//! `haloDepth` is the number of ghost rings, forwarded to rebuildHalo(): at depth
//! d every owned vertex's d-ring is held locally, so buildVertexStencil() with
//! k <= d has complete rows. Set it once here rather than re-stating it — refine()
//! and migrate() preserve it.
//!
//! Collective. Throws `std::invalid_argument` on a malformed argument and
//! `std::runtime_error` (on every rank) on a canonical-key collision.
//!
//! POSTCONDITIONS. Exactly buildIcosphere() + distribute()'s, so every downstream
//! operation is indifferent to which builder ran: gids are globally unique and
//! agreed, ownership partitions each entity kind, the local layout is canonical
//! (owned first then ghost, each kind ascending by gid), and `halo`'s three plans
//! are valid so haloExchange() is meaningful immediately. Gid NUMBERING differs
//! from the replicated path (a different partition numbers differently); the
//! gid-independent identity — the vertex position multiset and the face
//! corner-position multiset — is bitwise identical.
template <class MeshT, class Scalar>
void buildFromTriangleSoupDistributed(
    MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
    const TriangleSoup<Scalar>& localSoup,
    const std::vector<VertexKey>& localVertexKeys, //!< one per local vertex
    int haloDepth = 1 )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_BUILD_SOUP_DIST );
    static_assert( MeshT::dim == 3, "buildFromTriangleSoupDistributed expects "
                                    "a 3D embedding (Dim == 3)" );

    MPI_Comm comm = mesh.comm();
    const int R = mesh.rank();
    const int size = mesh.commSize();

    const std::size_t nlv = localSoup.numVertices();
    const std::size_t nlf = localSoup.numFaces();

    if ( localVertexKeys.size() != nlv )
        throw std::invalid_argument(
            "Tessera::buildFromTriangleSoupDistributed: one canonical key per "
            "LOCAL vertex is required, got " +
            std::to_string( localVertexKeys.size() ) + " keys for " +
            std::to_string( nlv ) + " local vertices." );
    if ( haloDepth < 1 )
        throw std::invalid_argument(
            "Tessera::buildFromTriangleSoupDistributed: haloDepth must be >= "
            "1, "
            "got " +
            std::to_string( haloDepth ) );
    for ( std::size_t t = 0; t < 3 * nlf; ++t )
        if ( localSoup.triangles[t] < 0 ||
             static_cast<std::size_t>( localSoup.triangles[t] ) >= nlv )
            throw std::invalid_argument(
                "Tessera::buildFromTriangleSoupDistributed: local triangle "
                "index " +
                std::to_string( localSoup.triangles[t] ) +
                " is out of range for " + std::to_string( nlv ) +
                " local vertices." );

    // ---- STEP 1: vertex dedup + gid/owner assignment by canonical key -------
    std::vector<double> vpos( 3 * nlv );
    for ( std::size_t i = 0; i < nlv; ++i )
        for ( int d = 0; d < 3; ++d )
            vpos[3 * i + d] =
                static_cast<double>( localSoup.positions[3 * i + d] );
    std::vector<GlobalId> vGid;
    std::vector<Rank> vOwner;
    long long globalV = 0;
    detail::assignVertexKeyGids( comm, R, size, localVertexKeys, vpos, 0, vGid,
                                 vOwner, globalV,
                                 "buildFromTriangleSoupDistributed" );

    // ---- STEP 2: face dedup by FaceKey, then contiguous face gids -----------
    // A face is owned by the lowest rank claiming its key; the others drop it.
    std::vector<FaceKey> fkey( nlf );
    for ( std::size_t f = 0; f < nlf; ++f )
        fkey[f] = makeFaceKey( vGid[localSoup.triangles[3 * f + 0]],
                               vGid[localSoup.triangles[3 * f + 1]],
                               vGid[localSoup.triangles[3 * f + 2]] );
    std::vector<char> fMine( nlf, 0 );
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DBUILD_FACES );
        std::vector<std::vector<detail::FKeyClaim>> adv( size );
        for ( std::size_t f = 0; f < nlf; ++f )
            adv[detail::faceKeyCoordRank( fkey[f], size )].push_back(
                { fkey[f] } );
        auto got = allToAllV( comm, adv );
        std::map<FaceKey, Rank> owner;
        for ( int s = 0; s < size; ++s )
        {
            const detail::FKeyClaim* p = got.from( s );
            const int c = got.count( s );
            for ( int i = 0; i < c; ++i )
            {
                auto it = owner.find( p[i].key );
                if ( it == owner.end() )
                    owner.emplace( p[i].key, static_cast<Rank>( s ) );
                else
                    it->second = std::min( it->second, static_cast<Rank>( s ) );
            }
        }
        std::vector<std::vector<detail::FKeyOwn>> rep( size );
        for ( int s = 0; s < size; ++s )
        {
            const detail::FKeyClaim* p = got.from( s );
            const int c = got.count( s );
            for ( int i = 0; i < c; ++i )
                rep[s].push_back( { p[i].key, owner.at( p[i].key ) } );
        }
        auto res = allToAllV( comm, rep );
        std::map<FaceKey, Rank> ownerOf;
        for ( const auto& m : res.data )
            ownerOf[m.key] = m.owner;
        // A locally duplicated triangle is kept once (dedup by key), so the
        // owned face set is a partition of the global one either way.
        std::set<FaceKey> takenLocally;
        for ( std::size_t f = 0; f < nlf; ++f )
            if ( ownerOf.at( fkey[f] ) == static_cast<Rank>( R ) &&
                 takenLocally.insert( fkey[f] ).second )
                fMine[f] = 1;
    }

    // Owned faces ascending by FaceKey, then gids from one MPI_Exscan, so a
    // rank's face gids are CONTIGUOUS and dense in [0, globalFaces).
    std::vector<std::size_t> ownedFace; // local soup face indices, key order
    {
        std::vector<std::pair<FaceKey, std::size_t>> ord;
        for ( std::size_t f = 0; f < nlf; ++f )
            if ( fMine[f] )
                ord.emplace_back( fkey[f], f );
        std::sort( ord.begin(), ord.end(),
                   []( const std::pair<FaceKey, std::size_t>& a,
                       const std::pair<FaceKey, std::size_t>& b )
                   { return a.first < b.first; } );
        for ( const auto& kv : ord )
            ownedFace.push_back( kv.second );
    }
    const std::size_t nOwnedF = ownedFace.size();
    std::vector<GlobalId> fGid( nlf, invalid_gid );
    {
        long long mine = static_cast<long long>( nOwnedF );
        long long base = 0;
        MPI_Exscan( &mine, &base, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            base = 0;
        for ( std::size_t i = 0; i < nOwnedF; ++i )
            fGid[ownedFace[i]] =
                static_cast<GlobalId>( base ) + static_cast<GlobalId>( i );
    }

    // ---- STEP 3: derive this rank's OWNED faces' unique edges, LOCALLY ------
    // O(local) memory, which is the whole point: the serial builder's
    // std::map<EdgeKey,int> ran over every edge of the GLOBAL mesh.
    std::map<EdgeKey, int> edgeOf;
    std::vector<EdgeKey> ekey; // local edge index -> key
    std::vector<std::array<GlobalId, 2>>
        eInc; // local edge -> incident face gids
    std::vector<std::array<int, 3>> faceEdge( nOwnedF );
    for ( std::size_t i = 0; i < nOwnedF; ++i )
    {
        const std::size_t f = ownedFace[i];
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId a = vGid[localSoup.triangles[3 * f + k]];
            const GlobalId b = vGid[localSoup.triangles[3 * f + ( k + 1 ) % 3]];
            const EdgeKey key = makeEdgeKey( a, b );
            int e;
            auto it = edgeOf.find( key );
            if ( it == edgeOf.end() )
            {
                e = static_cast<int>( ekey.size() );
                edgeOf.emplace( key, e );
                ekey.push_back( key );
                eInc.push_back( { invalid_gid, invalid_gid } );
            }
            else
                e = it->second;
            faceEdge[i][k] = e;
            if ( eInc[e][0] == invalid_gid )
                eInc[e][0] = fGid[f];
            else if ( eInc[e][1] == invalid_gid )
                eInc[e][1] = fGid[f];
        }
    }
    const int nLocalE = static_cast<int>( ekey.size() );

    // Edge gids + ownership through the edge coordinator — refine()'s Phase 3e
    // verbatim. The coordinator sees each EdgeKey exactly once, so it numbers its
    // own keys densely from an MPI_Exscan and hands the SAME gid to both sides of
    // a patch boundary; owner is the lowest advertising rank, which matches what
    // rebuildHalo()'s round B independently recomputes.
    std::vector<GlobalId> eGid( nLocalE, invalid_gid );
    std::vector<Rank> eOwner( nLocalE, 0 );
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DBUILD_EDGES );
        std::vector<std::vector<detail::DEdgeKeyGid>> adv( size );
        for ( int e = 0; e < nLocalE; ++e )
            adv[detail::edgeCoordRank( ekey[e], size )].push_back(
                { ekey[e], invalid_gid, static_cast<Rank>( R ) } );
        auto got = allToAllV( comm, adv );
        std::map<EdgeKey, detail::DEdgeKeyGid> aggE;
        for ( int s = 0; s < size; ++s )
        {
            const detail::DEdgeKeyGid* p = got.from( s );
            const int c = got.count( s );
            for ( int i = 0; i < c; ++i )
            {
                auto it = aggE.find( p[i].key );
                if ( it == aggE.end() )
                    aggE.emplace( p[i].key, detail::DEdgeKeyGid{
                                                p[i].key, invalid_gid,
                                                static_cast<Rank>( s ) } );
                else
                    it->second.owner =
                        std::min( it->second.owner, static_cast<Rank>( s ) );
            }
        }
        {
            long long myKeys = static_cast<long long>( aggE.size() );
            long long off = 0;
            MPI_Exscan( &myKeys, &off, 1, MPI_LONG_LONG, MPI_SUM, comm );
            if ( R == 0 )
                off = 0;
            long long k = 0;
            for ( auto& kv : aggE )
                kv.second.gid = static_cast<GlobalId>( off + k++ );
        }
        std::vector<std::vector<detail::DEdgeKeyGid>> rep( size );
        for ( int s = 0; s < size; ++s )
        {
            const detail::DEdgeKeyGid* p = got.from( s );
            const int c = got.count( s );
            for ( int i = 0; i < c; ++i )
                rep[s].push_back( aggE.at( p[i].key ) );
        }
        auto res = allToAllV( comm, rep );
        std::map<EdgeKey, detail::DEdgeKeyGid> answer;
        for ( const auto& m : res.data )
            answer[m.key] = m;
        for ( int e = 0; e < nLocalE; ++e )
        {
            const detail::DEdgeKeyGid& a = answer.at( ekey[e] );
            eGid[e] = a.gid;
            eOwner[e] = a.owner;
        }
    }

    // ---- STEP 4: assemble the OWNED entities into the mesh -------------------
    // Only the owned entities are materialized here; rebuildHalo()'s round G
    // recovers anything an owned face references but this rank does not hold, and
    // rounds B/C/D discover ownership, fetch the ghost rings, and canonicalize the
    // layout. Each kind is written owned-first, ascending by gid, which is the
    // layout round D will reproduce anyway.
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DBUILD_ASSEMBLE );

        // Vertices: every local vertex, deduped by gid, owned first.
        std::vector<GlobalId> vord;
        std::map<GlobalId, std::size_t> vLocalOf; // gid -> soup vertex index
        {
            std::vector<GlobalId> owned, ghost;
            for ( std::size_t i = 0; i < nlv; ++i )
                if ( vLocalOf.emplace( vGid[i], i ).second )
                    ( vOwner[i] == static_cast<Rank>( R ) ? owned : ghost )
                        .push_back( vGid[i] );
            std::sort( owned.begin(), owned.end() );
            std::sort( ghost.begin(), ghost.end() );
            vord = owned;
            vord.insert( vord.end(), ghost.begin(), ghost.end() );
            const std::size_t nOwnedV = owned.size();

            Cabana::AoSoA<typename MeshT::vertex_member_types,
                          Kokkos::HostSpace>
                hv( "dbuild_hv", vord.size() );
            auto gid = Cabana::slice<VertexField::Gid>( hv );
            auto own = Cabana::slice<VertexField::Owner>( hv );
            auto flg = Cabana::slice<VertexField::Flags>( hv );
            auto pos = Cabana::slice<VertexField::Position>( hv );
            for ( std::size_t li = 0; li < vord.size(); ++li )
            {
                const std::size_t i = vLocalOf.at( vord[li] );
                gid( li ) = vord[li];
                own( li ) = vOwner[i];
                flg( li ) = 0;
                for ( int d = 0; d < 3; ++d )
                    pos( li, d ) = localSoup.positions[3 * i + d];
            }
            mesh.resizeVertices( vord.size() );
            Cabana::deep_copy( mesh.vertices(), hv );
            mesh.setOwnedCounts( nOwnedV, 0, 0 );
        }

        // Edges: this rank's locally-derived edges, owned first.
        std::size_t nOwnedE = 0;
        {
            std::vector<int> ord;
            for ( int e = 0; e < nLocalE; ++e )
                if ( eOwner[e] == static_cast<Rank>( R ) )
                    ord.push_back( e );
            nOwnedE = ord.size();
            for ( int e = 0; e < nLocalE; ++e )
                if ( eOwner[e] != static_cast<Rank>( R ) )
                    ord.push_back( e );
            auto byGid = [&]( int a, int b ) { return eGid[a] < eGid[b]; };
            const auto mid =
                ord.begin() + static_cast<std::ptrdiff_t>( nOwnedE );
            std::sort( ord.begin(), mid, byGid );
            std::sort( mid, ord.end(), byGid );

            Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace>
                he( "dbuild_he", ord.size() );
            auto gid = Cabana::slice<EdgeField::Gid>( he );
            auto own = Cabana::slice<EdgeField::Owner>( he );
            auto lev = Cabana::slice<EdgeField::Level>( he );
            auto verts = Cabana::slice<EdgeField::Verts>( he );
            auto faces = Cabana::slice<EdgeField::Faces>( he );
            for ( std::size_t li = 0; li < ord.size(); ++li )
            {
                const int e = ord[li];
                gid( li ) = eGid[e];
                own( li ) = eOwner[e];
                lev( li ) = 0;
                verts( li, 0 ) = ekey[e].id[0];
                verts( li, 1 ) = ekey[e].id[1];
                // EdgeField::Faces is filled from this rank's OWN incidences only —
                // the same partial-by-construction contract migrate() leaves, and the
                // reason buildFaceAdjacency() exists rather than reading this field.
                faces( li, 0 ) = eInc[e][0];
                faces( li, 1 ) = eInc[e][1];
            }
            mesh.resizeEdges( ord.size() );
            Cabana::deep_copy( mesh.edges(), he );
        }

        // Faces: the owned faces only, ascending by gid (== ownedFace order, since
        // the gids were handed out in that order).
        {
            Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace>
                hf( "dbuild_hf", nOwnedF );
            auto gid = Cabana::slice<FaceField::Gid>( hf );
            auto own = Cabana::slice<FaceField::Owner>( hf );
            auto lev = Cabana::slice<FaceField::Level>( hf );
            auto verts = Cabana::slice<FaceField::Verts>( hf );
            auto edges = Cabana::slice<FaceField::Edges>( hf );
            for ( std::size_t i = 0; i < nOwnedF; ++i )
            {
                const std::size_t f = ownedFace[i];
                gid( i ) = fGid[f];
                own( i ) = static_cast<Rank>( R );
                lev( i ) = 0;
                for ( int k = 0; k < 3; ++k )
                {
                    verts( i, k ) = vGid[localSoup.triangles[3 * f + k]];
                    edges( i, k ) = eGid[faceEdge[i][k]];
                }
            }
            // A freshly built mesh is entirely RED; in Conforming mode the closure
            // members must say so explicitly (a zero ClosureParent reads as face
            // gid 0). No-op in HangingNode2to1 mode.
            initClosureFaceMembers<MeshT>( hf, 0, nOwnedF );
            mesh.resizeFaces( nOwnedF );
            Cabana::deep_copy( mesh.faces(), hf );
        }
        mesh.setOwnedCounts( mesh.numOwnedVertices(), nOwnedE, nOwnedF );
    }

    // ---- STEP 5: ghost layer, key tables, CSRs and halo plans ----------------
    // rebuildHalo() owns all of it, and it is the general NON-REPLICATED builder:
    // its round B resolves ownership by communication, which is the structural
    // reason this path needs no replicated mesh at all. It also rebuilds the
    // edge/face key Views and the vertex 1-ring CSRs from scratch, so steps 1-4
    // deliberately do not.
    rebuildHalo( mesh, halo, haloDepth );
}

namespace detail
{

//! One face of the icosphere's subdivision tree during the descent: its three
//! corners' canonical ids, in winding order.
struct IcoTri
{
    GlobalId c[3];
};

} // namespace detail

//! Generate and build an icosphere of the given subdivision level with no rank
//! materializing the global mesh. Collective.
//!
//! PARTITION BY THE SUBDIVISION TREE, NOT BY AN AXIS SORT. `facePartitionByAxis()`
//! sorts every face centroid globally, which requires every centroid, which
//! requires the global mesh — the exact thing being avoided. The icosphere is
//! generated by a deterministic 1->4 recursion, so face index space at depth `s`
//! is a perfect 20-ary-then-4-ary tree: the children of face `p` are
//! `4p .. 4p+3`, and face `f` at depth `s` descends from base face `f / 4^s`.
//! Rank `r` is assigned the contiguous index range `[r*F/P, (r+1)*F/P)` and
//! generates ONLY those faces, by descending only the subtrees that intersect the
//! range. Memory and time are O(F/P + s) per rank, with no communication and no
//! global sort. The result is also a hierarchical, locality-preserving partition
//! (the children of a base face stay together), which is a BETTER starting
//! partition than a single-axis sort. For `size > 20` the range simply cuts inside
//! a base patch, which is fine, and for `size > F` some ranks own nothing, which
//! is also fine.
//!
//! CANONICAL KEYS COME FREE FROM THE RECURSION, one coordinator round per
//! subdivision level (`s` extra collectives at setup, which is nothing). A base
//! vertex is `{ i, invalid_gid }`; a midpoint is `makeVertexKey( a, b )` over the
//! already-assigned canonical ids of its two parents. Keys are therefore assigned
//! level by level, each level's round numbering that level's new midpoints densely
//! above the running total. They are globally distinct across the whole vertex set
//! because two vertices adjacent at level `d` are separated by their midpoint at
//! level `d+1` and are never adjacent again, so a pair of ids is an edge of at
//! most one level's mesh.
//!
//! REPRODUCIBILITY. The generated positions are BITWISE IDENTICAL to
//! `generateIcosphere()`'s for the same subdivision, so the two paths are
//! interchangeable: the same base table, the same `0.5*(v_a + v_b)` then
//! `detail::normalize3` order of operations, and the same `double` intermediate
//! precision regardless of `Scalar`. Gid NUMBERING of course differs (a different
//! partition numbers differently); the position multiset does not.
//!
//! Throws `std::invalid_argument` on a negative `subdivisions` or `haloDepth < 1`.
template <class MeshT>
void buildIcosphereDistributed( MeshT& mesh,
                                MeshHalo<typename MeshT::memory_space>& halo,
                                int subdivisions, int haloDepth = 1 )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_BUILD_ICOSPHERE_DIST );
    static_assert( MeshT::dim == 3,
                   "buildIcosphereDistributed requires Dim == 3" );
    using Scalar = typename MeshT::scalar_type;
    using P3 = std::array<Scalar, 3>;

    if ( subdivisions < 0 )
        throw std::invalid_argument(
            "Tessera::buildIcosphereDistributed: subdivisions must be >= 0, "
            "got " +
            std::to_string( subdivisions ) );
    if ( haloDepth < 1 )
        throw std::invalid_argument(
            "Tessera::buildIcosphereDistributed: haloDepth must be >= 1, got " +
            std::to_string( haloDepth ) );

    MPI_Comm comm = mesh.comm();
    const int R = mesh.rank();
    const int size = mesh.commSize();

    // -- the base icosahedron: same table, same normalize3, as
    //    generateIcosphere(), so the base positions are bitwise identical -----
    const double t = ( 1.0 + std::sqrt( 5.0 ) ) / 2.0;
    const double base[12][3] = { { -1, t, 0 },  { 1, t, 0 },   { -1, -t, 0 },
                                 { 1, -t, 0 },  { 0, -1, t },  { 0, 1, t },
                                 { 0, -1, -t }, { 0, 1, -t },  { t, 0, -1 },
                                 { t, 0, 1 },   { -t, 0, -1 }, { -t, 0, 1 } };
    const int baseTris[60] = {
        0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
        4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
        6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1 };

    // Canonical id -> position / key, for the ids THIS RANK touches only. That
    // "only" is the memory guarantee: the maps are O(F/P), never O(F).
    std::map<GlobalId, P3> posOf;
    std::map<GlobalId, VertexKey> keyOf;
    for ( int i = 0; i < 12; ++i )
    {
        Scalar x = static_cast<Scalar>( base[i][0] );
        Scalar y = static_cast<Scalar>( base[i][1] );
        Scalar z = static_cast<Scalar>( base[i][2] );
        detail::normalize3( x, y, z );
        posOf[static_cast<GlobalId>( i )] = P3{ x, y, z };
        keyOf[static_cast<GlobalId>( i )] =
            makeVertexKey( static_cast<GlobalId>( i ) );
    }

    // -- this rank's contiguous face range at the FINAL depth ----------------
    const long long F = 20LL << ( 2 * subdivisions ); // 20 * 4^s
    const long long lo = static_cast<long long>( R ) * F / size;
    const long long hi = static_cast<long long>( R + 1 ) * F / size;
    const bool empty = ( lo >= hi );

    // Level-d ancestors of [lo, hi) are the CONTIGUOUS level-d index range
    // [lo / 4^(s-d), ceil(hi / 4^(s-d))) — which is what makes the descent a
    // simple per-level range rather than a recursive subtree walk.
    auto levelRange = [&]( int d, long long& l, long long& h )
    {
        if ( empty )
        {
            l = h = 0;
            return;
        }
        const long long blk = 1LL << ( 2 * ( subdivisions - d ) );
        l = lo / blk;
        h = ( hi + blk - 1 ) / blk;
    };

    std::vector<detail::IcoTri> cur;
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DBUILD_GENERATE );
        long long l0 = 0, h0 = 0;
        levelRange( 0, l0, h0 );
        for ( long long p = l0; p < h0; ++p )
        {
            detail::IcoTri tri;
            for ( int k = 0; k < 3; ++k )
                tri.c[k] = static_cast<GlobalId>( baseTris[3 * p + k] );
            cur.push_back( tri );
        }
    }

    long long nVertsSoFar = 12; // global vertex count through the current level
    for ( int d = 1; d <= subdivisions; ++d )
    {
        // 1. The distinct new midpoint keys of this rank's level-(d-1) faces,
        //    with their positions computed from the parents this rank holds.
        std::vector<VertexKey> newKeys;
        std::vector<double> newPos;
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_DBUILD_GENERATE );
            std::map<VertexKey, P3> pending;
            for ( const auto& tri : cur )
                for ( int k = 0; k < 3; ++k )
                {
                    const GlobalId a = tri.c[k];
                    const GlobalId b = tri.c[( k + 1 ) % 3];
                    const VertexKey key = makeVertexKey( a, b );
                    if ( pending.find( key ) != pending.end() )
                        continue;
                    const P3& pa = posOf.at( a );
                    const P3& pb = posOf.at( b );
                    // EXACTLY generateIcosphere()'s arithmetic: the sum is taken
                    // in Scalar, scaled by 0.5 in double, cast back, then
                    // normalized by the shared helper. Do not restructure it —
                    // bitwise agreement with the replicated path depends on it.
                    Scalar mx = static_cast<Scalar>( 0.5 * ( pa[0] + pb[0] ) );
                    Scalar my = static_cast<Scalar>( 0.5 * ( pa[1] + pb[1] ) );
                    Scalar mz = static_cast<Scalar>( 0.5 * ( pa[2] + pb[2] ) );
                    detail::normalize3( mx, my, mz );
                    pending.emplace( key, P3{ mx, my, mz } );
                }
            for ( const auto& kv : pending )
            {
                newKeys.push_back( kv.first );
                for ( int c = 0; c < 3; ++c )
                    newPos.push_back( static_cast<double>( kv.second[c] ) );
            }
        }

        // 2. One coordinator round: dense globally-agreed ids above the running
        //    total. The position round-trip doubles as a check that every rank
        //    computed the shared midpoint bit-identically.
        std::vector<GlobalId> newGid;
        std::vector<Rank> newOwner; // unused: ownership is resolved in step 1/5
        long long added = 0;
        detail::assignVertexKeyGids( comm, R, size, newKeys, newPos,
                                     static_cast<GlobalId>( nVertsSoFar ),
                                     newGid, newOwner, added,
                                     "buildIcosphereDistributed" );
        nVertsSoFar += added;

        // 3. Record the new vertices, then emit the level-d children that fall
        //    inside this rank's range. Child order matches generateIcosphere()'s
        //    { a,ab,ca }, { b,bc,ab }, { c,ca,bc }, { ab,bc,ca }, so child `c` of
        //    parent `p` is face 4p+c — the identity the range arithmetic uses.
        std::map<VertexKey, GlobalId> idOfKey;
        for ( std::size_t i = 0; i < newKeys.size(); ++i )
        {
            idOfKey[newKeys[i]] = newGid[i];
            posOf[newGid[i]] = P3{ static_cast<Scalar>( newPos[3 * i + 0] ),
                                   static_cast<Scalar>( newPos[3 * i + 1] ),
                                   static_cast<Scalar>( newPos[3 * i + 2] ) };
            keyOf[newGid[i]] = newKeys[i];
        }

        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_DBUILD_GENERATE );
        long long lPrev = 0, hPrev = 0, lNow = 0, hNow = 0;
        levelRange( d - 1, lPrev, hPrev );
        levelRange( d, lNow, hNow );
        std::vector<detail::IcoTri> next;
        for ( std::size_t i = 0; i < cur.size(); ++i )
        {
            const long long p = lPrev + static_cast<long long>( i );
            const GlobalId a = cur[i].c[0];
            const GlobalId b = cur[i].c[1];
            const GlobalId c = cur[i].c[2];
            const GlobalId ab = idOfKey.at( makeVertexKey( a, b ) );
            const GlobalId bc = idOfKey.at( makeVertexKey( b, c ) );
            const GlobalId ca = idOfKey.at( makeVertexKey( c, a ) );
            const GlobalId child[4][3] = {
                { a, ab, ca }, { b, bc, ab }, { c, ca, bc }, { ab, bc, ca } };
            for ( int k = 0; k < 4; ++k )
            {
                const long long idx = 4 * p + k;
                if ( idx < lNow || idx >= hNow )
                    continue;
                detail::IcoTri tri;
                for ( int j = 0; j < 3; ++j )
                    tri.c[j] = child[k][j];
                next.push_back( tri );
            }
        }
        cur.swap( next );
    }

    // -- hand the local patch to the general builder --------------------------
    // Every corner already carries a globally agreed canonical id AND its
    // canonical key, so the local soup is immediate. Deliverable A re-derives the
    // gids from the keys rather than reusing the canonical ids: there is exactly
    // one distributed-build code path, and this generator is its first caller
    // rather than a shortcut around it.
    TriangleSoup<Scalar> soup;
    std::vector<VertexKey> keys;
    {
        std::map<GlobalId, int> localOf;
        soup.triangles.reserve( 3 * cur.size() );
        for ( const auto& tri : cur )
            for ( int k = 0; k < 3; ++k )
            {
                auto it = localOf.find( tri.c[k] );
                int li;
                if ( it == localOf.end() )
                {
                    li = static_cast<int>( keys.size() );
                    localOf.emplace( tri.c[k], li );
                    keys.push_back( keyOf.at( tri.c[k] ) );
                    const P3& p = posOf.at( tri.c[k] );
                    for ( int c = 0; c < 3; ++c )
                        soup.positions.push_back( p[c] );
                }
                else
                    li = it->second;
                soup.triangles.push_back( li );
            }
    }

    buildFromTriangleSoupDistributed( mesh, halo, soup, keys, haloDepth );
}

} // namespace Tessera

#endif // TESSERA_DISTRIBUTED_BUILDER_HPP
