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
#include "Tessera_HaloRebuild.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_RefinementMode.hpp"
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
// Mesh migration (Step 7)
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
// Rounds. Only S and A are migration proper; G and B/C/D are the general halo
// rebuild, which lives in Tessera_HaloRebuild.hpp because refine() needs it too
// (see that header for what each does). The entire interface between the two
// halves is the three gid-keyed maps faceById / vById / eById that round A fills.
//   S  (RefinementMode::Conforming only) Sibling-cohesion fixup on `dest`. See
//      the block comment at the fixup in migrate(); purely local, no comm.
//   G  detail::gatherReferencedTuples() — recover any vertex/edge an owned face
//      references but this rank does not hold, so round A can move a face with
//      its full vertex/edge pack.
//   A  Move each owned face + its 3 vertices + 3 edges to dest (three allToAllV).
//      The receiver's owned faces are the faces it received; its candidate
//      vertices/edges are their (deduped) endpoints.
//   B/C/D  detail::finishHaloAndAssemble() — ownership, ghost fetch, and the
//      owned-first assembly with the three halo plans. The mesh is left
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

// detail::TupleBlob / gidCoordRank / VtxInc / GidOwn / EdgeInc live in
// Tessera_HaloRebuild.hpp, alongside the rounds that consume them.

//! What migrate() had to do beyond applying `dest` verbatim. Returned rather
//! than logged so a caller/test can assert on it; ignoring it is fine, so every
//! pre-existing `migrate( mesh, halo, dest )` call site is unaffected.
struct MigrateStats
{
    //! Owned faces whose `dest` entry the sibling-cohesion fixup overrode
    //! (RefinementMode::Conforming only; always 0 in HangingNode2to1 mode).
    long long siblingFixups = 0;
    //! Closure-sibling groups seen locally, i.e. distinct retired parent gids.
    long long siblingGroups = 0;
};

//! Rewrite `dest` (indexed by owned face local index) so that every closure
//! sibling group ends up on ONE rank, when the group may already be SPLIT
//! across ranks on entry. Returns the number of entries overridden, summed
//! globally; a no-op returning 0 in HangingNode2to1 mode.
//!
//! This is the collective counterpart of migrate()'s round S, and exists for
//! exactly one caller shape: something that hands a conforming mesh a partition
//! it did not derive from the mesh's own layout. readMesh() is that caller —
//! its fresh dense-index block partition cuts wherever the block boundaries
//! fall, so it can (and routinely does) land two children of one retired parent
//! on different ranks. Round S cannot see that: it is deliberately local, and a
//! rank holding one child of a split group has no way to know the group extends
//! elsewhere.
//!
//! One allToAllV round trip, keyed on the parent gid (parent % size), matching
//! the gid-coordinator idiom the rest of the library uses. The winner is the
//! destination of the LOWEST-GID sibling — the same globally-agreed rule round S
//! applies — so running this and then round S is idempotent, and the choice
//! introduces no partition dependence.
template <class MeshT>
long long repairClosureCohesion( const MeshT& mesh, std::vector<Rank>& dest )
{
    if constexpr ( MeshT::refinement_mode != RefinementMode::Conforming )
    {
        (void)mesh;
        (void)dest;
        return 0;
    }
    else
    {
        MPI_Comm comm = mesh.comm();
        const int size = mesh.commSize();
        const int nof = static_cast<int>( mesh.numOwnedFaces() );

        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf_cohesion", mesh.numFaces() );
        Cabana::deep_copy( hf, mesh.faces() );
        auto f_gid = Cabana::slice<FaceField::Gid>( hf );
        auto f_parent = Cabana::slice<MeshT::closure_parent_field>( hf );

        // (parent gid, child gid, that child's proposed destination).
        struct ChildMsg
        {
            GlobalId parent;
            GlobalId child;
            Rank dest;
        };
        std::vector<std::vector<ChildMsg>> toCoord( size );
        // Local face index of each message, parallel to toCoord, so the reply
        // (grouped by source rank, in send order) maps straight back.
        std::vector<std::vector<int>> sentLocal( size );
        for ( int f = 0; f < nof; ++f )
        {
            const GlobalId p = f_parent( f );
            if ( p == invalid_gid )
                continue;
            const int c = static_cast<int>( p % static_cast<GlobalId>( size ) );
            toCoord[c].push_back( ChildMsg{ p, f_gid( f ), dest[f] } );
            sentLocal[c].push_back( f );
        }
        auto got = allToAllV( comm, toCoord );

        // Coordinator: per parent, the destination of the lowest-gid child.
        std::map<GlobalId, std::pair<GlobalId, Rank>> leader;
        for ( const auto& m : got.data )
        {
            auto it = leader.find( m.parent );
            if ( it == leader.end() )
                leader.emplace( m.parent, std::make_pair( m.child, m.dest ) );
            else if ( m.child < it->second.first )
                it->second = { m.child, m.dest };
        }

        // Reply to every advertisement in received order, per source rank.
        std::vector<std::vector<Rank>> reply( size );
        for ( int s = 0; s < size; ++s )
        {
            const ChildMsg* p = got.from( s );
            const int c = got.count( s );
            for ( int k = 0; k < c; ++k )
                reply[s].push_back( leader.at( p[k].parent ).second );
        }
        auto back = allToAllV( comm, reply );

        long long fixups = 0;
        for ( int s = 0; s < size; ++s )
        {
            const Rank* p = back.from( s );
            const int c = back.count( s );
            for ( int k = 0; k < c; ++k )
            {
                const int f = sentLocal[s][k];
                if ( dest[f] != p[k] )
                {
                    dest[f] = p[k];
                    ++fixups;
                }
            }
        }

        long long global = 0;
        MPI_Allreduce( &fixups, &global, 1, MPI_LONG_LONG, MPI_SUM, comm );
        return global;
    }
}

//! Migrate owned faces to `dest` (indexed by owned face local index), moving their
//! vertices/edges + whole field pack, recomputing lowest-rank ownership, and
//! rebuilding the 1-deep halo. See the header comment for the algorithm.
template <class MeshT>
MigrateStats migrate( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                      const std::vector<Rank>& dest )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_MIGRATE );
    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;
    using VTuple = detail::HostVertexTuple<MeshT>;
    using ETuple = detail::HostEdgeTuple<MeshT>;
    using FTuple = detail::HostFaceTuple<MeshT>;

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

    // ======================================================================
    // Round S — sibling-cohesion fixup on `dest` (Conforming mode only).
    // ======================================================================
    // A closure child names its retired red parent outright, so un-closing is
    // local PER CHILD and siblings need not be co-resident for a child to know
    // its parent. What they must not do is get SPLIT across ranks: unclose()
    // would then restore the same red parent on two ranks, duplicating a face
    // and breaking the ownership partition. Co-residency is therefore an
    // invariant of the conforming mesh, and migrate() is the only thing that
    // can break it.
    //
    // It is REPAIRED, not rejected. A violating `dest` is the normal case, not
    // an error: computeLoadBalance() partitions by face CENTROID and closure
    // siblings have different centroids, so Zoltan2 routinely scatters them.
    // Rejecting would make loadBalance() unusable on a conforming mesh, and
    // pushing the repair onto every caller would duplicate this loop at each
    // one. The repair is well-defined and cheap: all children of a parent
    // follow the LOWEST-GID sibling's destination -- a choice that reads only
    // globally-agreed gids, so it does not reintroduce partition dependence.
    // The number of entries overridden is returned in MigrateStats so a caller
    // that cares (or a test) can see it; a large count means the partitioner's
    // cost model and the closure disagree, which is what the per-parent weight
    // in ownedFaceWeights() exists to reduce.
    //
    // Purely local: siblings are co-resident on entry (the invariant), so every
    // group is visible on the one rank that holds it. `destUse` aliases `dest`
    // untouched in HangingNode2to1 mode -- no copy, no scan.
    MigrateStats stats;
    std::vector<Rank> destFixed;
    if constexpr ( MeshT::refinement_mode == RefinementMode::Conforming )
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_MIGRATE_SIBLING );
        auto f_gid = Cabana::slice<FaceField::Gid>( hf );
        auto f_parent = Cabana::slice<MeshT::closure_parent_field>( hf );

        // parent gid -> (lowest child gid seen, that child's destination).
        std::map<GlobalId, std::pair<GlobalId, Rank>> leader;
        for ( int f = 0; f < nof; ++f )
        {
            const GlobalId p = f_parent( f );
            if ( p == invalid_gid )
                continue;
            auto it = leader.find( p );
            if ( it == leader.end() )
                leader.emplace( p, std::make_pair( f_gid( f ), dest[f] ) );
            else if ( f_gid( f ) < it->second.first )
                it->second = { f_gid( f ), dest[f] };
        }
        stats.siblingGroups = static_cast<long long>( leader.size() );

        if ( !leader.empty() )
        {
            destFixed = dest;
            for ( int f = 0; f < nof; ++f )
            {
                const GlobalId p = f_parent( f );
                if ( p == invalid_gid )
                    continue;
                const Rank d = leader.at( p ).second;
                if ( destFixed[f] != d )
                {
                    destFixed[f] = d;
                    ++stats.siblingFixups;
                }
            }
        }
    }
    const std::vector<Rank>& destUse = destFixed.empty() ? dest : destFixed;

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

    // Round G. Recover any vertex/edge an owned face references but this rank does
    // not hold, so round A below can move a face with its full vertex/edge pack.
    {
        std::vector<GlobalId> refV, refE;
        refV.reserve( static_cast<std::size_t>( nof ) * 3 );
        refE.reserve( static_cast<std::size_t>( nof ) * 3 );
        for ( int f = 0; f < nof; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                refV.push_back( f_verts( f, k ) );
                refE.push_back( f_edges( f, k ) );
            }
        detail::gatherReferencedTuples( comm, R, size, refV, refE, heldV,
                                        heldE );
    }

    // ======================================================================
    // Round A — move owned faces + their vertices/edges to destinations.
    // ======================================================================
    // Owned faces (unique) and candidate vertices/edges (deduped by gid).
    std::map<GlobalId, FTuple> faceById;     // this rank's new owned faces
    std::map<GlobalId, VTuple> vById;        // vertices referenced locally
    std::map<GlobalId, ETuple> eById;        // edges referenced locally
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_MIGRATE_MOVE );
        std::vector<std::vector<detail::TupleBlob<FTuple>>> sendF( size );
        std::vector<std::vector<detail::TupleBlob<VTuple>>> sendV( size );
        std::vector<std::vector<detail::TupleBlob<ETuple>>> sendE( size );
        for ( int f = 0; f < nof; ++f )
        {
            const Rank d = destUse[f];
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
    // Rounds B, C, D — ownership, ghost fetch, owned-first assembly + plans.
    // ======================================================================
    // Shared verbatim with rebuildHalo(); the three maps round A just filled are
    // the entire interface. INVALIDATION: this reallocates the AoSoAs, key Views
    // and CSRs and replaces the halo plans (see finishHaloAndAssemble()).
    detail::finishHaloAndAssemble( mesh, halo, faceById, vById, eById );

    return stats;
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

//! Per-owned-face work weight for a partitioner, one entry per OWNED face in
//! local index order (the contract every caller already relies on -- unchanged).
//!
//! A red leaf face is one unit of work. In RefinementMode::Conforming the
//! VISIBLE faces a partitioner sees include the transient closure children, and
//! a red parent's 2-4 children are NOT 2-4 units of work: they are one red face
//! retriangulated, they carry that parent's level and user fields, and
//! migrate()'s sibling-cohesion fixup will move them as a block regardless of
//! what the partitioner decides individually. Weighting them 1.0 each would let
//! the closure -- an O(level-jump-boundary) set that changes shape on every
//! refine -- masquerade as real load and pull parts toward refinement fronts.
//!
//! So each closure child gets 1/(number of siblings) and a passed-through red
//! face gets 1.0. The total weight is then exactly the red-face count, and the
//! weight of a sibling group is 1.0 however the group is split -- which is what
//! makes the post-fixup partition's load equal the load Zoltan2 optimized.
template <class MeshT>
std::vector<double> ownedFaceWeights( const MeshT& mesh )
{
    const std::size_t nof = mesh.numOwnedFaces();
    std::vector<double> w( nof, 1.0 );

    if constexpr ( MeshT::refinement_mode == RefinementMode::Conforming )
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf", mesh.numFaces() );
        Cabana::deep_copy( hf, mesh.faces() );
        auto f_parent = Cabana::slice<MeshT::closure_parent_field>( hf );

        std::map<GlobalId, int> siblings;
        for ( std::size_t f = 0; f < nof; ++f )
            if ( f_parent( f ) != invalid_gid )
                ++siblings[f_parent( f )];
        for ( std::size_t f = 0; f < nof; ++f )
            if ( f_parent( f ) != invalid_gid )
                w[f] =
                    1.0 / static_cast<double>( siblings.at( f_parent( f ) ) );
    }
    return w;
}

} // namespace Tessera

#endif // TESSERA_MESH_MIGRATE_HPP
