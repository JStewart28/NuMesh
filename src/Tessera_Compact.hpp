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

#ifndef TESSERA_COMPACT_HPP
#define TESSERA_COMPACT_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_EditFamily.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_HaloExchange.hpp"
#include "Tessera_HaloRebuild.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Tessera
{

// ============================================================================
// Mesh compaction — tombstone removal, and periodic gid renumbering
// ============================================================================
//
// Tessera's editors only ever ADD entities. The moment a coarsening operation
// exists (collapseEdges()) it produces dead entities, and every derived
// structure of the mesh -- the owned-first ordering, the two vertex CSRs, the
// edgeKeys/faceKeys side tables, the three halo plans -- has to be rebuilt
// around the survivors. compact() is that rebuild, and it exists BEFORE
// collapse so collapse does not grow its own private half-compaction.
//
// THE TOMBSTONE CONVENTION IS UNIFORM ACROSS THE THREE ENTITY KINDS:
// `Gid == invalid_gid` marks a dead entity. VertexField::Flags exists and the
// edge/face packs have no equivalent, so using Flags for vertices only would
// mean two mechanisms for one idea.
//
// COMPACTION IS A HALO REBUILD, NOT A PRIVATE PERMUTATION. rebuildHalo()'s
// round D already does everything the removal needs: it orders owned-first and
// gid-ascending, whole-tuple copies every AoSoA (so user fields travel with no
// per-field plumbing), rebuilds both CSRs and both key tables, and REPLACES the
// three halo plans. Critically, it derives the held vertex and edge set purely
// from the OWNED FACES (Tessera_HaloRebuild.hpp, the vById/eById fill), so a
// vertex or edge that no surviving face references is dropped with no explicit
// compaction of those two AoSoAs at all. compact() therefore has only three
// steps of its own: verify the tombstone set closes, drop the dead faces, and
// hand the survivors to rebuildHalo(). Gid preservation, the canonical
// ordering, and the generation bump all fall out unchanged.
//
// GIDS ARE PRESERVED BY DEFAULT, AND RENUMBERING IS A SEPARATE, PERIODIC CALL.
// Preserving gids makes compact() a purely local reordering plus one halo
// rebuild: no rank has to be told that anything was renamed. Renumbering is a
// global relabelling that invalidates every gid a peer holds. A remesher
// calling compact() every step wants the cheap one; the gid-space growth below
// is a slow leak that a call every few hundred steps fixes. Conflating them
// would make the common case pay for the rare one.
//
// THE GID-SPACE LEAK, AND WHY IT IS NOT COSMETIC. Gids are assigned by
// MPI_Exscan onto a monotonically RISING global count, so a long run of
// split-and-collapse rounds grows the gid SPACE without bound even while the
// mesh stays the same size. Several code paths index by gid into a DENSE host
// array sized to a max gid, and every one of them grows with the space rather
// than with the mesh:
//
//   * detail::buildKindPlan() takes `gid2local` and indexes it by raw gid
//     (Tessera_Distribute.hpp);
//   * rebuildHalo()'s round D BUILDS those vectors, sized to the local max gid
//     (detail::make_g2l, Tessera_HaloRebuild.hpp) -- so compact() is itself on
//     the leak path, not merely a fixer of it;
//   * distribute()'s `build_order` builds v2l/e2l/f2l the same way
//     (Tessera_Distribute.hpp).
//
// migrate() avoids it (std::map<GlobalId,...>), so the hazard is confined, but
// it is real. compactAndRenumberGids() is the sanctioned mitigation; removing
// the dense-by-gid indexing is separate, larger work.

//! What one compaction did. EVERY FIELD IS GLOBAL and identical on every rank:
//! the removal counts are owned-count differences summed across ranks, and the
//! gid space is a global maximum. A local count would not be a statement about
//! the mesh -- a rank's LOCAL count also moves when the ghost set changes.
struct CompactStats
{
    long long verticesRemoved = 0; //!< global owned vertices dropped
    long long edgesRemoved = 0;    //!< global owned edges dropped
    long long facesRemoved = 0;    //!< global owned faces dropped

    //! Size of the gid SPACE, summed over the three entity kinds, as
    //! (global max live gid + 1) per kind. Compare against the live entity
    //! count: the gap between them is the leak described above. Always
    //! reported, by both calls -- for compact() the two are equal (that
    //! equality IS the statement that gids were preserved) unless the removal
    //! happened to include the globally maximal gid of some kind, in which case
    //! the space shrinks by exactly the vacated tail.
    long long gidSpaceBefore = 0;
    long long gidSpaceAfter = 0;
};

namespace detail
{

//! Sum over the three kinds of (global max OWNED gid + 1), in one MPI_Allreduce.
//! Owned entities partition the mesh, so the owned maximum is the global one; a
//! kind with no live entity anywhere contributes 0. Dead entities are excluded:
//! their gid reads invalid_gid, which is the largest representable value and
//! would swamp the maximum.
template <class MeshT>
long long gidSpace( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto eg = Cabana::slice<EdgeField::Gid>( he );
    auto fg = Cabana::slice<FaceField::Gid>( hf );

    long long loc[3] = { -1, -1, -1 };
    for ( std::size_t i = 0; i < mesh.numOwnedVertices(); ++i )
        if ( vg( i ) != invalid_gid )
            loc[0] = std::max( loc[0], static_cast<long long>( vg( i ) ) );
    for ( std::size_t i = 0; i < mesh.numOwnedEdges(); ++i )
        if ( eg( i ) != invalid_gid )
            loc[1] = std::max( loc[1], static_cast<long long>( eg( i ) ) );
    for ( std::size_t i = 0; i < mesh.numOwnedFaces(); ++i )
        if ( fg( i ) != invalid_gid )
            loc[2] = std::max( loc[2], static_cast<long long>( fg( i ) ) );

    long long glob[3] = { -1, -1, -1 };
    MPI_Allreduce( loc, glob, 3, MPI_LONG_LONG, MPI_MAX, mesh.comm() );
    return ( glob[0] + 1 ) + ( glob[1] + 1 ) + ( glob[2] + 1 );
}

//! Global owned counts of the three kinds, in one MPI_Allreduce.
template <class MeshT>
void ownedCounts( MeshT& mesh, long long out[3] )
{
    long long loc[3] = { static_cast<long long>( mesh.numOwnedVertices() ),
                         static_cast<long long>( mesh.numOwnedEdges() ),
                         static_cast<long long>( mesh.numOwnedFaces() ) };
    MPI_Allreduce( loc, out, 3, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
}

//! STEP 1 — verify the tombstone set CLOSES, before anything is mutated.
//!
//! A gid is dead iff the rank that OWNS it marked it dead: a tombstone on a
//! GHOST copy says nothing about the entity (and is silently repaired, since
//! round G re-fetches any reference an owned face has lost). So the check is
//! necessarily collective and cannot be a local sweep. Every rank advertises
//! (a) the gid of every live vertex/edge it OWNS and (b) every vertex/edge gid
//! referenced by a live OWNED FACE, both to the gid coordinator (gid % size);
//! a referenced gid with no live claim is dangling and the coordinator reports
//! it back to the referencing rank.
//!
//! WHY THE REFERENCE SET IS EXACTLY "live owned faces": that is the set
//! rebuildHalo() reconstructs the mesh from, so it is precisely what must
//! resolve. A live edge or vertex that no surviving face references is NOT an
//! error -- round D drops it whether or not the caller tombstoned it, which is
//! why tombstoneVertex()/tombstoneEdge() feed this check rather than driving a
//! removal pass. EdgeField::Faces is deliberately NOT treated as a reference:
//! it is best-effort by design and already carries gids naming faces no rank
//! holds (see Tessera_FaceAdjacency.hpp).
//!
//! Throws std::runtime_error naming the offending live face and the dead gid.
//! The throw is COLLECTIVE -- the count is Allreduced and every rank throws, so
//! a caller bug cannot deadlock the ranks that happen not to see it.
template <class MeshT>
void verifyTombstoneClosure( MeshT& mesh )
{
    MPI_Comm comm = mesh.comm();
    const int R = mesh.rank();
    const int size = mesh.commSize();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vo = Cabana::slice<VertexField::Owner>( hv );
    auto eg = Cabana::slice<EdgeField::Gid>( he );
    auto eo = Cabana::slice<EdgeField::Owner>( he );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    auto fe = Cabana::slice<FaceField::Edges>( hf );

    //! (referenced entity gid, the live face that references it). `kind` keeps
    //! the two entity kinds in one exchange: 0 = vertex, 1 = edge.
    struct Ref
    {
        GlobalId gid;
        GlobalId faceGid;
        int kind;
    };
    struct Alive
    {
        GlobalId gid;
        int kind;
    };

    std::vector<std::vector<Alive>> alive( size );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
        if ( vg( i ) != invalid_gid && vo( i ) == static_cast<Rank>( R ) )
            alive[gidCoordRank( vg( i ), size )].push_back( { vg( i ), 0 } );
    for ( std::size_t i = 0; i < mesh.numEdges(); ++i )
        if ( eg( i ) != invalid_gid && eo( i ) == static_cast<Rank>( R ) )
            alive[gidCoordRank( eg( i ), size )].push_back( { eg( i ), 1 } );

    std::vector<std::vector<Ref>> refs( size );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        if ( fg( f ) == invalid_gid )
            continue; // a dead face references nothing
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId v = fv( f, k );
            const GlobalId e = fe( f, k );
            refs[gidCoordRank( v, size )].push_back( { v, fg( f ), 0 } );
            refs[gidCoordRank( e, size )].push_back( { e, fg( f ), 1 } );
        }
    }

    auto aliveGot = allToAllV( comm, alive );
    auto refsGot = allToAllV( comm, refs );

    std::set<std::pair<GlobalId, int>> liveSet;
    for ( const auto& a : aliveGot.data )
        liveSet.emplace( a.gid, a.kind );

    // Report every dangling reference back to the rank that made it, so the
    // message names a live face THAT RANK owns.
    std::vector<std::vector<Ref>> back( size );
    for ( int s = 0; s < size; ++s )
    {
        const Ref* p = refsGot.from( s );
        const int c = refsGot.count( s );
        for ( int i = 0; i < c; ++i )
            if ( liveSet.find( { p[i].gid, p[i].kind } ) == liveSet.end() )
                back[s].push_back( p[i] );
    }
    auto backGot = allToAllV( comm, back );

    long long localBad = static_cast<long long>( backGot.data.size() );
    long long globalBad = 0;
    MPI_Allreduce( &localBad, &globalBad, 1, MPI_LONG_LONG, MPI_SUM, comm );
    if ( globalBad == 0 )
        return;

    std::string msg = "Tessera::compact: the tombstone set does not close -- " +
                      std::to_string( globalBad ) +
                      " reference(s) from a LIVE face to a DEAD entity. ";
    if ( localBad > 0 )
    {
        const Ref& r = backGot.data[0];
        msg += "On rank " + std::to_string( R ) + ", live face gid " +
               std::to_string( r.faceGid ) + " references dead " +
               ( r.kind == 0 ? "vertex" : "edge" ) + " gid " +
               std::to_string( r.gid ) + " (and " +
               std::to_string( localBad - 1 ) + " more on this rank). ";
    }
    else
    {
        msg += "None of them originate on rank " + std::to_string( R ) +
               "; this throw is the collective half. ";
    }
    msg += "Tombstoning does NOT repair connectivity: a caller that kills a "
           "face must also kill exactly the entities that become orphaned, and "
           "must not leave a live face referencing a dead gid. compact() "
           "refuses rather than producing a corrupt mesh.";
    throw std::runtime_error( msg );
}

//! Drop the dead OWNED faces, keeping the survivors in their current (already
//! gid-ascending) order, and hand the mesh to rebuildHalo(). Ghost faces are
//! dropped wholesale: the ghost set genuinely CHANGES under a removal -- a
//! ghost whose owner deleted it must disappear -- so this cannot be a plan
//! patch, and round C re-fetches the true ghost layer anyway.
template <class MeshT>
void dropDeadOwnedFaces( MeshT& mesh )
{
    using FMT = typename MeshT::face_member_types;
    const std::size_t nof = mesh.numOwnedFaces();

    Cabana::AoSoA<FMT, Kokkos::HostSpace> hf( "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fg = Cabana::slice<FaceField::Gid>( hf );

    std::size_t nLive = 0;
    for ( std::size_t f = 0; f < nof; ++f )
        if ( fg( f ) != invalid_gid )
            ++nLive;

    Cabana::AoSoA<FMT, Kokkos::HostSpace> lf( "lf", nLive );
    std::size_t j = 0;
    for ( std::size_t f = 0; f < nof; ++f )
        if ( fg( f ) != invalid_gid )
            lf.setTuple( j++, hf.getTuple( f ) );

    mesh.resizeFaces( nLive );
    Cabana::deep_copy( mesh.faces(), lf );
    // Owned-only face set: rebuildHalo() reads exactly [0, numOwnedFaces) and
    // rebuilds everything else. The vertex/edge owned counts are untouched
    // here and are overwritten by round D.
    mesh.setOwnedCounts( mesh.numOwnedVertices(), mesh.numOwnedEdges(),
                         nLive );
}

//! ORDER-PRESERVING GLOBAL RENUMBERING of one entity kind: the new gid of an
//! entity is the NUMBER OF LIVE ENTITIES OF THAT KIND WITH A SMALLER OLD GID.
//!
//! That definition, rather than "MPI_Exscan over owned counts", is what makes
//! renumbering RANK-COUNT DETERMINISTIC. An exscan hands rank r a contiguous
//! block, so the resulting old->new map depends on WHO OWNED WHAT and two runs
//! at different rank counts disagree. An order statistic depends only on the
//! set of live gids, so every rank count produces the identical global map --
//! and it still yields exactly [0, N) per kind, which is the whole point.
//!
//! Computed without gathering: gids are bucketed by VALUE (bucket b = rank b
//! handles gid range b), each bucket coordinator sorts its own bucket and an
//! MPI_Exscan over the per-bucket counts gives each bucket's base. Buckets
//! ascend with rank index, so the exscan is exactly the prefix count. The
//! bucketing is a load-balance choice only -- the answer does not depend on it.
//!
//! `ownedGids` is this rank's owned gids; the return value is their new gids,
//! positionally. Collective.
inline std::vector<GlobalId>
orderPreservingRenumber( MPI_Comm comm, int size,
                         const std::vector<GlobalId>& ownedGids )
{
    GlobalId localMax = 0;
    for ( GlobalId g : ownedGids )
        localMax = std::max( localMax, g );
    GlobalId globalMax = 0;
    MPI_Allreduce( &localMax, &globalMax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                   comm );
    const GlobalId width =
        globalMax / static_cast<GlobalId>( size ) + 1; // >= 1

    std::vector<std::vector<GlobalId>> send( size );
    std::vector<std::vector<std::size_t>> sentIdx( size );
    for ( std::size_t i = 0; i < ownedGids.size(); ++i )
    {
        int b = static_cast<int>( ownedGids[i] / width );
        if ( b >= size )
            b = size - 1;
        send[b].push_back( ownedGids[i] );
        sentIdx[b].push_back( i );
    }
    auto got = allToAllV( comm, send );

    // This bucket's live gids, ascending. Gids are globally unique, so the only
    // duplicates possible are none -- but dedup anyway so the rank is a rank.
    std::set<GlobalId> bucket( got.data.begin(), got.data.end() );
    long long nBucket = static_cast<long long>( bucket.size() );
    long long base = 0;
    MPI_Exscan( &nBucket, &base, 1, MPI_LONG_LONG, MPI_SUM, comm );
    int rank = 0;
    MPI_Comm_rank( comm, &rank );
    if ( rank == 0 )
        base = 0; // MPI_Exscan leaves rank 0's receive buffer undefined

    std::map<GlobalId, GlobalId> newOf;
    {
        GlobalId n = static_cast<GlobalId>( base );
        for ( GlobalId g : bucket )
            newOf.emplace( g, n++ );
    }

    // Reply in the order received, so the requester can walk sentIdx in step.
    std::vector<std::vector<GlobalId>> reply( size );
    for ( int s = 0; s < size; ++s )
    {
        const GlobalId* p = got.from( s );
        const int c = got.count( s );
        for ( int i = 0; i < c; ++i )
            reply[s].push_back( newOf.at( p[i] ) );
    }
    auto replyGot = allToAllV( comm, reply );

    std::vector<GlobalId> out( ownedGids.size(), invalid_gid );
    for ( int b = 0; b < size; ++b )
    {
        const GlobalId* p = replyGot.from( b );
        const int c = replyGot.count( b );
        for ( int i = 0; i < c; ++i )
            out[sentIdx[b][i]] = p[i];
    }
    return out;
}

//! compact() without the editing-family claim, so compactAndRenumberGids() can
//! reuse it without claiming the family twice.
template <class MeshT>
CompactStats compactImpl( MeshT& mesh,
                          MeshHalo<typename MeshT::memory_space>& halo )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_COMPACT );
    CompactStats st;
    st.gidSpaceBefore = gidSpace( mesh );
    long long before[3];
    ownedCounts( mesh, before );

    verifyTombstoneClosure( mesh );
    dropDeadOwnedFaces( mesh );
    rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) );

    long long after[3];
    ownedCounts( mesh, after );
    st.verticesRemoved = before[0] - after[0];
    st.edgesRemoved = before[1] - after[1];
    st.facesRemoved = before[2] - after[2];
    st.gidSpaceAfter = gidSpace( mesh );
    return st;
}

} // namespace detail

// ============================================================================
// Tombstone marking
// ============================================================================
//
//! Mark a local entity dead. Tombstoning does NOT repair connectivity: a caller
//! that kills a face must also kill the entities that become orphaned, and must
//! not leave a live entity referencing a dead gid. compact() verifies this and
//! throws on a dangling reference rather than producing a corrupt mesh.
//!
//! Purely local and non-collective; the entity's OWNER is the rank whose mark
//! counts, and a mark on a ghost copy is ignored (compact() re-fetches the
//! entity from its owner). Overwrites `Gid`, so the entity's identity is gone
//! from this rank the moment it is marked.
//!
//! Marking a vertex or an edge is OPTIONAL for removal: compact() drops every
//! vertex and edge that no surviving face references, marked or not. What the
//! mark buys is the CHECK -- it is how compact() can tell "the caller meant to
//! delete this" from "the caller forgot", and turn the latter into a throw.
template <class MeshT>
void tombstoneVertex( MeshT& mesh, LocalIndex v )
{
    auto gid = Cabana::slice<VertexField::Gid>( mesh.vertices() );
    Kokkos::parallel_for(
        "tessera_tombstone_vertex",
        Kokkos::RangePolicy<typename MeshT::execution_space>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) { gid( v ) = invalid_gid; } );
    Kokkos::fence();
}

//! Mark a local edge dead. See tombstoneVertex().
template <class MeshT>
void tombstoneEdge( MeshT& mesh, LocalIndex e )
{
    auto gid = Cabana::slice<EdgeField::Gid>( mesh.edges() );
    Kokkos::parallel_for(
        "tessera_tombstone_edge",
        Kokkos::RangePolicy<typename MeshT::execution_space>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) { gid( e ) = invalid_gid; } );
    Kokkos::fence();
}

//! Mark a local face dead. See tombstoneVertex(). Unlike a vertex or an edge, a
//! face is removed ONLY if it is marked: faces are what compact() rebuilds the
//! mesh from.
template <class MeshT>
void tombstoneFace( MeshT& mesh, LocalIndex f )
{
    auto gid = Cabana::slice<FaceField::Gid>( mesh.faces() );
    Kokkos::parallel_for(
        "tessera_tombstone_face",
        Kokkos::RangePolicy<typename MeshT::execution_space>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) { gid( f ) = invalid_gid; } );
    Kokkos::fence();
}

// ============================================================================
// compact
// ============================================================================

//! Remove every tombstoned entity, restore owned-first ordering, rebuild the two
//! CSRs, the key side tables, and the three halo plans.
//!
//! GIDS ARE PRESERVED -- a surviving entity keeps the gid it had, so every
//! cross-rank reference held by another rank stays valid and no communication is
//! needed to agree on renaming. Connectivity fields hold gids, so there is NO
//! connectivity rewrite here at all: that is the payoff of the preservation
//! decision, and it is what makes compact() cheap enough to call every step.
//!
//! What is removed: every tombstoned OWNED FACE, plus every vertex and edge that
//! no surviving face references (whether or not it was tombstoned -- see
//! tombstoneVertex()). The tombstone set must CLOSE: a live face may not
//! reference an entity its owner marked dead, and compact() throws
//! std::runtime_error naming the offending face and gid rather than producing a
//! corrupt mesh.
//!
//! A rank may legitimately end with ZERO owned entities; its peers drop it from
//! their plans and the collectives still complete. That is a real load-balance
//! state, not a pathological one.
//!
//! Belongs to the REMESH editing family, so compacting a refine()d mesh throws,
//! and compacting a freshly built mesh TAGS it Remesh -- a later refine() on it
//! is then refused (Tessera_EditFamily.hpp).
//!
//! Collective (the halo rebuild is). Preserves halo depth. Bumps generation.
//!
//! INVALIDATION: reallocates the AoSoAs, key Views and CSRs and replaces the
//! halo plans, so every slice/CSR/key-View taken out before this call is
//! dangling. Re-slice from the mesh afterwards.
template <class MeshT>
CompactStats compact( MeshT& mesh,
                      MeshHalo<typename MeshT::memory_space>& halo )
{
    requireEditFamily( mesh, EditFamily::Remesh, "compact" );
    return detail::compactImpl( mesh, halo );
}

//! compact(), then renumber gids contiguously from 0 per kind, so the gid space
//! stops growing across many edit rounds (see the leak note at the top of this
//! header). Strictly more expensive than compact() -- every rank must learn the
//! new gid of every entity it ghosts, every connectivity field must be
//! rewritten, and the halo is rebuilt a SECOND time because the plans were keyed
//! on old gids -- so it is a separate call the caller invokes periodically
//! rather than something compact() does every time. Two halo rebuilds is the
//! honest cost and is not fused.
//!
//! The new gid of an entity is the number of live entities of its kind with a
//! smaller old gid, which makes the result RANK-COUNT DETERMINISTIC as well as
//! contiguous (detail::orderPreservingRenumber()).
//!
//! The new gids ride to the ghosts on the `Gid` FIELD ITSELF, not on a side
//! channel: haloExchange() is whole-AoSoA, whole-tuple, and Tessera has no
//! generic per-entity value exchange. So the owned gids are rewritten in place,
//! all three AoSoAs are exchanged (every ghost receives its owner's new gid
//! inside the tuple), and the local old->new map is reconstructed from the old
//! gids saved beforehand.
//!
//! EdgeField::Faces is mapped where the named face is held locally and set to
//! invalid_gid where it is not. It is best-effort by design
//! (Tessera_FaceAdjacency.hpp) and a gid left in the OLD space after a
//! renumbering would silently ALIAS a different live face, which is worse than
//! absent.
//!
//! Same family, collectivity, depth and invalidation contract as compact().
template <class MeshT>
CompactStats
compactAndRenumberGids( MeshT& mesh,
                        MeshHalo<typename MeshT::memory_space>& halo )
{
    requireEditFamily( mesh, EditFamily::Remesh, "compactAndRenumberGids" );

    CompactStats st = detail::compactImpl( mesh, halo );

    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_COMPACT_RENUMBER );
    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();

    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;

    const std::size_t nv = mesh.numVertices();
    const std::size_t ne = mesh.numEdges();
    const std::size_t nf = mesh.numFaces();

    Cabana::AoSoA<VMT, Kokkos::HostSpace> hv( "hv", nv );
    Cabana::AoSoA<EMT, Kokkos::HostSpace> he( "he", ne );
    Cabana::AoSoA<FMT, Kokkos::HostSpace> hf( "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );

    // Old gids by LOCAL INDEX. haloExchange() does not reorder, so local index
    // i still names the same entity after the exchange below -- which is what
    // lets the old->new map be reconstructed without a side channel.
    std::vector<GlobalId> oldV( nv ), oldE( ne ), oldF( nf );
    {
        auto g = Cabana::slice<VertexField::Gid>( hv );
        for ( std::size_t i = 0; i < nv; ++i )
            oldV[i] = g( i );
    }
    {
        auto g = Cabana::slice<EdgeField::Gid>( he );
        for ( std::size_t i = 0; i < ne; ++i )
            oldE[i] = g( i );
    }
    {
        auto g = Cabana::slice<FaceField::Gid>( hf );
        for ( std::size_t i = 0; i < nf; ++i )
            oldF[i] = g( i );
    }

    // Owned blocks only: a ghost's new gid is its owner's answer, and it
    // arrives in the exchange.
    auto ownedPrefix = []( const std::vector<GlobalId>& all, std::size_t n )
    { return std::vector<GlobalId>( all.begin(), all.begin() + n ); };
    const std::vector<GlobalId> newOwnedV = detail::orderPreservingRenumber(
        comm, size, ownedPrefix( oldV, mesh.numOwnedVertices() ) );
    const std::vector<GlobalId> newOwnedE = detail::orderPreservingRenumber(
        comm, size, ownedPrefix( oldE, mesh.numOwnedEdges() ) );
    const std::vector<GlobalId> newOwnedF = detail::orderPreservingRenumber(
        comm, size, ownedPrefix( oldF, mesh.numOwnedFaces() ) );

    {
        auto g = Cabana::slice<VertexField::Gid>( hv );
        for ( std::size_t i = 0; i < newOwnedV.size(); ++i )
            g( i ) = newOwnedV[i];
    }
    {
        auto g = Cabana::slice<EdgeField::Gid>( he );
        for ( std::size_t i = 0; i < newOwnedE.size(); ++i )
            g( i ) = newOwnedE[i];
    }
    {
        auto g = Cabana::slice<FaceField::Gid>( hf );
        for ( std::size_t i = 0; i < newOwnedF.size(); ++i )
            g( i ) = newOwnedF[i];
    }
    Cabana::deep_copy( mesh.vertices(), hv );
    Cabana::deep_copy( mesh.edges(), he );
    Cabana::deep_copy( mesh.faces(), hf );

    // Every ghost learns its owner's new gid, inside the tuple.
    haloExchange( mesh, halo );

    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );

    std::map<GlobalId, GlobalId> vmap, emap, fmap;
    {
        auto g = Cabana::slice<VertexField::Gid>( hv );
        for ( std::size_t i = 0; i < nv; ++i )
            vmap[oldV[i]] = g( i );
    }
    {
        auto g = Cabana::slice<EdgeField::Gid>( he );
        for ( std::size_t i = 0; i < ne; ++i )
            emap[oldE[i]] = g( i );
    }
    {
        auto g = Cabana::slice<FaceField::Gid>( hf );
        for ( std::size_t i = 0; i < nf; ++i )
            fmap[oldF[i]] = g( i );
    }

    auto lookup = []( const std::map<GlobalId, GlobalId>& m, GlobalId g,
                      const char* what )
    {
        auto it = m.find( g );
        if ( it == m.end() )
            throw std::runtime_error(
                std::string(
                    "Tessera::compactAndRenumberGids: unresolvable " ) +
                what + " gid " + std::to_string( g ) +
                " -- every entity a local face references is held locally "
                "after compact() (rebuildHalo()'s round G postcondition), so "
                "this means the mesh was not compact on entry." );
        return it->second;
    };

    {
        auto verts = Cabana::slice<FaceField::Verts>( hf );
        auto edges = Cabana::slice<FaceField::Edges>( hf );
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                verts( f, k ) = lookup( vmap, verts( f, k ), "vertex" );
                edges( f, k ) = lookup( emap, edges( f, k ), "edge" );
            }
    }
    {
        auto verts = Cabana::slice<EdgeField::Verts>( he );
        auto faces = Cabana::slice<EdgeField::Faces>( he );
        for ( std::size_t e = 0; e < ne; ++e )
        {
            for ( int j = 0; j < 2; ++j )
                verts( e, j ) = lookup( vmap, verts( e, j ), "vertex" );
            for ( int j = 0; j < 2; ++j )
            {
                auto it = fmap.find( faces( e, j ) );
                faces( e, j ) = ( it == fmap.end() ) ? invalid_gid : it->second;
            }
        }
    }
    Cabana::deep_copy( mesh.edges(), he );
    Cabana::deep_copy( mesh.faces(), hf );

    // The plans and both key tables were keyed on the OLD gids.
    rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) );

    st.gidSpaceAfter = detail::gidSpace( mesh );
    return st;
}

} // namespace Tessera

#endif // TESSERA_COMPACT_HPP
