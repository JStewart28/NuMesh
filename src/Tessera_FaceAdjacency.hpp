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

#ifndef TESSERA_FACE_ADJACENCY_HPP
#define TESSERA_FACE_ADJACENCY_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_CsrAdjacency.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_GenerationGuard.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_RefineParallel.hpp" // detail::edgeCoordRank
#include "Tessera_Types.hpp"

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// FaceAdjacency — face -> face adjacency through shared edges
// ============================================================================
//
// "Which faces share an edge with this face" is not derivable locally, for two
// independent reasons, and this is why the query needs a collective builder
// rather than a walk over the connectivity the mesh already holds.
//
//   1. EdgeField::Faces holds the GLOBAL IDS of an edge's (at most two)
//      incident faces. Those gids may name faces this rank does not hold, and
//      migrate() carries them verbatim without repair. A gid is not a usable
//      neighbour handle.
//   2. A vertex-incidence walk only finds an edge-neighbour that happens to be
//      co-resident. The local face set is `owned faces plus faces incident on
//      an owned vertex`, so an owned face all three of whose corners are ghosts
//      can have edge-neighbours that are not locally held at all. The 1-deep
//      VERTEX halo does not guarantee edge-neighbour co-residency.
//
// refine() says exactly this, which is why its 2:1 balance routes every
// cross-rank mark decision through an EDGE COORDINATOR rather than through the
// halo. buildFaceAdjacency() reuses that machinery (detail::edgeCoordRank +
// allToAllV) and adds no new communication mechanism.
//
// THE RETURN TYPE HAS TWO HALVES, AND WHICH ONE YOU MAY USE IS THE
// PRECONDITION SPLIT WORTH READING BEFORE ANYTHING ELSE:
//
//   * A TOPOLOGICAL consumer -- growing a refinement mark by neighbour rings,
//     choosing an independent set of edge flips, any conflict-resolution pass
//     that COMMUNICATES with the neighbour's owner -- uses `nbrGid`/`nbrOwner`
//     only. Those are always valid whether the neighbour is resident or not, so
//     such a consumer works at halo depth 1 and needs no further precondition.
//   * A GEOMETRIC consumer -- one that reads the neighbour's vertex positions
//     out of the local AoSoA -- must go through `csr`, whose entries are
//     invalid_local exactly where the neighbour is not held. It must CHECK
//     `numNonResident == 0` rather than assume it. A deeper halo
//     (distribute(..., depth >= 2)) makes that likely but does not guarantee it
//     in general.
//
// Returning `invalid_local` and saying nothing would recreate the
// silent-incompleteness failure mode a short stencil row has; returning gids
// only would be complete but unusable on device. Hence both, with a counter.
//
// WHAT COUNTS AS "SHARING AN EDGE" is exact EdgeKey equality: two faces are
// adjacent iff they have a common edge, i.e. a common pair of corner vertex
// gids. On a conforming mesh (the `RefinementMode::Conforming` default) that
// makes every face's degree exactly 3 on a closed surface. Under
// `RefinementMode::HangingNode2to1` a T-junction is NOT a shared edge: the
// coarse side holds (a,b) while the fine side holds (a,m) and (m,b), three
// distinct edges with one incidence each, so a face on either side of a
// T-junction has degree < 3. That is the mode contract showing through, not a
// defect -- HangingNode2to1 keeps no record of which edges carry a hanging node
// (README -> Known Issues), so there is nothing local or remote to match (a,m)
// against (a,b) with. A consumer that needs geometric neighbours across a
// refinement front wants a Conforming mesh.
//
// CONFORMING MODE holds no surprise either, and the reason is worth recording
// because the obvious worry is unfounded: a `Conforming` mesh's face AoSoA
// stores ONLY the visible (closed) faces. The retired red parents exist solely
// inside refine(), which un-closes, splits, and re-closes within one call; they
// are never stored. So adjacency over the local faces IS adjacency over the
// visible faces, with nothing to skip and no way for a consumer to be handed a
// row for a face that is not part of the current triangulation.
//
// INVALIDATION: `csr` is generation-guarded (GenerationHandle) exactly like
// VertexStencil::csr -- stamped with the mesh generation at build time, so a
// handle held across a topology op aborts with a diagnostic instead of reading
// dangling storage. `nbrGid`/`nbrOwner` are bare Views parallel to
// `csr.get().neighbors`; they are only meaningful together with it, so the
// guard on the CSR is the guard on all three. Rebuild after any
// distribute/migrate/refine/splitEdges/loadBalance.
//
// NOT PROVIDED, deliberately: growing a refinement mask by neighbour rings (the
// consumer's loop over this CSR plus a mark exchange it already has), vertex ->
// vertex or face -> vertex adjacency (both already exist or are derivable), and
// incremental maintenance across a topology edit (it is rebuilt, like every
// other derived structure, and the generation guard enforces that).
template <class MemorySpace>
struct FaceAdjacency
{
    using memory_space = MemorySpace;

    //! Row per LOCAL face; entries are local face indices, or invalid_local
    //! when that neighbour is not held on this rank. Rows for OWNED faces are
    //! complete in the sense that every true edge-neighbour appears -- as a
    //! local index if resident, as invalid_local if not. Rows for GHOST faces
    //! are best-effort and may be short; do not iterate them.
    GenerationHandle<CsrAdjacency<MemorySpace>> csr;

    //! Parallel to csr.get().neighbors: the neighbour's global id and owning
    //! rank. ALWAYS valid, resident or not. This is what a mark-propagation
    //! consumer uses -- it sends to nbrOwner and names the face by nbrGid, and
    //! never needs the neighbour locally.
    Kokkos::View<GlobalId*, MemorySpace> nbrGid;
    Kokkos::View<Rank*, MemorySpace> nbrOwner;

    //! Count of owned-row entries with invalid_local, summed over this rank.
    //! Zero means every owned face's neighbours are co-resident, so a geometric
    //! consumer may use `csr` directly. GHOST rows are excluded: they are
    //! best-effort by contract, so counting them would make the flag say
    //! nothing about the rows a consumer is allowed to iterate.
    long long numNonResident = 0;
};

namespace detail
{

//! One (edge, incident owned face) advertisement to the edge coordinator.
struct FaceAdjAdvert
{
    EdgeKey key;
    GlobalId faceGid;
    Rank faceOwner;
};

//! The coordinator's reply to one advertisement: the OTHER face incident on
//! that edge. `faceGid` echoes the advertised face so the receiver knows which
//! of its rows the entry belongs to.
struct FaceAdjReply
{
    EdgeKey key;
    GlobalId faceGid;
    GlobalId nbrGid;
    Rank nbrOwner;
};

//! One neighbour entry before flattening: sorted ascending by gid so a row's
//! order is a function of global ids alone and therefore rank-count invariant.
struct FaceAdjEntry
{
    GlobalId gid;
    Rank owner;
    LocalIndex local;

    bool operator<( const FaceAdjEntry& o ) const { return gid < o.gid; }
};

} // namespace detail

// ============================================================================
// buildFaceAdjacency
// ============================================================================
//
// Build the edge-adjacency of every local face. COLLECTIVE on mesh.comm().
// Rows are sorted ascending by neighbour GLOBAL ID (not local index), so the
// row order is rank-count invariant and two runs at different rank counts can
// be compared as exact lists rather than as sets.
//
// Structure, mirroring refine() Phase 2:
//
//   1. ADVERTISE. Each OWNED face sends (EdgeKey, faceGid, faceOwner) to
//      edgeCoordRank(key, comm_size) for each of its three edges. Routing is by
//      the hash; disambiguation is by the full EdgeKey at the coordinator,
//      exactly as refine() does. Ghost faces are not advertised, so no extra
//      traffic.
//   2. COORDINATE. Each coordinator groups its advertisements by EdgeKey,
//      DEDUPLICATED BY FACE GID, and so sees the edge's true set of incident
//      faces. The dedup is not cosmetic: on a replicated mesh (straight out of
//      the builder, before distribute()) every rank advertises every face, so
//      counting advertisements rather than distinct faces would read a
//      perfectly manifold edge as 2*comm_size-fold and abort. It replies to
//      each advertiser with the OTHER face's (gid, owner). One advertisement is
//      a boundary edge and yields no reply; more than two distinct faces is
//      NON-MANIFOLD input and fails loudly naming the offending EdgeKey, rather
//      than silently truncating -- Tessera's data model assumes manifold, and
//      buildFromTriangleSoup() will happily accept a soup that is not (it keeps
//      the first two incidences in EdgeField::Faces and drops the rest).
//   3. ASSEMBLE owned rows from the replies; fill ghost rows best-effort from
//      purely local information.
//   4. RESOLVE to local indices through a gid -> local face index map built on
//      the host from the face AoSoA's Gid slice (a std::unordered_map, as
//      buildVertexStencil already does), accumulating numNonResident.
//   5. SORT each row by nbrGid and deep-copy the three arrays to the device.
//
// GHOST ROWS are derived locally rather than by advertising the ghosts, and
// from the LOCAL FACE -> EDGE incidence rather than from EdgeField::Faces. Both
// choices are about not lying: EdgeField::Faces holds gids that migrate()
// carries verbatim, so it can name a face nothing on this rank holds and whose
// owner is unknown, whereas pairing up the local faces that reference the same
// edge gid yields entries whose gid, owner AND local index are all read from
// the face AoSoA and are therefore all true. It is still best-effort -- a
// ghost's true neighbour that is not resident simply does not appear, and the
// row is short with no invalid_local to mark the gap. That is what
// "do not iterate ghost rows" means.
//
// The whole build runs on the host with ordered containers and then deep-copies
// -- the same locus and idiom as buildVertexStencil() and the halo builder.
//
// The non-manifold failure is COLLECTIVE, not local: the offending edge is seen
// by one coordinator, but a throw on that rank alone would deadlock every other
// rank in the following allToAllV. So the offender's rank is agreed by
// MPI_Allreduce(MAX) and its key broadcast, and then every rank throws the same
// std::runtime_error.
template <class MeshT>
FaceAdjacency<typename MeshT::memory_space> buildFaceAdjacency( MeshT& mesh )
{
    using memory_space = typename MeshT::memory_space;

    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();

    const int nLocalF = static_cast<int>( mesh.numFaces() );
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );
    const int nLocalE = static_cast<int>( mesh.numEdges() );

    // ---- host snapshots ----------------------------------------------------
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "faceadj_hf", static_cast<std::size_t>( nLocalF ) );
    Cabana::deep_copy( hf, mesh.faces() );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_own = Cabana::slice<FaceField::Owner>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_edges = Cabana::slice<FaceField::Edges>( hf );

    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "faceadj_he", static_cast<std::size_t>( nLocalE ) );
    Cabana::deep_copy( he, mesh.edges() );
    auto e_gid = Cabana::slice<EdgeField::Gid>( he );

    std::unordered_map<GlobalId, int> localOfFaceGid;
    localOfFaceGid.reserve( static_cast<std::size_t>( nLocalF ) * 2 );
    for ( int f = 0; f < nLocalF; ++f )
        localOfFaceGid[f_gid( f )] = f;

    // ---- 1. advertise every owned face's three edges ------------------------
    std::vector<std::vector<detail::FaceAdjAdvert>> send(
        static_cast<std::size_t>( size ) );
    for ( int f = 0; f < nOwnedF; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const EdgeKey key =
                makeEdgeKey( f_verts( f, k ), f_verts( f, ( k + 1 ) % 3 ) );
            send[detail::edgeCoordRank( key, size )].push_back(
                { key, f_gid( f ), f_own( f ) } );
        }
    auto adverts = allToAllV( comm, send );

    // ---- 2. coordinate: group by EdgeKey, dedup by face gid ----------------
    std::map<EdgeKey, std::vector<std::pair<GlobalId, Rank>>> incident;
    for ( const auto& a : adverts.data )
    {
        auto& v = incident[a.key];
        bool seen = false;
        for ( const auto& p : v )
            if ( p.first == a.faceGid )
            {
                seen = true;
                break;
            }
        if ( !seen )
            v.emplace_back( a.faceGid, a.faceOwner );
    }

    // Non-manifold detection, agreed collectively so every rank throws.
    {
        EdgeKey bad{};
        std::size_t badCount = 0;
        for ( const auto& kv : incident )
            if ( kv.second.size() > 2 )
            {
                bad = kv.first;
                badCount = kv.second.size();
                break;
            }
        int myBad = badCount > 0 ? mesh.rank() : -1;
        int badRank = -1;
        MPI_Allreduce( &myBad, &badRank, 1, MPI_INT, MPI_MAX, comm );
        if ( badRank >= 0 )
        {
            GlobalId pack[3] = { bad.id[0], bad.id[1],
                                 static_cast<GlobalId>( badCount ) };
            MPI_Bcast( pack, 3, MPI_UINT64_T, badRank, comm );
            throw std::runtime_error(
                std::string( "Tessera::buildFaceAdjacency: NON-MANIFOLD input. "
                             "The edge with EdgeKey {" ) +
                std::to_string( pack[0] ) + ", " + std::to_string( pack[1] ) +
                "} has " + std::to_string( pack[2] ) +
                " distinct incident faces; Tessera's data model assumes at "
                "most "
                "two. Truncating to two would silently discard a face's "
                "adjacency, so this is a hard failure. Fix the input soup or "
                "split the surface into manifold components." );
        }
    }

    // Reply to each advertisement with the OTHER incident face.
    std::vector<std::vector<detail::FaceAdjReply>> back(
        static_cast<std::size_t>( size ) );
    for ( int r = 0; r < size; ++r )
    {
        const detail::FaceAdjAdvert* first = adverts.from( r );
        const int n = adverts.count( r );
        for ( int i = 0; i < n; ++i )
        {
            const detail::FaceAdjAdvert& a = first[i];
            const auto& v = incident[a.key];
            for ( const auto& p : v )
                if ( p.first != a.faceGid )
                    back[r].push_back(
                        { a.key, a.faceGid, p.first, p.second } );
        }
    }
    auto replies = allToAllV( comm, back );

    // ---- 3./4. assemble rows -----------------------------------------------
    std::vector<std::vector<detail::FaceAdjEntry>> rows(
        static_cast<std::size_t>( nLocalF ) );

    // Owned rows: purely from the coordinator replies.
    for ( const auto& m : replies.data )
    {
        auto it = localOfFaceGid.find( m.faceGid );
        if ( it == localOfFaceGid.end() )
            continue; // cannot happen: we advertised it, so we hold it.
        auto nb = localOfFaceGid.find( m.nbrGid );
        rows[it->second].push_back(
            { m.nbrGid, m.nbrOwner,
              nb == localOfFaceGid.end() ? invalid_local : nb->second } );
    }

    // Ghost rows: best-effort, from the local face -> edge incidence.
    if ( nOwnedF < nLocalF )
    {
        std::unordered_map<GlobalId, int> localOfEdgeGid;
        localOfEdgeGid.reserve( static_cast<std::size_t>( nLocalE ) * 2 );
        for ( int e = 0; e < nLocalE; ++e )
            localOfEdgeGid[e_gid( e )] = e;

        std::vector<std::vector<int>> facesOfEdge(
            static_cast<std::size_t>( nLocalE ) );
        for ( int f = 0; f < nLocalF; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                auto it = localOfEdgeGid.find( f_edges( f, k ) );
                if ( it != localOfEdgeGid.end() )
                    facesOfEdge[it->second].push_back( f );
            }

        for ( int f = nOwnedF; f < nLocalF; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                auto it = localOfEdgeGid.find( f_edges( f, k ) );
                if ( it == localOfEdgeGid.end() )
                    continue;
                for ( int g : facesOfEdge[it->second] )
                    if ( g != f )
                        rows[f].push_back( { f_gid( g ), f_own( g ), g } );
            }
    }

    // ---- 5. sort each row by neighbour gid, then flatten -------------------
    std::vector<int> off( static_cast<std::size_t>( nLocalF ) + 1, 0 );
    std::vector<LocalIndex> nbr;
    std::vector<GlobalId> ngid;
    std::vector<Rank> nown;
    long long numNonResident = 0;
    for ( int f = 0; f < nLocalF; ++f )
    {
        std::sort( rows[f].begin(), rows[f].end() );
        for ( const auto& e : rows[f] )
        {
            nbr.push_back( e.local );
            ngid.push_back( e.gid );
            nown.push_back( e.owner );
            if ( f < nOwnedF && e.local == invalid_local )
                ++numNonResident;
        }
        off[f + 1] = static_cast<int>( nbr.size() );
    }

    CsrAdjacency<memory_space> csr;
    detail::fillCsr( csr, off, nbr, "face_adjacency" );

    FaceAdjacency<memory_space> adj;
    adj.csr = GenerationHandle<CsrAdjacency<memory_space>>(
        csr, mesh.generation(), mesh.generationPtr() );
    adj.numNonResident = numNonResident;

    const std::size_t nEntries = ngid.size();
    adj.nbrGid = Kokkos::View<GlobalId*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_adj_nbr_gid" ),
        nEntries );
    adj.nbrOwner = Kokkos::View<Rank*, memory_space>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_adj_nbr_owner" ),
        nEntries );
    auto h_gid = Kokkos::create_mirror_view( adj.nbrGid );
    auto h_own = Kokkos::create_mirror_view( adj.nbrOwner );
    for ( std::size_t i = 0; i < nEntries; ++i )
    {
        h_gid( i ) = ngid[i];
        h_own( i ) = nown[i];
    }
    Kokkos::deep_copy( adj.nbrGid, h_gid );
    Kokkos::deep_copy( adj.nbrOwner, h_own );

    return adj;
}

} // namespace Tessera

#endif // TESSERA_FACE_ADJACENCY_HPP
