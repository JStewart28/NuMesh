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

#ifndef TESSERA_ZOLTAN2_BALANCER_HPP
#define TESSERA_ZOLTAN2_BALANCER_HPP

#include "Tessera_MeshMigrate.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Reduction.hpp"
#include "Tessera_Types.hpp"

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_CoordinatePartitioningGraph.hpp>
#include <Zoltan2_PartitioningProblem.hpp>

#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Tpetra_Map.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace Tessera
{

// ============================================================================
// Internal Zoltan2 geometric load balancer (Step 7b)
// ============================================================================
//
// computeLoadBalance() returns a per-owned-face destination rank, in exactly
// the order ownedFaceCentroids/Gids/Weights produce, ready to hand to
// migrate() as an externally-computed assignment would be. loadBalance() is a
// thin wrapper composing the two.
//
// Three solve modes are available (LoadBalanceMode, below). Neither of the two
// that removes the rank-0 bottleneck gathers the mesh: Distributed solves once
// over a Teuchos::MpiComm on mesh.comm(), and Sampled — the DEFAULT — solves an
// O(comm size) gid-order sample on rank 0 and classifies locally. See the
// determinism note below for why Sampled and not Distributed is the default.
//
// Never "rcb": Zoltan2's deterministic RCB algorithm breaks on Tuolumne (see
// Canopy_TreePartitioner.hpp). MultiJagged is used in every mode, and that
// finding stands regardless of which comm the solve runs over.
//
// ---------------------------------------------------------------------------
// Why the solve is no longer rank-0-only, and what the determinism concern
// actually was
// ---------------------------------------------------------------------------
//
// This header used to say that MultiJagged "is not guaranteed deterministic
// across ranks, so only rank 0 solves". That concern is about EVERY RANK
// SOLVING THE SAME PROBLEM INDEPENDENTLY and getting different answers, which
// would produce an inconsistent global partition. It does not apply to a
// SINGLE DISTRIBUTED SOLVE over an MpiComm: there is one solve, therefore one
// answer, distributed by construction. The SerialComm choice was inherited
// from a per-rank-solve pattern (Canopy's replicated tree) that this code has
// never used.
//
// What GatherRoot cost: rank 0's memory and solve time scaled with the GLOBAL
// face count while every other rank idled, and the MPI_Gatherv/MPI_Scatterv
// pair was a full global data movement on top of the migration that follows.
// Distributed removes both. Rank 0 receives ZERO faces for its solve, which is
// the measurable deliverable (tests/test_loadbalance_distributed.cpp check 6).
//
// Run-to-run reproducibility at a fixed rank count is a separate property from
// cross-rank agreement, and Zoltan2 does not promise it. It was MEASURED rather
// than assumed, and THE MEASUREMENT CAME BACK NEGATIVE: two identical meshes
// balanced with Distributed in one run disagree on 0, 4, 8, 16 or 18 of 5120
// faces at np5 depending on the invocation (np1-np4 agreed in every invocation).
// GatherRoot is no better -- its own solve is a Kokkos-parallel MultiJagged and
// a second loadBalance() of an already-balanced mesh moves 0 to 42 faces from
// one invocation to the next. So the default is Sampled, which is the ONLY mode
// measured reproducible: its classification step is exact local arithmetic on
// broadcast cut coordinates, and its gid-order sample makes the cuts independent
// of the starting partition as well. Its dest checksum was bit-identical across
// two ctest invocations x both registrations x both execution spaces at every
// rank count 1-5. Distributed remains fully available and is the better choice
// when partition quality matters more than bitwise repeatability (it fits the
// cuts to every face rather than to a sample). See README -> Known Issues and
// docs/design.md -> Load balancing; pinned by
// tests/test_loadbalance_distributed.cpp check 4.

//! Which solve strategy computeLoadBalance() uses.
enum class LoadBalanceMode
{
    //! Gather every rank's faces to rank 0, solve there over a
    //! Teuchos::SerialComm, MPI_Scatterv the assignment back. The original
    //! behaviour, retained as the reference implementation and the fallback.
    //! Rank 0's cost scales with the GLOBAL face count.
    GatherRoot,
    //! One Zoltan2 MultiJagged solve over a Teuchos::MpiComm on mesh.comm(),
    //! each rank contributing its own owned faces. No gather, no scatter; the
    //! solution is already in this rank's owned-face order, and rank 0 receives
    //! zero faces. Best partition quality of the three (the cuts see every
    //! face), but NOT run-to-run reproducible — see the determinism note above.
    Distributed,
    //! Rank 0 solves a deterministic gid-order SAMPLE (target 64 coordinates
    //! per part), broadcasts the resulting axis-aligned part-box structure, and
    //! every rank classifies its OWN faces against those boxes locally. Rank
    //! 0's cost is O(comm size), independent of the global face count, and the
    //! classification is exact local arithmetic — so the result is reproducible
    //! and independent of the starting partition, at the cost of fitting the
    //! cuts to a sample rather than to the whole mesh. DEFAULT.
    Sampled
};

//! Optional instrumentation returned by computeLoadBalance()/loadBalance().
struct LoadBalanceStats
{
    //! Mode that actually ran.
    LoadBalanceMode mode = LoadBalanceMode::Sampled;
    //! Global owned-face count at entry.
    long long globalFaces = 0;
    //! Number of faces rank 0 received as input to its solve. The point of the
    //! task: GatherRoot == globalFaces, Distributed == 0, Sampled == O(size).
    long long rootSolveFaces = 0;
    //! Sampled only: the gid stride the sample was taken with.
    long long sampleStride = 0;
    //! Sampled only: the broadcast cut structure, flattened as
    //! [partId, mins[Dim]..., maxs[Dim]...] per part box. Identical on every
    //! rank, and a pure function of the mesh (not of its current partition).
    std::vector<double> cuts;
    //! Single-rank (or otherwise trivial) fast path taken; no solve ran.
    bool fastPath = false;
};

namespace detail
{

using LbAdapter = Zoltan2::BasicVectorAdapter<Tpetra::Map<int, int64_t>>;

//! Run one Zoltan2 MultiJagged partitioning problem.
//!
//! `centroids` is row-major [i*Dim + d] over `numLocal` local coordinates;
//! `ids` are globally unique ids (real face gids in the distributed path);
//! `parts` is resized to `numLocal` and filled with the part assignment.
//! When `boxes` is non-null the axis-aligned per-part bounding boxes are kept
//! and flattened into it as [partId, mins[Dim]..., maxs[Dim]...] per box.
//!
//! Always "multijagged" — never "rcb" (broken on Tuolumne).
template <int Dim>
void lbMultiJagged( const Teuchos::RCP<const Teuchos::Comm<int>>& tcomm,
                    int numLocal, const std::vector<int64_t>& ids,
                    const std::vector<double>& centroids,
                    const std::vector<double>& weights, int numParts,
                    double imbalanceTolerance, std::vector<int>& parts,
                    std::vector<double>* boxes )
{
    using gno_t = typename LbAdapter::gno_t;
    using scalar_t = typename LbAdapter::scalar_t;
    using lno_t = typename LbAdapter::lno_t;

    // Pad every array to at least one element so the adapter never sees a null
    // pointer on a rank that happens to own nothing (routine after a skewed
    // migrate). The advertised length stays numLocal.
    const std::size_t pad = static_cast<std::size_t>( numLocal ) + 1;

    std::vector<gno_t> gids( pad, 0 );
    for ( int i = 0; i < numLocal; ++i )
        gids[i] = static_cast<gno_t>( ids[i] );

    // Deinterleave the row-major centroids into Dim contiguous per-dimension
    // arrays, as the generic multivector adapter constructor needs (so this
    // works for Dim=2 as well as Dim=3).
    std::vector<std::vector<scalar_t>> dimVals(
        Dim, std::vector<scalar_t>( pad, scalar_t( 0 ) ) );
    for ( int i = 0; i < numLocal; ++i )
        for ( int d = 0; d < Dim; ++d )
            dimVals[d][i] = static_cast<scalar_t>(
                centroids[static_cast<std::size_t>( i ) * Dim + d] );

    std::vector<scalar_t> w( pad, scalar_t( 1 ) );
    for ( int i = 0; i < numLocal; ++i )
        w[i] = static_cast<scalar_t>( weights[i] );

    std::vector<const scalar_t*> entries( Dim );
    std::vector<int> entryStride( Dim, 1 );
    for ( int d = 0; d < Dim; ++d )
        entries[d] = dimVals[d].data();
    std::vector<const scalar_t*> wvals = { w.data() };
    std::vector<int> wstride = { 1 };

    LbAdapter adapter( static_cast<lno_t>( numLocal ), gids.data(), entries,
                       entryStride, wvals, wstride );

    Teuchos::ParameterList params;
    params.set( "algorithm", "multijagged" );
    params.set( "num_global_parts", numParts );
    params.set( "imbalance_tolerance", 1.0 + imbalanceTolerance );
    params.set( "debug_level", "no_status" );
    if ( boxes )
        params.set( "mj_keep_part_boxes", true );
    Teuchos::ParameterList zoltanParams;
    zoltanParams.set( "DEBUG_LEVEL", "0" );
    params.set( "zoltan_parameters", zoltanParams );

    Zoltan2::PartitioningProblem<LbAdapter> problem( &adapter, &params, tcomm );
    problem.solve();
    const auto& solution = problem.getSolution();

    parts.assign( static_cast<std::size_t>( numLocal ), 0 );
    if ( numLocal > 0 )
    {
        const int* partsView = solution.getPartListView();
        for ( int i = 0; i < numLocal; ++i )
            parts[i] = partsView[i];
    }

    if ( boxes )
    {
        // The box vector is held by the algorithm (an RCP member), so the
        // reference stays valid while `problem` is alive — copy out here.
        const auto& pb = solution.getPartBoxesView();
        boxes->clear();
        boxes->reserve( pb.size() * ( 1 + 2 * Dim ) );
        for ( const auto& b : pb )
        {
            boxes->push_back( static_cast<double>( b.getpId() ) );
            const auto* mins = b.getlmins();
            const auto* maxs = b.getlmaxs();
            for ( int d = 0; d < Dim; ++d )
                boxes->push_back( static_cast<double>( mins[d] ) );
            for ( int d = 0; d < Dim; ++d )
                boxes->push_back( static_cast<double>( maxs[d] ) );
        }
    }
}

//! Classify one point against a flattened part-box list (see lbMultiJagged).
//!
//! A point inside a box takes that box's part; a point inside several (it lies
//! exactly on a cut) takes the LOWEST part id; a point inside none — MJ's boxes
//! only cover the sample's bounding box — takes the part of the nearest box by
//! squared distance, lowest part id on a tie. Pure local arithmetic on globally
//! agreed box coordinates, so every rank classifies identically.
template <int Dim>
int lbClassify( const std::vector<double>& boxes, const double* p )
{
    constexpr int stride = 1 + 2 * Dim;
    const std::size_t nb = boxes.size() / stride;
    int best = -1;
    double bestDist = std::numeric_limits<double>::max();
    for ( std::size_t b = 0; b < nb; ++b )
    {
        const double* rec = &boxes[b * stride];
        const int pid = static_cast<int>( rec[0] );
        const double* mins = rec + 1;
        const double* maxs = rec + 1 + Dim;
        double dist = 0.0;
        for ( int d = 0; d < Dim; ++d )
        {
            const double lo = mins[d] - p[d];
            const double hi = p[d] - maxs[d];
            const double e = lo > hi ? lo : hi;
            if ( e > 0.0 )
                dist += e * e;
        }
        if ( dist < bestDist || ( dist == bestDist && pid < best ) )
        {
            bestDist = dist;
            best = pid;
        }
    }
    return best < 0 ? 0 : best;
}

} // namespace detail

//! Compute a per-owned-face destination-rank assignment via Zoltan2 geometric
//! MultiJagged partitioning. Order matches ownedFaceCentroids/Gids/Weights.
//!
//! Collective on mesh.comm(). `mode` selects the solve strategy — see
//! LoadBalanceMode; the default Distributed gathers nothing to rank 0. Pass a
//! LoadBalanceStats to recover the instrumentation (in particular how many
//! faces rank 0 received for its solve).
template <class MeshT>
std::vector<Rank>
computeLoadBalance( MeshT& mesh, double imbalanceTolerance = 0.05,
                    LoadBalanceMode mode = LoadBalanceMode::Sampled,
                    LoadBalanceStats* stats = nullptr )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;

    MPI_Comm comm = mesh.comm();
    const int rank = mesh.rank();
    const int size = mesh.commSize();

    const int nof = static_cast<int>( mesh.numOwnedFaces() );
    const std::vector<Scalar> centroidsS = ownedFaceCentroids( mesh );
    const std::vector<double> weights = ownedFaceWeights( mesh );
    const std::vector<double> centroids( centroidsS.begin(), centroidsS.end() );

    if ( stats )
    {
        stats->mode = mode;
        stats->rootSolveFaces = 0;
        stats->sampleStride = 0;
        stats->cuts.clear();
        stats->fastPath = false;
        stats->globalFaces = globalOwnedFaces( mesh );
    }

    // Single-rank fast path: nothing to balance. Every mode takes it.
    if ( size == 1 )
    {
        if ( stats )
            stats->fastPath = true;
        return std::vector<Rank>( nof, 0 );
    }

    std::vector<Rank> dest( nof, static_cast<Rank>( rank ) );

    // ---------------------------------------------------------------------
    // Distributed: one solve over mesh.comm(). No gather, no scatter.
    // ---------------------------------------------------------------------
    if ( mode == LoadBalanceMode::Distributed )
    {
        const std::vector<GlobalId> gids = ownedFaceGids( mesh );
        std::vector<int64_t> ids( nof );
        for ( int f = 0; f < nof; ++f )
            ids[f] = static_cast<int64_t>( gids[f] );

        std::vector<int> parts;
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_LB_SOLVE );
            auto tcomm = Teuchos::rcp(
                new Teuchos::MpiComm<int>( Teuchos::opaqueWrapper( comm ) ) );
            detail::lbMultiJagged<Dim>( tcomm, nof, ids, centroids, weights,
                                        size, imbalanceTolerance, parts,
                                        nullptr );
        }
        // getPartListView() is already in this rank's owned-face order, which
        // is the order migrate() requires — nothing to scatter.
        for ( int f = 0; f < nof; ++f )
            dest[f] = static_cast<Rank>( parts[f] );
        return dest;
    }

    // ---------------------------------------------------------------------
    // Sampled: rank 0 solves an O(size) gid-order sample, broadcasts the cut
    // structure, every rank classifies its own faces locally.
    // ---------------------------------------------------------------------
    if ( mode == LoadBalanceMode::Sampled )
    {
        const std::vector<GlobalId> gids = ownedFaceGids( mesh );
        const long long globalF = globalOwnedFaces( mesh );

        // Sample selection is `gid % stride == 0` — a rule on GLOBALLY AGREED
        // gids, so the selected set is a property of the mesh and not of its
        // current partition, which is exactly the property that makes this
        // mode reproducible. Live face gids are sparse in Conforming mode, so
        // the stride derived from the gid range can undersample; halve until
        // the sample is large enough to partition (bounded, and every step
        // reads only collectives, so all ranks agree on the stride).
        GlobalId localMax = 0;
        for ( int f = 0; f < nof; ++f )
            localMax = std::max( localMax, gids[f] );
        long long maxGid = 0;
        {
            long long lm = static_cast<long long>( localMax );
            MPI_Allreduce( &lm, &maxGid, 1, MPI_LONG_LONG, MPI_MAX, comm );
        }
        const long long target = 64LL * size;
        const long long floorSample = std::min( globalF, 8LL * size );
        long long stride = std::max( 1LL, ( maxGid + 1 ) / target );
        long long sampleTotal = 0;
        std::vector<int> sel;
        for ( int it = 0; it < 64; ++it )
        {
            sel.clear();
            for ( int f = 0; f < nof; ++f )
                if ( static_cast<long long>( gids[f] ) % stride == 0 )
                    sel.push_back( f );
            long long ls = static_cast<long long>( sel.size() );
            MPI_Allreduce( &ls, &sampleTotal, 1, MPI_LONG_LONG, MPI_SUM, comm );
            if ( sampleTotal >= floorSample || stride == 1 )
                break;
            stride = std::max( 1LL, stride / 2 );
        }

        struct SampleRec
        {
            GlobalId gid;
            double c[Dim];
            double w;
        };
        std::vector<SampleRec> mine( sel.size() );
        for ( std::size_t i = 0; i < sel.size(); ++i )
        {
            const int f = sel[i];
            mine[i].gid = gids[f];
            for ( int d = 0; d < Dim; ++d )
                mine[i].c[d] =
                    centroids[static_cast<std::size_t>( f ) * Dim + d];
            mine[i].w = weights[f];
        }

        const int myCount = static_cast<int>( mine.size() );
        std::vector<int> counts( size, 0 );
        MPI_Gather( &myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm );
        std::vector<int> bcounts( size, 0 ), bdispls( size, 0 );
        int total = 0;
        if ( rank == 0 )
        {
            for ( int r = 0; r < size; ++r )
            {
                bdispls[r] = total * static_cast<int>( sizeof( SampleRec ) );
                bcounts[r] =
                    counts[r] * static_cast<int>( sizeof( SampleRec ) );
                total += counts[r];
            }
        }
        std::vector<SampleRec> all( rank == 0 ? total : 0 );
        MPI_Gatherv( mine.data(), myCount * (int)sizeof( SampleRec ), MPI_BYTE,
                     all.data(), bcounts.data(), bdispls.data(), MPI_BYTE, 0,
                     comm );

        std::vector<double> boxes;
        int nboxDoubles = 0;
        if ( rank == 0 )
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_LB_SOLVE );
            // Sort by gid so the solve input is byte-identical whatever the
            // starting partition was: the gather arrives in RANK order, and
            // which rank held which sampled face is partition-dependent.
            std::sort( all.begin(), all.end(),
                       []( const SampleRec& a, const SampleRec& b )
                       { return a.gid < b.gid; } );
            std::vector<int64_t> ids( total );
            std::vector<double> sc( static_cast<std::size_t>( total ) * Dim );
            std::vector<double> sw( total );
            for ( int i = 0; i < total; ++i )
            {
                ids[i] = static_cast<int64_t>( all[i].gid );
                for ( int d = 0; d < Dim; ++d )
                    sc[static_cast<std::size_t>( i ) * Dim + d] = all[i].c[d];
                sw[i] = all[i].w;
            }
            std::vector<int> parts;
            auto tcomm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
            detail::lbMultiJagged<Dim>( tcomm, total, ids, sc, sw, size,
                                        imbalanceTolerance, parts, &boxes );
            nboxDoubles = static_cast<int>( boxes.size() );
        }
        MPI_Bcast( &nboxDoubles, 1, MPI_INT, 0, comm );
        boxes.resize( nboxDoubles );
        MPI_Bcast( boxes.data(), nboxDoubles, MPI_DOUBLE, 0, comm );

        for ( int f = 0; f < nof; ++f )
            dest[f] = static_cast<Rank>( detail::lbClassify<Dim>(
                boxes, &centroids[static_cast<std::size_t>( f ) * Dim] ) );

        if ( stats )
        {
            stats->rootSolveFaces = sampleTotal;
            stats->sampleStride = stride;
            stats->cuts = boxes;
        }
        return dest;
    }

    // ---------------------------------------------------------------------
    // GatherRoot: the reference implementation. Gather every rank's owned-face
    // count, then its centroids/weights, to rank 0; solve there over a
    // SerialComm; MPI_Scatterv the per-face part assignment back. Order is
    // preserved: rank r's entries occupy [displs[r], displs[r] + counts[r]).
    // Rank 0's memory and solve time scale with the GLOBAL face count.
    // ---------------------------------------------------------------------
    std::vector<int> counts( size, 0 );
    MPI_Gather( &nof, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm );

    std::vector<int> displs( size + 1, 0 );
    int total = 0;
    if ( rank == 0 )
    {
        for ( int r = 0; r < size; ++r )
        {
            displs[r] = total;
            total += counts[r];
        }
        displs[size] = total;
    }

    std::vector<double> gCentroids( static_cast<std::size_t>( total ) * Dim );
    std::vector<double> gWeights( total );
    {
        // Centroid gather is row-major [f*Dim+d]; scale counts/displs by Dim.
        std::vector<int> cCounts( size ), cDispls( size );
        for ( int r = 0; r < size; ++r )
        {
            cCounts[r] = counts[r] * Dim;
            cDispls[r] = displs[r] * Dim;
        }
        MPI_Gatherv( centroids.data(), nof * Dim, MPI_DOUBLE, gCentroids.data(),
                     cCounts.data(), cDispls.data(), MPI_DOUBLE, 0, comm );
    }
    MPI_Gatherv( weights.data(), nof, MPI_DOUBLE, gWeights.data(),
                 counts.data(), displs.data(), MPI_DOUBLE, 0, comm );

    std::vector<int> gParts( total, 0 );
    if ( rank == 0 )
    {
        TESSERA_SCOPED_TIMER_DETAILED( ::Tessera::Profiling::TIMER_LB_SOLVE );
        // Synthesized 0..total-1 ids: the real face gids are not gathered on
        // this path (Distributed uses them directly instead).
        std::vector<int64_t> ids( total );
        for ( int i = 0; i < total; ++i )
            ids[i] = static_cast<int64_t>( i );
        auto tcomm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
        detail::lbMultiJagged<Dim>( tcomm, total, ids, gCentroids, gWeights,
                                    size, imbalanceTolerance, gParts, nullptr );
    }
    if ( stats && rank == 0 )
        stats->rootSolveFaces = total;
    if ( stats )
        MPI_Bcast( &stats->rootSolveFaces, 1, MPI_LONG_LONG, 0, comm );

    std::vector<int> myParts( nof );
    MPI_Scatterv( gParts.data(), counts.data(), displs.data(), MPI_INT,
                  myParts.data(), nof, MPI_INT, 0, comm );

    for ( int f = 0; f < nof; ++f )
        dest[f] = static_cast<Rank>( myParts[f] );
    return dest;
}

//! Internal load-balance: compute a Zoltan2 geometric assignment and migrate
//! to it. Thin wrapper over migrate() (Step 7) — no separate migration path.
//!
//! Returns migrate()'s MigrateStats. On a RefinementMode::Conforming mesh that
//! is the interesting output: Zoltan2 partitions by face centroid and closure
//! siblings have different centroids, so a nonzero `siblingFixups` is expected
//! here, not a defect. The per-parent weighting in ownedFaceWeights() is what
//! keeps the resulting perturbation load-neutral.
template <class MeshT>
MigrateStats loadBalance( MeshT& mesh,
                          MeshHalo<typename MeshT::memory_space>& halo,
                          double imbalanceTolerance = 0.05,
                          LoadBalanceMode mode = LoadBalanceMode::Sampled,
                          LoadBalanceStats* stats = nullptr )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_LOAD_BALANCE );
    const std::vector<Rank> dest =
        computeLoadBalance( mesh, imbalanceTolerance, mode, stats );
    return migrate( mesh, halo, dest );
}

} // namespace Tessera

#endif // TESSERA_ZOLTAN2_BALANCER_HPP
