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
#include "Tessera_Types.hpp"

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>

#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Tpetra_Map.hpp>

#include <mpi.h>

#include <vector>

namespace Tessera
{

// ============================================================================
// Internal Zoltan2 geometric load balancer (Step 7b)
// ============================================================================
//
// computeLoadBalance() gathers the owned-face centroids/weights/gids of every
// rank to rank 0 (this mesh is NOT replicated the way Canopy's tree is, so the
// geometric input must be assembled before Zoltan2 can see it), runs Zoltan2
// MultiJagged on rank 0 only, and scatters the resulting part assignment back
// to each rank in the same per-rank order its ownedFaceCentroids/Gids/Weights
// were gathered in. The returned `dest` is ready to hand to migrate() exactly
// as an externally-computed assignment would be — loadBalance() is a thin
// wrapper (below) composing the two.
//
// Never "rcb": Zoltan2's deterministic RCB algorithm breaks on Tuolumne (see
// Canopy_TreePartitioner.hpp). MultiJagged is used instead, but it is not
// guaranteed deterministic across ranks, so — following the same pattern —
// only rank 0 solves (over a Teuchos::SerialComm, keeping Zoltan2's internal
// bookkeeping out of MPICH entirely) and the result is MPI_Bcast/Scatterv'd
// to every other rank.

//! Compute a per-owned-face destination-rank assignment via Zoltan2 geometric
//! MultiJagged partitioning. Order matches ownedFaceCentroids/Gids/Weights.
template <class MeshT>
std::vector<Rank> computeLoadBalance( MeshT& mesh,
                                      double imbalanceTolerance = 0.05 )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;

    MPI_Comm comm = mesh.comm();
    const int rank = mesh.rank();
    const int size = mesh.commSize();

    const int nof = static_cast<int>( mesh.numOwnedFaces() );
    const std::vector<Scalar> centroids = ownedFaceCentroids( mesh );
    const std::vector<double> weights = ownedFaceWeights( mesh );

    // Single-rank fast path: nothing to balance.
    if ( size == 1 )
        return std::vector<Rank>( nof, 0 );

    // ---- Gather every rank's owned-face count, then its centroids/weights,
    // to rank 0. Order is preserved: rank r's entries occupy
    // [displs[r], displs[r] + counts[r]) in the gathered arrays. -----------
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
        std::vector<double> sendCentroids( centroids.begin(), centroids.end() );
        MPI_Gatherv( sendCentroids.data(), nof * Dim, MPI_DOUBLE,
                     gCentroids.data(), cCounts.data(), cDispls.data(),
                     MPI_DOUBLE, 0, comm );
    }
    MPI_Gatherv( weights.data(), nof, MPI_DOUBLE, gWeights.data(),
                 counts.data(), displs.data(), MPI_DOUBLE, 0, comm );

    // ---- Solve on rank 0 only (MultiJagged is not guaranteed deterministic
    // across ranks), then scatter the per-face part assignment back. --------
    std::vector<int> gParts( total, 0 );
    if ( rank == 0 )
    {
        TESSERA_SCOPED_TIMER_DETAILED( ::Tessera::Profiling::TIMER_LB_SOLVE );
        using adapter_t =
            Zoltan2::BasicVectorAdapter<Tpetra::Map<int, int64_t>>;
        using gno_t = typename adapter_t::gno_t;
        using scalar_t = typename adapter_t::scalar_t;
        using lno_t = typename adapter_t::lno_t;

        std::vector<gno_t> ids( total );
        for ( int i = 0; i < total; ++i )
            ids[i] = static_cast<gno_t>( i );

        // Deinterleave the row-major [f*Dim+d] centroids into Dim contiguous
        // per-dimension arrays, as the multivector adapter constructor needs.
        std::vector<std::vector<scalar_t>> dimVals(
            Dim, std::vector<scalar_t>( total ) );
        for ( int i = 0; i < total; ++i )
            for ( int d = 0; d < Dim; ++d )
                dimVals[d][i] = static_cast<scalar_t>(
                    gCentroids[static_cast<std::size_t>( i ) * Dim + d] );

        std::vector<const scalar_t*> entries( Dim );
        std::vector<int> entryStride( Dim, 1 );
        for ( int d = 0; d < Dim; ++d )
            entries[d] = dimVals[d].data();

        std::vector<const scalar_t*> wvals = { gWeights.data() };
        std::vector<int> wstride = { 1 };

        adapter_t adapter( static_cast<lno_t>( total ), ids.data(), entries,
                           entryStride, wvals, wstride );

        Teuchos::ParameterList params;
        params.set( "algorithm", "multijagged" );
        params.set( "num_global_parts", size );
        params.set( "imbalance_tolerance", 1.0 + imbalanceTolerance );
        params.set( "debug_level", "no_status" );
        Teuchos::ParameterList zoltanParams;
        zoltanParams.set( "DEBUG_LEVEL", "0" );
        params.set( "zoltan_parameters", zoltanParams );

        auto teuchosComm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
        Zoltan2::PartitioningProblem<adapter_t> problem( &adapter, &params,
                                                         teuchosComm );
        problem.solve();
        const auto& solution = problem.getSolution();
        const int* partsView = solution.getPartListView();
        for ( int i = 0; i < total; ++i )
            gParts[i] = partsView[i];
    }

    std::vector<int> myParts( nof );
    MPI_Scatterv( gParts.data(), counts.data(), displs.data(), MPI_INT,
                  myParts.data(), nof, MPI_INT, 0, comm );

    std::vector<Rank> dest( nof );
    for ( int i = 0; i < nof; ++i )
        dest[i] = static_cast<Rank>( myParts[i] );
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
                          double imbalanceTolerance = 0.05 )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_LOAD_BALANCE );
    const std::vector<Rank> dest =
        computeLoadBalance( mesh, imbalanceTolerance );
    return migrate( mesh, halo, dest );
}

} // namespace Tessera

#endif // TESSERA_ZOLTAN2_BALANCER_HPP
