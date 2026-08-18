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

// Regression test: internal Zoltan2 loadBalance() (Step 7b). Builds a
// replicated icosphere, distributes it (Step 5) to a balanced partition, takes
// its rank-count-independent topology checksum, then deliberately dumps every
// owned face onto rank 0 (migrate() with dest=0 everywhere) to create a
// maximally imbalanced mesh. Calling loadBalance() must then:
//   - reduce the max-owned-face imbalance across ranks (the whole point of the
//     internal path),
//   - preserve every distribution invariant migrate() already guarantees
//     (ownership partition, owned 1-ring locality),
//   - preserve the topology checksum (loadBalance() moves entities, it does
//     not alter the global mesh — same contract as migrate()).
// Single rank is a checked no-op (nothing to balance). Runs on host (Serial)
// and device (default, HIP), np1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using namespace Tessera;

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // Pinned to the hanging-node mode: the refine() call below produces a
    // hanging-node mesh, which is no longer the Mesh default. The conforming
    // counterpart is the conforming_migrate test's loadBalance case.
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );
    const long long NvG = static_cast<long long>( mesh.numVertices() );
    const long long NeG = static_cast<long long>( mesh.numEdges() );
    const long long NfG = static_cast<long long>( mesh.numFaces() );

    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );

    unsigned long long pcv, pce, pcf;
    TesseraTest::topologyChecksum( mesh, pcv, pce, pcf );

    // Deliberately imbalance: dump every owned face onto rank 0.
    {
        std::vector<Rank> dest( mesh.numOwnedFaces(), static_cast<Rank>( 0 ) );
        migrate( mesh, halo, dest );
    }
    const long long before = static_cast<long long>( mesh.numOwnedFaces() );
    long long maxBefore = 0;
    MPI_Allreduce( &before, &maxBefore, 1, MPI_LONG_LONG, MPI_MAX,
                   MPI_COMM_WORLD );

    loadBalance( mesh, halo );

    const long long after = static_cast<long long>( mesh.numOwnedFaces() );
    long long maxAfter = 0;
    MPI_Allreduce( &after, &maxAfter, 1, MPI_LONG_LONG, MPI_MAX,
                   MPI_COMM_WORLD );

    int fails = 0;
    // Single-rank: nothing to balance, loadBalance() is a no-op.
    if ( size == 1 )
    {
        if ( maxAfter != NfG )
            ++fails;
    }
    else
    {
        // Balance must improve: the maximum owned-face count across ranks
        // must drop well below "everything on one rank", and land within a
        // generous tolerance of the ideal even split (MultiJagged's own
        // imbalance_tolerance plus geometric slack on a coarse icosphere).
        const long long ideal = ( NfG + size - 1 ) / size;
        if ( maxAfter >= maxBefore )
            ++fails; // no improvement over the maximally imbalanced start
        if ( maxAfter > 2 * ideal )
            ++fails; // grossly unbalanced result
    }

    int f2 = TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    f2 += TesseraTest::owned1RingLocal( mesh );

    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    if ( cv != pcv || ce != pce || cf != pcf )
        ++f2; // loadBalance() must not change the global entity set

    int g2 = 0;
    MPI_Allreduce( &f2, &g2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    fails += g2;

    if ( rank == 0 )
        std::printf( "  [%s] loadBalance %s (maxBefore=%lld maxAfter=%lld "
                     "ideal=%lld inv=%d)\n",
                     tag, fails == 0 ? "ok" : "FAIL", maxBefore, maxAfter,
                     ( NfG + size - 1 ) / size, g2 );

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return global;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0, size = 1;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        if ( rank == 0 )
            std::printf( "test_loadbalance: internal Zoltan2 loadBalance "
                         "(size %d)\n",
                         size );

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
