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

// Regression test: distributed mesh + ownership + 1-deep halo (Step 5).
//
// Builds a replicated icosphere, partitions its faces geometrically, distributes
// to a local (owned + ghost) mesh, and checks:
//   - ownership is a partition of every entity kind (sum of owned == global,
//     no gid owned twice),
//   - every owned vertex has its full 1-ring held locally,
//   - the edgeKeys()/faceKeys() side tables are sized to the LOCAL entity counts
//     and agree with the local connectivity entry for entry, with no duplicate
//     owned key globally,
//   - a halo exchange fills every ghost with its owner's data: ghost tuples are
//     corrupted, haloExchange() is run, and each ghost must be restored to the
//     gid/owner its owner holds (owned entities untouched). At one rank there are
//     no ghosts and haloExchange() is a no-op.
// Runs on host (Serial) and device (default, HIP) spaces, np1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using namespace Tessera;

// Corrupt every ghost tuple, halo-sync, and verify each ghost is restored to the
// gid/owner its owner holds; owned entities [0,n_owned) must be untouched. Uses
// member 0 == gid, member 1 == owner (true for all entity kinds). Returns LOCAL
// fails.
template <class AoSoAType, class Mem>
int corrupt_sync_verify( MPI_Comm comm, int rank, AoSoAType& a,
                         std::size_t n_owned, HaloExchangePlan<Mem>& plan )
{
    using exec = typename AoSoAType::execution_space;
    const int n = static_cast<int>( a.size() );
    const int no = static_cast<int>( n_owned );

    // Record the expected ghost gid/owner (correct just after distribute).
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h0( "h0",
                                                                           n );
    Cabana::deep_copy( h0, a );
    auto g0 = Cabana::slice<0>( h0 );
    auto o0 = Cabana::slice<1>( h0 );
    std::vector<GlobalId> expg;
    std::vector<Rank> expo;
    for ( int i = no; i < n; ++i )
    {
        expg.push_back( g0( i ) );
        expo.push_back( o0( i ) );
    }

    // Corrupt ghost slots on device.
    auto g = Cabana::slice<0>( a );
    auto o = Cabana::slice<1>( a );
    Kokkos::parallel_for(
        "corrupt", Kokkos::RangePolicy<exec>( no, n ),
        KOKKOS_LAMBDA( const int i ) {
            g( i ) = ~static_cast<GlobalId>( 0 );
            o( i ) = -1;
        } );
    Kokkos::fence();

    haloExchange( comm, a, plan );

    // Verify restoration and that owned entities were untouched.
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h1( "h1",
                                                                           n );
    Cabana::deep_copy( h1, a );
    auto g1 = Cabana::slice<0>( h1 );
    auto o1 = Cabana::slice<1>( h1 );
    int fails = 0;
    for ( int i = 0; i < no; ++i )
        if ( o1( i ) != rank )
            ++fails; // owned owner must be this rank, untouched
    for ( int i = no; i < n; ++i )
    {
        if ( g1( i ) != expg[i - no] || o1( i ) != expo[i - no] )
            ++fails;
        if ( o1( i ) == -1 || g1( i ) == ~static_cast<GlobalId>( 0 ) )
            ++fails; // still corrupt -> sync did not reach this ghost
    }
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );
    const long long NvG = static_cast<long long>( mesh.numVertices() );
    const long long NeG = static_cast<long long>( mesh.numEdges() );
    const long long NfG = static_cast<long long>( mesh.numFaces() );

    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );

    int fails = 0;
    // Immediately after distribute(), before anything else has touched the mesh:
    // the canonical-key side tables must describe THIS rank's local entities, not
    // the replicated mesh the builder produced. The extent half of this check is
    // the direct reproducer for the defect distribute() used to have.
    fails += TesseraTest::checkKeyTables( mesh );
    fails += TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += corrupt_sync_verify( MPI_COMM_WORLD, rank, mesh.vertices(),
                                  mesh.numOwnedVertices(), halo.vplan );
    fails += corrupt_sync_verify( MPI_COMM_WORLD, rank, mesh.edges(),
                                  mesh.numOwnedEdges(), halo.eplan );
    fails += corrupt_sync_verify( MPI_COMM_WORLD, rank, mesh.faces(),
                                  mesh.numOwnedFaces(), halo.fplan );

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    if ( rank == 0 )
        std::printf( "  [%s] distribute %s (localV=%zu ownedV=%zu)\n", tag,
                     global == 0 ? "ok" : "FAIL", mesh.numVertices(),
                     mesh.numOwnedVertices() );
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
            std::printf( "test_distribute: distributed mesh + halo (size %d)\n",
                         size );

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    // run() already reduced to a global count identical on every rank.
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
