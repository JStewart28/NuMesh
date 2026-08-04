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

// Regression test: mesh migration + general (non-replicated) halo rebuild
// (Step 7). Builds a replicated icosphere, distributes it (Step 5), then applies
// externally-computed face assignments through migrate() and checks, after each
// move, that the mesh is a correctly distributed, haloed mesh:
//   - ownership is a partition of every entity kind (Σ owned == global; no gid
//     owned twice) under the recomputed lowest-rank rule,
//   - every owned vertex holds its full 1-ring locally (the general ghost rebuild
//     deferred from Step 6b),
//   - the rank-count-independent topology checksum is UNCHANGED from before
//     migration (migration moves entities, it does not alter the global mesh),
//   - a halo exchange fills every ghost from its owner (corrupt -> sync -> verify).
// Two assignments exercise the external path:
//   A  a deterministic per-gid hash re-partition (crosses all boundaries); every
//      owned face must have landed on the rank the hash requested.
//   B  a global rotation dest=(rank+1)%size applied to the result of A (a synthetic
//      Canopy-style whole-partition reassignment), proving migrate() composes and
//      preserves topology.
// Runs on host (Serial) and device (default, HIP), np1-5.

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

// Deterministic per-face-gid destination rank (a pure function so every rank
// agrees and the landing rank is checkable after the move).
static inline int hashOwner( GlobalId g, int size )
{
    return static_cast<int>( ( g * 2654435761ULL ) %
                             static_cast<GlobalId>( size ) );
}

// Corrupt every ghost tuple, halo-sync, verify restoration (as in test_distribute).
template <class AoSoAType, class Mem>
int corrupt_sync_verify( MPI_Comm comm, int rank, AoSoAType& a,
                         std::size_t n_owned, HaloExchangePlan<Mem>& plan )
{
    using exec = typename AoSoAType::execution_space;
    const int n = static_cast<int>( a.size() );
    const int no = static_cast<int>( n_owned );

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

    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h1( "h1",
                                                                           n );
    Cabana::deep_copy( h1, a );
    auto g1 = Cabana::slice<0>( h1 );
    auto o1 = Cabana::slice<1>( h1 );
    int fails = 0;
    for ( int i = 0; i < no; ++i )
        if ( o1( i ) != rank )
            ++fails;
    for ( int i = no; i < n; ++i )
    {
        if ( g1( i ) != expg[i - no] || o1( i ) != expo[i - no] )
            ++fails;
        if ( o1( i ) == -1 || g1( i ) == ~static_cast<GlobalId>( 0 ) )
            ++fails;
    }
    return fails;
}

// Full invariant sweep on a distributed, haloed mesh.
template <class MeshT, class Mem>
int checkDistributed( MeshT& mesh, MeshHalo<Mem>& halo, int rank, long long NvG,
                      long long NeG, long long NfG, unsigned long long pcv,
                      unsigned long long pce, unsigned long long pcf )
{
    int fails = 0;
    fails += TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    fails += TesseraTest::owned1RingLocal( mesh );

    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    if ( cv != pcv || ce != pce || cf != pcf )
        ++fails; // migration must not change the global entity set

    fails += corrupt_sync_verify( mesh.comm(), rank, mesh.vertices(),
                                  mesh.numOwnedVertices(), halo.vplan );
    fails += corrupt_sync_verify( mesh.comm(), rank, mesh.edges(),
                                  mesh.numOwnedEdges(), halo.eplan );
    fails += corrupt_sync_verify( mesh.comm(), rank, mesh.faces(),
                                  mesh.numOwnedFaces(), halo.fplan );
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // Pinned to the hanging-node mode: the refine() calls below produce a
    // hanging-node mesh, which is no longer the Mesh default. The conforming
    // counterpart is the conforming_migrate test.
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

    // Reference checksum of the correctly distributed mesh (rank-count
    // independent). Every migration below must reproduce it exactly.
    unsigned long long pcv, pce, pcf;
    TesseraTest::topologyChecksum( mesh, pcv, pce, pcf );

    int fails = 0;

    // ---- Scenario A: per-gid hash re-partition (external) ------------------
    {
        auto fg = ownedFaceGids( mesh );
        std::vector<Rank> dest( fg.size() );
        for ( std::size_t f = 0; f < fg.size(); ++f )
            dest[f] = static_cast<Rank>( hashOwner( fg[f], size ) );
        migrate( mesh, halo, dest );

        int local = 0;
        auto og = ownedFaceGids( mesh );
        for ( GlobalId g : og )
            if ( hashOwner( g, size ) != rank )
                ++local; // face did not land on the requested rank
        int landed = 0;
        MPI_Allreduce( &local, &landed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

        int f2 =
            checkDistributed( mesh, halo, rank, NvG, NeG, NfG, pcv, pce, pcf );
        int g2 = 0;
        MPI_Allreduce( &f2, &g2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( landed != 0 || g2 != 0 )
            ++fails;
        if ( rank == 0 )
            std::printf(
                "  [%s] migrate-hash %s (landed=%d inv=%d ownedF=%zu)\n", tag,
                ( landed == 0 && g2 == 0 ) ? "ok" : "FAIL", landed, g2,
                mesh.numOwnedFaces() );
    }

    // ---- Scenario B: global rotation of the whole partition (synthetic) ----
    {
        std::vector<Rank> dest( mesh.numOwnedFaces(),
                                static_cast<Rank>( ( rank + 1 ) % size ) );
        migrate( mesh, halo, dest );

        int f2 =
            checkDistributed( mesh, halo, rank, NvG, NeG, NfG, pcv, pce, pcf );
        int g2 = 0;
        MPI_Allreduce( &f2, &g2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( g2 != 0 )
            ++fails;
        if ( rank == 0 )
            std::printf( "  [%s] migrate-rotate %s (inv=%d ownedF=%zu)\n", tag,
                         g2 == 0 ? "ok" : "FAIL", g2, mesh.numOwnedFaces() );
    }

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
            std::printf(
                "test_migrate_mesh: migration + halo rebuild (size %d)\n",
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
