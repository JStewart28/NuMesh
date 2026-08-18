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

// Unit test: structured-key (EdgeKey/FaceKey) determinism.
//
// The cross-rank identity of edges/faces is a structured key that MUST be
// independent of the order its constituent vertex gids are supplied — that is
// what lets every rank compute the same key for a shared entity with no
// communication. This test verifies order-invariance and canonical (sorted)
// form on both the host (Serial) and device (default, HIP on Tuolumne)
// execution spaces.

#include <Tessera_Types.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Tessera;

// Returns the number of failures detected running the checks on `Space`.
template <class Space>
int run_checks_on( const char* space_name )
{
    using memory_space = typename Space::memory_space;
    const int N = 20000;

    Kokkos::View<int*, memory_space> fails( "fails", 1 );
    Kokkos::deep_copy( fails, 0 );

    Kokkos::parallel_for(
        "key_determinism", Kokkos::RangePolicy<Space>( 0, N ),
        KOKKOS_LAMBDA( const int i ) {
            // Spread three pseudo-ids across a modest range so collisions
            // (equal ids) also occur and are handled.
            const GlobalId a =
                static_cast<GlobalId>( ( 2654435761u * (unsigned)i ) % 1000u );
            const GlobalId b =
                static_cast<GlobalId>( ( 40503u * (unsigned)i + 7u ) % 1000u );
            const GlobalId c =
                static_cast<GlobalId>( ( 2246822519u * (unsigned)i ) % 1000u );

            // Edge key: order-invariant and sorted ascending.
            const EdgeKey e1 = makeEdgeKey( a, b );
            const EdgeKey e2 = makeEdgeKey( b, a );
            if ( e1 != e2 )
                Kokkos::atomic_add( &fails( 0 ), 1 );
            if ( !( e1.id[0] <= e1.id[1] ) )
                Kokkos::atomic_add( &fails( 0 ), 1 );

            // Face key: all 6 permutations equal, and sorted ascending.
            const FaceKey f0 = makeFaceKey( a, b, c );
            const FaceKey fp[5] = {
                makeFaceKey( a, c, b ), makeFaceKey( b, a, c ),
                makeFaceKey( b, c, a ), makeFaceKey( c, a, b ),
                makeFaceKey( c, b, a ) };
            for ( int p = 0; p < 5; ++p )
                if ( fp[p] != f0 )
                    Kokkos::atomic_add( &fails( 0 ), 1 );
            if ( !( f0.id[0] <= f0.id[1] && f0.id[1] <= f0.id[2] ) )
                Kokkos::atomic_add( &fails( 0 ), 1 );
        } );
    Kokkos::fence();

    auto h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), fails );
    if ( h( 0 ) != 0 )
        std::printf( "  [%s] FAIL: %d mismatches\n", space_name, h( 0 ) );
    else
        std::printf( "  [%s] ok\n", space_name );
    return h( 0 );
}

// Host-side sanity of equality / ordering operators on the key type itself.
int host_scalar_checks()
{
    int fails = 0;
    const EdgeKey e = makeEdgeKey( 7, 3 );
    if ( !( e.id[0] == 3 && e.id[1] == 7 ) )
        ++fails;
    if ( !( makeEdgeKey( 5, 5 ).id[0] == 5 ) )
        ++fails; // degenerate equal ids
    if ( !( makeEdgeKey( 1, 2 ) < makeEdgeKey( 1, 3 ) ) )
        ++fails;
    if ( makeFaceKey( 9, 1, 5 ) != makeFaceKey( 1, 5, 9 ) )
        ++fails;
    if ( fails )
        std::printf( "  [host-scalar] FAIL: %d\n", fails );
    else
        std::printf( "  [host-scalar] ok\n" );
    return fails;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        if ( rank == 0 )
            std::printf( "test_keys: structured-key determinism\n" );

        fails += host_scalar_checks();
        fails += run_checks_on<Kokkos::Serial>( "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run_checks_on<Kokkos::DefaultExecutionSpace>( "Default" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
