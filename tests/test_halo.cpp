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

// Unit test: halo exchange primitives (Step 4b).
//
// Part A: allToAllV — variable-length neighbour topology exchange. Rank r sends
// each destination d a list of length d; the receiver checks it gets exactly
// `myrank` items from every source with the expected encoded values.
//
// Part B: HaloExchangePlan + haloExchange — a ring halo where each rank sends its
// owned block to (rank+1) and fills its ghost block from (rank-1). Verifies every
// ghost slot receives the owner's whole tuple (all fields). At one rank the plan
// is empty (self-peer dropped) and haloExchange is a no-op — the ghosts keep
// their sentinel, exercising the MI300A self-send guard.
//
// Runs on the host (Serial) and device (default, HIP on Tuolumne) spaces, np1-5.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <type_traits>
#include <vector>

using namespace Tessera;

// ---- Part A: allToAllV ------------------------------------------------------
int test_alltoallv( int rank, int size )
{
    int fails = 0;
    // send[d] = list of length d, entries encode (source rank, position).
    std::vector<std::vector<GlobalId>> send( size );
    for ( int d = 0; d < size; ++d )
        for ( int k = 0; k < d; ++k )
            send[d].push_back( static_cast<GlobalId>( rank ) * 1000 + k );

    auto res = allToAllV( MPI_COMM_WORLD, send );

    // This rank is destination `rank`, so every source s sent us a list of
    // length `rank` with values s*1000 + k.
    for ( int s = 0; s < size; ++s )
    {
        if ( res.count( s ) != rank )
        {
            ++fails;
            continue;
        }
        const GlobalId* p = res.from( s );
        for ( int k = 0; k < rank; ++k )
            if ( p[k] != static_cast<GlobalId>( s ) * 1000 + k )
                ++fails;
    }
    if ( rank == 0 )
        std::printf( "  [alltoallv] %s\n", fails == 0 ? "ok" : "FAIL" );
    return fails;
}

// ---- Part B: ring halo ------------------------------------------------------
using HaloMembers = Cabana::MemberTypes<GlobalId, int, double>;
enum
{
    HGID = 0,
    HOWNER = 1,
    HVAL = 2
};
static constexpr int N_OWNED = 6;
static constexpr int N_GHOST = 6;

static KOKKOS_INLINE_FUNCTION double halo_payload( GlobalId g )
{
    return static_cast<double>( g ) * 3.0 + 1.0;
}

template <class Exec>
int test_ring_halo( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    const int ntot = N_OWNED + N_GHOST;
    Cabana::AoSoA<HaloMembers, mem> aosoa( "halo", ntot );
    auto gid = Cabana::slice<HGID>( aosoa );
    auto own = Cabana::slice<HOWNER>( aosoa );
    auto val = Cabana::slice<HVAL>( aosoa );

    // owned [0,N_OWNED): real data; ghost [N_OWNED,ntot): sentinel.
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<Exec>( 0, ntot ),
        KOKKOS_LAMBDA( const int i ) {
            if ( i < N_OWNED )
            {
                const GlobalId g = static_cast<GlobalId>( rank ) * 1000 + i;
                gid( i ) = g;
                own( i ) = rank;
                val( i ) = halo_payload( g );
            }
            else
            {
                gid( i ) = ~static_cast<GlobalId>( 0 );
                own( i ) = -1;
                val( i ) = -999.0;
            }
        } );
    Kokkos::fence();

    HaloExchangePlan<mem> plan;
    std::map<int, std::vector<LocalIndex>> send_by_peer, recv_by_peer;
    const int send_to = ( rank + 1 ) % size;
    const int recv_from = ( rank - 1 + size ) % size;
    for ( int j = 0; j < N_OWNED; ++j )
        send_by_peer[send_to].push_back( j );
    for ( int j = 0; j < N_GHOST; ++j )
        recv_by_peer[recv_from].push_back( N_OWNED + j );
    plan.setFromHost( rank, send_by_peer, recv_by_peer );

    haloExchange( MPI_COMM_WORLD, aosoa, plan );

    // Verify ghosts on host.
    Cabana::AoSoA<HaloMembers, Kokkos::HostSpace> h( "h", ntot );
    Cabana::deep_copy( h, aosoa );
    auto hgid = Cabana::slice<HGID>( h );
    auto hown = Cabana::slice<HOWNER>( h );
    auto hval = Cabana::slice<HVAL>( h );

    int fails = 0;
    if ( size == 1 )
    {
        // Empty plan: ghosts keep their sentinel (no-op exchange).
        for ( int j = 0; j < N_GHOST; ++j )
            if ( hown( N_OWNED + j ) != -1 )
                ++fails;
    }
    else
    {
        for ( int j = 0; j < N_GHOST; ++j )
        {
            const GlobalId expg = static_cast<GlobalId>( recv_from ) * 1000 + j;
            if ( hgid( N_OWNED + j ) != expg ||
                 hown( N_OWNED + j ) != recv_from ||
                 hval( N_OWNED + j ) != halo_payload( expg ) )
                ++fails;
        }
    }
    // Owned entities must be untouched by the exchange.
    for ( int j = 0; j < N_OWNED; ++j )
        if ( hown( j ) != rank )
            ++fails;

    std::printf( "  [%s] ring halo %s\n", tag, fails == 0 ? "ok" : "FAIL" );
    return fails;
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
            std::printf( "test_halo: allToAllV + HaloExchangePlan (size %d)\n",
                         size );

        fails += test_alltoallv( rank, size );
        fails += test_ring_halo<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += test_ring_halo<Kokkos::DefaultExecutionSpace>( rank, size,
                                                                    "Default" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
