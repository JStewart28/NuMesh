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

// Unit test: whole-tuple AoSoA migration (Step 4a).
//
// Exercises Tessera::migrate() over a payload AoSoA at 1-5 MPI ranks on both the
// host (Serial) and device (default, HIP on Tuolumne) spaces. Each element
// carries a globally-unique id, its source rank, and a payload derived from the
// id. Three destination patterns are checked:
//   - scatter : dest = gid % size (balanced redistribution)
//   - shift   : dest = (rank+1) % size (every element leaves; num_kept == 0)
//   - identity: dest = rank (nothing moves; fast path, migrate returns 0)
// After each migrate the test verifies global conservation (no element lost or
// duplicated), that every element landed on the rank its destination rule names,
// and that its payload survived intact.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <set>
#include <type_traits>
#include <vector>

using Members = Cabana::MemberTypes<Tessera::GlobalId, int, double[3]>;
enum
{
    SGID = 0,
    SRANK = 1,
    PAY = 2
};

static constexpr int N_PER_RANK = 500;

// Payload value for global id g, component d.
static KOKKOS_INLINE_FUNCTION double payload_of( Tessera::GlobalId g, int d )
{
    return static_cast<double>( g ) * 8.0 + d;
}

// Fresh AoSoA of N_PER_RANK elements owned initially by `rank`.
template <class Exec>
Cabana::AoSoA<Members, typename Exec::memory_space> make_local( int rank )
{
    using mem = typename Exec::memory_space;
    Cabana::AoSoA<Members, mem> aosoa( "payload", N_PER_RANK );
    auto sgid = Cabana::slice<SGID>( aosoa );
    auto srank = Cabana::slice<SRANK>( aosoa );
    auto pay = Cabana::slice<PAY>( aosoa );
    const Tessera::GlobalId base =
        static_cast<Tessera::GlobalId>( rank ) * N_PER_RANK;
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<Exec>( 0, N_PER_RANK ),
        KOKKOS_LAMBDA( const int i ) {
            const Tessera::GlobalId g = base + i;
            sgid( i ) = g;
            srank( i ) = rank;
            for ( int d = 0; d < 3; ++d )
                pay( i, d ) = payload_of( g, d );
        } );
    Kokkos::fence();
    return aosoa;
}

// Compute a destination-rank view from a host functor dest(gid)->rank.
template <class Exec, class AoSoAType, class Fn>
Kokkos::View<int*, typename Exec::memory_space>
dest_from( const AoSoAType& aosoa, Fn fn )
{
    using mem = typename Exec::memory_space;
    const int n = static_cast<int>( aosoa.size() );
    // Read gids via a host copy of the AoSoA (a slice's data() is not a
    // contiguous array under the SoA layout, so it can't be reinterpreted).
    Cabana::AoSoA<Members, Kokkos::HostSpace> h( "h", n );
    Cabana::deep_copy( h, aosoa );
    auto h_sgid = Cabana::slice<SGID>( h );
    Kokkos::View<int*, mem> dest( "dest", n );
    auto h_dest = Kokkos::create_mirror_view( dest );
    for ( int i = 0; i < n; ++i )
        h_dest( i ) = fn( h_sgid( i ) );
    Kokkos::deep_copy( dest, h_dest );
    return dest;
}

// Verify: local count == expected; every element's dest-rule maps to this rank;
// payload intact; and global count conserved. Returns failure count.
template <class AoSoAType, class Fn>
int verify( const AoSoAType& aosoa, int rank, int size, Fn dest_rule,
            const char* label )
{
    int fails = 0;
    const int n = static_cast<int>( aosoa.size() );

    // Global conservation.
    long long local = n, global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    if ( global != static_cast<long long>( N_PER_RANK ) * size )
        ++fails;

    // Copy to host and check every element.
    Cabana::AoSoA<Members, Kokkos::HostSpace> h( "h", n );
    Cabana::deep_copy( h, aosoa );
    auto sgid = Cabana::slice<SGID>( h );
    auto srank = Cabana::slice<SRANK>( h );
    auto pay = Cabana::slice<PAY>( h );

    std::set<Tessera::GlobalId> seen;
    for ( int i = 0; i < n; ++i )
    {
        const Tessera::GlobalId g = sgid( i );
        if ( dest_rule( g ) != rank ) // landed on the wrong rank
            ++fails;
        if ( srank( i ) != static_cast<int>( g / N_PER_RANK ) ) // source tag
            ++fails;
        for ( int d = 0; d < 3; ++d )
            if ( pay( i, d ) != payload_of( g, d ) ) // payload intact
                ++fails;
        if ( !seen.insert( g ).second ) // duplicate
            ++fails;
    }

    std::printf( "  [%s] %s (rank %d holds %d)\n", label,
                 fails == 0 ? "ok" : "FAIL", rank, n );
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    Tessera::MigrateBuffers<mem> bufs;
    int fails = 0;
    char label[64];

    // scatter: dest = gid % size
    {
        auto a = make_local<Exec>( rank );
        auto scatter = [size]( Tessera::GlobalId g )
        { return static_cast<int>( g % size ); };
        auto dest = dest_from<Exec>( a, scatter );
        Tessera::migrate( MPI_COMM_WORLD, a, dest, bufs );
        std::snprintf( label, sizeof( label ), "%s/scatter", tag );
        fails += verify( a, rank, size, scatter, label );
    }
    // shift: dest = (rank+1) % size for every local element
    {
        auto a = make_local<Exec>( rank );
        const int target = ( rank + 1 ) % size;
        auto dest = dest_from<Exec>( a, [target]( Tessera::GlobalId )
                                     { return target; } );
        Tessera::migrate( MPI_COMM_WORLD, a, dest, bufs );
        // after a uniform +1 shift, this rank holds what (rank-1) sent: every
        // element g with (g/N_PER_RANK) == (rank-1+size)%size.
        const int src = ( rank - 1 + size ) % size;
        auto rule = [src, size]( Tessera::GlobalId g )
        {
            return ( static_cast<int>( g / N_PER_RANK ) == src )
                       ? ( ( src + 1 ) % size )
                       : -1;
        };
        std::snprintf( label, sizeof( label ), "%s/shift", tag );
        fails += verify( a, rank, size, rule, label );
    }
    // identity: dest = rank (fast path, nothing moves)
    {
        auto a = make_local<Exec>( rank );
        auto dest =
            dest_from<Exec>( a, [rank]( Tessera::GlobalId ) { return rank; } );
        const int sent = Tessera::migrate( MPI_COMM_WORLD, a, dest, bufs );
        if ( sent != 0 || static_cast<int>( a.size() ) != N_PER_RANK )
            ++fails;
        std::snprintf( label, sizeof( label ), "%s/identity", tag );
        auto rule = [rank]( Tessera::GlobalId ) { return rank; };
        fails += verify( a, rank, size, rule, label );
    }
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
            std::printf(
                "test_migrate: whole-tuple AoSoA migration (size %d)\n", size );

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
