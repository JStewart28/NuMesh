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

// Regression test: the global scalar collectives and the owned-count reductions.
//
// Covers globalMin / globalMax / globalSum / globalAllFinite and
// globalOwnedVertices / Edges / Faces / Euler.
//
// Every check places the extremum, or the one dissenting rank, AWAY from rank 0
// wherever the pattern allows, so a rank-0 shortcut cannot pass. Value patterns
// are closed-form in `size`, so the expectation is arithmetic rather than a
// re-derivation from Tessera.
//
// The scalar collectives are host-side MPI wrappers; the owned-count checks run
// against a real distributed icosphere, before and after a uniform refine, so
// the helpers are exercised across a topology change and a generation bump.
// Registered at both SERIAL and HIP over ranks 1-5.

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

using namespace Tessera;

// A local "is everything I hold finite" verdict, built the way a real consumer
// builds it: a sweep over the data, not a literal bool.
bool localFiniteVerdict( const std::vector<double>& data )
{
    for ( double x : data )
        if ( !std::isfinite( x ) )
            return false;
    return true;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    int fails = 0;
    auto check = [&fails]( bool ok )
    {
        if ( !ok )
            ++fails;
    };

    // A bare mesh is enough for the scalar collectives: they touch only comm().
    MeshT mesh( MPI_COMM_WORLD );

    // ---- 10. globalMin, the three pre-existing checks, verbatim ------------
    {
        // Pattern A: local = rank + 1  -> min = 1 (at rank 0).
        const double a = globalMin( mesh, static_cast<double>( rank + 1 ) );
        check( a == 1.0 );

        // Pattern B: local = size - rank -> min = 1 (at the last rank).
        const double b = globalMin( mesh, static_cast<double>( size - rank ) );
        check( b == 1.0 );

        // Integer overload: local = -rank -> min = -(size-1).
        const int c = globalMin( mesh, -rank );
        check( c == -( size - 1 ) );
    }

    // ---- 1. globalSum, exact integers --------------------------------------
    {
        // local = rank + 1 -> size*(size+1)/2.
        const int a = globalSum( mesh, rank + 1 );
        check( a == size * ( size + 1 ) / 2 );

        // local = 1 -> size.
        const int b = globalSum( mesh, 1 );
        check( b == size );

        // local = -rank -> -size*(size-1)/2. The only non-zero contributions
        // come from ranks > 0, so rank 0's term cannot carry the result.
        const int c = globalSum( mesh, -rank );
        check( c == -size * ( size - 1 ) / 2 );
    }

    // ---- 2. globalSum, long long (the case that failed before mpiTypeOf) ---
    {
        const long long kBig = 1000000007LL;
        const long long a =
            globalSum( mesh, static_cast<long long>( rank ) * kBig );
        const long long expect =
            kBig * ( static_cast<long long>( size ) *
                     static_cast<long long>( size - 1 ) / 2 );
        check( a == expect );
    }

    // ---- 3. globalSum, double (dyadic -> exact at these sizes) -------------
    {
        const double a = globalSum( mesh, 0.5 );
        check( a == 0.5 * static_cast<double>( size ) );
    }

    // ---- 4. globalMax ------------------------------------------------------
    {
        // local = rank -> size - 1, max at the LAST rank.
        const int a = globalMax( mesh, rank );
        check( a == size - 1 );

        // local = size - rank -> size, max at rank 0.
        const int b = globalMax( mesh, size - rank );
        check( b == size );

        // Double overload, same two patterns.
        const double c = globalMax( mesh, static_cast<double>( rank ) );
        check( c == static_cast<double>( size - 1 ) );
        const double d = globalMax( mesh, static_cast<double>( size - rank ) );
        check( d == static_cast<double>( size ) );
    }

    // ---- 5. globalAllFinite, positive --------------------------------------
    {
        std::vector<double> clean = { 0.0, -1.5, static_cast<double>( rank ),
                                      1e300 };
        check( globalAllFinite( mesh, localFiniteVerdict( clean ) ) == true );
    }

    // ---- 6. globalAllFinite, negative, three ways --------------------------
    // The LAST rank's local sweep sees the poison; everyone else is clean.
    // Every rank must receive false.
    {
        const double poisons[3] = { std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::infinity(),
                                    -std::numeric_limits<double>::infinity() };
        for ( int p = 0; p < 3; ++p )
        {
            std::vector<double> data = { 0.0, 1.0, 2.0, 3.0 };
            if ( rank == size - 1 )
                data[2] = poisons[p];
            const bool g = globalAllFinite( mesh, localFiniteVerdict( data ) );
            // At one rank the "last rank" IS rank 0, which is still the
            // dissenting rank, so the expectation is false at every size.
            check( g == false );
        }
    }

    // ---- 7. globalAllFinite at one rank returns the local verdict ----------
    if ( size == 1 )
    {
        std::vector<double> clean = { 1.0, 2.0 };
        std::vector<double> bad = { 1.0,
                                    std::numeric_limits<double>::quiet_NaN() };
        check( globalAllFinite( mesh, localFiniteVerdict( clean ) ) == true );
        check( globalAllFinite( mesh, localFiniteVerdict( bad ) ) == false );
    }

    // ---- 8. Owned-count reductions on a real distributed mesh --------------
    // Icosphere subdivision 2: V=162 E=480 F=320. Closed form, not Tessera's
    // word: V - E + F = 2 and 3F = 2E for a closed triangle mesh.
    const long long V0 = 162, E0 = 480, F0 = 320;
    {
        MeshT m( MPI_COMM_WORLD );
        buildIcosphere( m, 2 );
        auto faceOwner = facePartitionByAxis( m );
        MeshHalo<mem> halo;
        distribute( m, halo, faceOwner );

        const long long gV = globalOwnedVertices( m );
        const long long gE = globalOwnedEdges( m );
        const long long gF = globalOwnedFaces( m );
        const long long euler = globalOwnedEuler( m );

        check( gV == V0 );
        check( gE == E0 );
        check( gF == F0 );
        check( euler == 2 );

        if ( rank == 0 )
            std::printf(
                "  [%s] owned counts V=%lld E=%lld F=%lld euler=%lld\n", tag,
                gV, gE, gF, euler );
    }

    // ---- 9. Owned-count reductions after a uniform refine() ----------------
    // V' = V + E = 642, E' = 2E + 3F = 1920, F' = 4F = 1280, Euler still 2.
    {
        MeshT m( MPI_COMM_WORLD );
        buildIcosphere( m, 2 );
        auto faceOwner = facePartitionByAxis( m );
        MeshHalo<mem> halo;
        distribute( m, halo, faceOwner );

        std::vector<char> mask( m.numOwnedFaces(), 1 );
        refine( m, halo, mask );

        const long long gV = globalOwnedVertices( m );
        const long long gE = globalOwnedEdges( m );
        const long long gF = globalOwnedFaces( m );
        const long long euler = globalOwnedEuler( m );

        check( gV == V0 + E0 );
        check( gE == 2 * E0 + 3 * F0 );
        check( gF == 4 * F0 );
        check( euler == 2 );

        if ( rank == 0 )
            std::printf(
                "  [%s] refined counts V=%lld E=%lld F=%lld euler=%lld\n", tag,
                gV, gE, gF, euler );
    }

    if ( rank == 0 )
        std::printf( "  [%s] %s\n", tag, fails == 0 ? "ok" : "FAIL" );
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
            std::printf( "test_global_reduce (size=%d)\n", size );

        fails += run<Kokkos::Serial>( rank, size, "serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( rank, size, "device" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    if ( global_fails == 0 )
    {
        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        if ( rank == 0 )
            std::printf( "test_global_reduce: ok\n" );
    }
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
