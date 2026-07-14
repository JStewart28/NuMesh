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

// Unit test: global scalar reduction (globalMin).
//
// globalMin is a thin MPI_Allreduce(MPI_MIN) wrapper over mesh.comm(). Two
// per-rank value patterns place the minimum at different ranks (rank 0 and the
// last rank) so the test confirms a real collective, not a rank-0 shortcut.
// Single rank reduces to the identity. Host-only reduction; SERIAL-registered.

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>

using namespace Tessera;

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>,
                           FaceFields<>, Kokkos::HostSpace, Kokkos::Serial>;
        MeshT mesh( MPI_COMM_WORLD );
        const int rank = mesh.rank();
        const int size = mesh.commSize();

        // Pattern A: local = rank + 1  -> min = 1 (at rank 0).
        const double a = globalMin( mesh, static_cast<double>( rank + 1 ) );
        if ( a != 1.0 )
            ++fails;

        // Pattern B: local = size - rank -> min = 1 (at the last rank).
        const double b = globalMin( mesh, static_cast<double>( size - rank ) );
        if ( b != 1.0 )
            ++fails;

        // Integer overload: local = -rank -> min = -(size-1).
        const int c = globalMin( mesh, -rank );
        if ( c != -( size - 1 ) )
            ++fails;

        if ( rank == 0 )
            std::printf( "test_global_reduce: %s (size=%d)\n",
                         fails == 0 ? "ok" : "FAIL", size );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
