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

// Minimal Tessera example — verifies Kokkos initialization, MPI linkage,
// and the active execution space on each rank.
//
// Usage (no arguments):
//   Tuolumne: flux run --ntasks <N> --nodes=1 --exclusive --cores-per-task=1
//                 ./hello_tessera
//   Local:    mpirun -n <N> ./hello_tessera

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdio>

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    Kokkos::initialize( argc, argv );
    {
        if ( rank == 0 )
        {
            std::printf( "Hello from Tessera!\n" );
            std::printf( "  Kokkos execution space: %s\n",
                         Kokkos::DefaultExecutionSpace::name() );
            std::printf( "  MPI ranks: %d\n", size );
        }
        MPI_Barrier( MPI_COMM_WORLD );
        std::printf( "  rank %d of %d ready\n", rank, size );
    }
    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
