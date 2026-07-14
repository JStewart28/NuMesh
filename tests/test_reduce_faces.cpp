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

// Unit test: face -> vertex reduction (reduceVertexFromFaces).
//
// A caller-supplied "one-third area" op accumulates (1/3) * faceArea of every
// incident face into a per-vertex area field. Because each vertex is owned by
// exactly one rank, the GLOBAL sum over owned vertices of this vertex area
// equals (1/3) * sum over all (vertex, incident-face) pairs of faceArea =
// sum over unique faces of faceArea = the total surface area. That total is
// checked independently against the global sum of faceArea over owned faces —
// a partition-independent identity. Runs over 1-5 ranks; Serial and default.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Tessera;

// Caller-owned device functor: vertexArea(v) += 1/3 * faceArea(f).
template <class MeshT>
struct ThirdAreaOp
{
    template <class FaceSlice, class VertSlice>
    KOKKOS_INLINE_FUNCTION void operator()( int v, int f,
                                            const MeshGeometry<MeshT>& g,
                                            FaceSlice, VertSlice vs ) const
    {
        using Scalar = typename MeshT::scalar_type;
        vs( v ) += faceArea( g, f ) / Scalar( 3 );
    }
};

template <class Scalar, class Exec>
int run( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<Scalar, 3, VertexFields<Scalar>, EdgeFields<>,
                       FaceFields<>, mem, Exec>;
    constexpr std::size_t VAREA = userVertexField<0>();
    int fails = 0;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );
    MeshHalo<mem> halo;
    auto owner = facePartitionByAxis( mesh );
    distribute( mesh, halo, owner );
    haloExchange( mesh, halo );

    auto geom = buildMeshGeometry( mesh );
    const int n_owned_v = static_cast<int>( mesh.numOwnedVertices() );
    const int n_owned_f = static_cast<int>( mesh.numOwnedFaces() );

    // zero the vertex-area field, then reduce faces -> vertices.
    {
        auto va = mesh.template vertexSlice<VAREA>();
        Kokkos::parallel_for(
            "zero_varea",
            Kokkos::RangePolicy<Exec>( 0,
                                       static_cast<int>( mesh.numVertices() ) ),
            KOKKOS_LAMBDA( const int i ) { va( i ) = Scalar( 0 ); } );
        Kokkos::fence();
    }
    reduceVertexFromFaces(
        mesh, geom, mesh.template faceSlice<FaceField::Gid>(),
        mesh.template vertexSlice<VAREA>(), ThirdAreaOp<MeshT>{} );

    // sum vertex areas over owned vertices.
    Scalar local_varea = 0;
    {
        auto va = mesh.template vertexSlice<VAREA>();
        Kokkos::parallel_reduce(
            "sum_varea", Kokkos::RangePolicy<Exec>( 0, n_owned_v ),
            KOKKOS_LAMBDA( const int v, Scalar& acc ) { acc += va( v ); },
            local_varea );
    }

    // reference: sum faceArea over owned faces.
    Scalar local_farea = 0;
    Kokkos::parallel_reduce(
        "sum_farea", Kokkos::RangePolicy<Exec>( 0, n_owned_f ),
        KOKKOS_LAMBDA( const int f, Scalar& acc ) {
            acc += faceArea( geom, f );
        },
        local_farea );

    double g_varea = 0, g_farea = 0;
    double d_varea = local_varea, d_farea = local_farea;
    MPI_Allreduce( &d_varea, &g_varea, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( &d_farea, &g_farea, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    if ( std::abs( g_varea - g_farea ) > 1e-9 * g_farea )
        ++fails;

    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
        std::printf( "  [%s] %s (vertexArea sum=%.9f, faceArea sum=%.9f)\n",
                     tag, fails == 0 ? "ok" : "FAIL", g_varea, g_farea );
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
            std::printf( "test_reduce_faces: face -> vertex reduction\n" );

        fails += run<double, Kokkos::Serial>( "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<double, Kokkos::DefaultExecutionSpace>( "Default" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
