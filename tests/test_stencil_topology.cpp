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

// Unit test: k-ring vertex stencil topology (buildVertexStencil).
//
// On a single-rank icosphere (gid == local index), the k=1 stencil row of every
// vertex must equal the vertex->vertex 1-ring derived independently from the
// edge list, and the k=2 row must (a) be a strict superset of the k=1 row,
// (b) contain exactly the vertices at graph distance 1 or 2, and (c) exclude the
// source vertex itself. Runs on Serial and the default (HIP) space.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <type_traits>
#include <vector>

using namespace Tessera;

template <class Scalar, class Exec>
int run( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    int fails = 0;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    const int nv = static_cast<int>( mesh.numVertices() );

    // Independent 1-ring adjacency from the edge list (gid == local here).
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto e_v = Cabana::slice<EdgeField::Verts>( he );
    std::vector<std::set<int>> onering( nv );
    for ( std::size_t e = 0; e < mesh.numEdges(); ++e )
    {
        const int a = static_cast<int>( e_v( e, 0 ) );
        const int b = static_cast<int>( e_v( e, 1 ) );
        onering[a].insert( b );
        onering[b].insert( a );
    }

    auto rows_of = [&]( const VertexStencil<mem>& st )
    {
        const auto& csr = st.csr.get();
        auto off = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        csr.offsets );
        auto nbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        csr.neighbors );
        std::vector<std::set<int>> rows( nv );
        for ( int v = 0; v < nv; ++v )
            for ( int p = off( v ); p < off( v + 1 ); ++p )
                rows[v].insert( static_cast<int>( nbr( p ) ) );
        return rows;
    };

    auto st1 = buildVertexStencil( mesh, 1 );
    auto st2 = buildVertexStencil( mesh, 2 );
    auto rows1 = rows_of( st1 );
    auto rows2 = rows_of( st2 );

    for ( int v = 0; v < nv; ++v )
    {
        // k=1 row == independent 1-ring, self excluded.
        if ( rows1[v] != onering[v] || rows1[v].count( v ) )
            ++fails;

        // reference 2-ring (distance 1 or 2), self excluded.
        std::set<int> ref2 = onering[v];
        for ( int x : onering[v] )
            for ( int y : onering[x] )
                if ( y != v )
                    ref2.insert( y );

        if ( rows2[v] != ref2 || rows2[v].count( v ) )
            ++fails;
        // k=2 superset of k=1.
        for ( int w : rows1[v] )
            if ( !rows2[v].count( w ) )
                ++fails;
    }

    if ( st1.k != 1 || st2.k != 2 )
        ++fails;

    std::printf( "  [%s] %s (nv=%d)\n", tag, fails == 0 ? "ok" : "FAIL", nv );
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
            std::printf( "test_stencil_topology: k-ring vertex stencil\n" );

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
