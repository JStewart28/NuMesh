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

// Unit test: raw geometric primitives (faceArea, faceNormalRaw, edgeVector,
// cotangentAtCorner) on an analytic mesh with hand-computed answers.
//
// The mesh is the unit square [0,1]^2 in the z=0 plane split into two triangles
// (0,1,2) and (0,2,3), for which every primitive has an exact value:
//   - each face area = 1/2, total = 1
//   - each face raw normal = (0,0,1) (magnitude = 2*area = 1)
//   - cotangents of the right-angle corners = 0, of the 45-degree corners = 1
//   - edge vectors equal the endpoint position differences
// Results are computed on the device and checked on the host, so the same
// coverage runs for Serial and the default (HIP) execution space.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Tessera;

template <class Scalar, class Exec>
int run( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    int fails = 0;

    // Unit square split into two triangles, in the z=0 plane.
    TriangleSoup<Scalar> soup;
    soup.positions = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0 };
    soup.triangles = { 0, 1, 2, 0, 2, 3 };

    MeshT mesh( MPI_COMM_WORLD );
    buildFromTriangleSoup( mesh, soup );
    auto geom = buildMeshGeometry( mesh );

    const int nf = static_cast<int>( mesh.numFaces() );
    const int ne = static_cast<int>( mesh.numEdges() );

    // ---- per-face area, raw normal, and the three corner cotangents ---------
    Kokkos::View<Scalar*, mem> area( "area", nf );
    Kokkos::View<Scalar* [3], mem> nrm( "nrm", nf );
    Kokkos::View<Scalar* [3], mem> cot( "cot", nf );
    Kokkos::parallel_for(
        "geom_faces", Kokkos::RangePolicy<Exec>( 0, nf ),
        KOKKOS_LAMBDA( const int f ) {
            area( f ) = faceArea( geom, f );
            Scalar n[3];
            faceNormalRaw( geom, f, n );
            for ( int d = 0; d < 3; ++d )
                nrm( f, d ) = n[d];
            for ( int c = 0; c < 3; ++c )
                cot( f, c ) = cotangentAtCorner( geom, f, c );
        } );
    Kokkos::fence();

    // ---- per-edge vector ----------------------------------------------------
    Kokkos::View<Scalar* [3], mem> evec( "evec", ne );
    Kokkos::parallel_for(
        "geom_edges", Kokkos::RangePolicy<Exec>( 0, ne ),
        KOKKOS_LAMBDA( const int e ) {
            Scalar v[3];
            edgeVector( geom, e, v );
            for ( int d = 0; d < 3; ++d )
                evec( e, d ) = v[d];
        } );
    Kokkos::fence();

    auto h_area =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), area );
    auto h_nrm =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), nrm );
    auto h_cot =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), cot );
    auto h_evec =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), evec );

    const double tol = 1e-6;
    double total = 0.0;
    for ( int f = 0; f < nf; ++f )
    {
        total += static_cast<double>( h_area( f ) );
        if ( std::abs( static_cast<double>( h_area( f ) ) - 0.5 ) > tol )
            ++fails;
        // raw normal (0,0,1): magnitude 1 == 2*area, pointing +z.
        if ( std::abs( static_cast<double>( h_nrm( f, 0 ) ) ) > tol ||
             std::abs( static_cast<double>( h_nrm( f, 1 ) ) ) > tol ||
             std::abs( static_cast<double>( h_nrm( f, 2 ) ) - 1.0 ) > tol )
            ++fails;
        // Each triangle here is right-angled: one corner cot 0, two corners 1.
        int zeros = 0, ones = 0;
        for ( int c = 0; c < 3; ++c )
        {
            const double v = static_cast<double>( h_cot( f, c ) );
            if ( std::abs( v ) < tol )
                ++zeros;
            else if ( std::abs( v - 1.0 ) < tol )
                ++ones;
        }
        if ( zeros != 1 || ones != 2 )
            ++fails;
    }
    if ( std::abs( total - 1.0 ) > tol )
        ++fails;

    // Edge vectors: reference from host copies of positions + local endpoints.
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto h_ev = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                     geom.edgeVerts );
    for ( int e = 0; e < ne; ++e )
    {
        const int v0 = h_ev( e, 0 );
        const int v1 = h_ev( e, 1 );
        for ( int d = 0; d < 3; ++d )
        {
            const double expect = static_cast<double>( pos( v1, d ) ) -
                                  static_cast<double>( pos( v0, d ) );
            if ( std::abs( static_cast<double>( h_evec( e, d ) ) - expect ) >
                 tol )
                ++fails;
        }
    }

    std::printf( "  [%s] %s (nf=%d ne=%d total_area=%.6f)\n", tag,
                 fails == 0 ? "ok" : "FAIL", nf, ne, total );
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
            std::printf( "test_geometry: raw geometric primitives\n" );

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
