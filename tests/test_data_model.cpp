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

// Unit test: core data model.
//
// Exercises Mesh<Scalar, Dim, ...> construction, AoSoA sizing, and typed
// read/write of both CORE slices (gid, position[Dim], connectivity) and a USER
// field pack (a Scalar[2] vertex field + a Scalar face field), plus the
// edge-key side table. Runs for both `double` and `float` (templated precision)
// on both the host (Serial) and device (default, HIP on Tuolumne) spaces, and
// includes a Dim=2 instantiation to prove the embedding dimension parameter
// compiles and runs.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Tessera;

// Build a 3D mesh with a user field pack on `Exec`, fill core + user data on
// device, copy back to host, and verify. Returns failure count.
template <class Scalar, class Exec>
int run_3d( const char* label )
{
    using mem = typename Exec::memory_space;
    using VF =
        VertexFields<Scalar[2]>;   // one user vertex field (e.g. vorticity)
    using FF = FaceFields<Scalar>; // one user face field  (e.g. curvature)
    using MeshT = Mesh<Scalar, 3, VF, EdgeFields<>, FF, mem, Exec>;

    MeshT mesh( MPI_COMM_WORLD );

    const std::size_t nv = 64, ne = 96, nf = 40;
    mesh.resizeVertices( nv );
    mesh.resizeEdges( ne );
    mesh.resizeFaces( nf );

    int fails = 0;
    if ( mesh.numVertices() != nv || mesh.numEdges() != ne ||
         mesh.numFaces() != nf )
        ++fails;

    // ---- fill vertices: gid, position[3], user vorticity[2] ----
    auto vgid = mesh.template vertexSlice<VertexField::Gid>();
    auto vpos = mesh.template vertexSlice<VertexField::Position>();
    auto vvort = mesh.template vertexSlice<userVertexField<0>()>();
    Kokkos::parallel_for(
        "fill_verts", Kokkos::RangePolicy<Exec>( 0, nv ),
        KOKKOS_LAMBDA( const int i ) {
            vgid( i ) = static_cast<GlobalId>( i );
            for ( int d = 0; d < 3; ++d )
                vpos( i, d ) = static_cast<Scalar>( i * 10 + d );
            vvort( i, 0 ) = static_cast<Scalar>( i );
            vvort( i, 1 ) = static_cast<Scalar>( -i );
        } );

    // ---- fill edges: gid, endpoint verts; and build the edge-key side table
    // --
    auto egid = mesh.template edgeSlice<EdgeField::Gid>();
    auto everts = mesh.template edgeSlice<EdgeField::Verts>();
    mesh.edgeKeys() = Kokkos::View<EdgeKey*, mem>(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ), ne );
    auto ekeys = mesh.edgeKeys();
    Kokkos::parallel_for(
        "fill_edges", Kokkos::RangePolicy<Exec>( 0, ne ),
        KOKKOS_LAMBDA( const int i ) {
            const GlobalId v0 = static_cast<GlobalId>( i );
            const GlobalId v1 = static_cast<GlobalId>( ( i + 1 ) % 64 );
            egid( i ) = static_cast<GlobalId>( 1000 + i );
            everts( i, 0 ) = v0;
            everts( i, 1 ) = v1;
            ekeys( i ) = makeEdgeKey( v0, v1 );
        } );

    // ---- fill faces: gid, corner verts, edge ids, user curvature ----
    auto fgid = mesh.template faceSlice<FaceField::Gid>();
    auto fverts = mesh.template faceSlice<FaceField::Verts>();
    auto fcurv = mesh.template faceSlice<userFaceField<0>()>();
    Kokkos::parallel_for(
        "fill_faces", Kokkos::RangePolicy<Exec>( 0, nf ),
        KOKKOS_LAMBDA( const int i ) {
            fgid( i ) = static_cast<GlobalId>( 2000 + i );
            for ( int k = 0; k < 3; ++k )
                fverts( i, k ) = static_cast<GlobalId>( ( i + k ) % 64 );
            fcurv( i ) = static_cast<Scalar>( i ) * static_cast<Scalar>( 0.5 );
        } );
    Kokkos::fence();

    // ---- copy back to host and verify ----
    using host_v =
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>;
    host_v hv( "hv", nv );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto hgid = Cabana::slice<VertexField::Gid>( hv );
    auto hpos = Cabana::slice<VertexField::Position>( hv );
    auto hvort = Cabana::slice<userVertexField<0>()>( hv );
    for ( std::size_t i = 0; i < nv; ++i )
    {
        if ( hgid( i ) != static_cast<GlobalId>( i ) )
            ++fails;
        for ( int d = 0; d < 3; ++d )
            if ( hpos( i, d ) != static_cast<Scalar>( i * 10 + d ) )
                ++fails;
        if ( hvort( i, 0 ) != static_cast<Scalar>( i ) ||
             hvort( i, 1 ) != static_cast<Scalar>( -static_cast<long>( i ) ) )
            ++fails;
    }

    // edge-key side table round-trips the endpoint pair (order-invariant).
    auto hkeys =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ekeys );
    for ( std::size_t i = 0; i < ne; ++i )
    {
        const GlobalId v0 = static_cast<GlobalId>( i );
        const GlobalId v1 = static_cast<GlobalId>( ( i + 1 ) % 64 );
        if ( hkeys( i ) != makeEdgeKey( v0, v1 ) )
            ++fails;
    }

    std::printf( "  [%s] %s\n", label, fails == 0 ? "ok" : "FAIL" );
    return fails;
}

// Prove the Dim=2 embedding compiles and runs (planar mesh) on `Exec`.
template <class Scalar, class Exec>
int run_2d( const char* label )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<Scalar, 2, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    MeshT mesh( MPI_COMM_WORLD );
    const std::size_t nv = 8;
    mesh.resizeVertices( nv );
    auto vpos = mesh.template vertexSlice<VertexField::Position>();
    Kokkos::parallel_for(
        "fill_2d", Kokkos::RangePolicy<Exec>( 0, nv ),
        KOKKOS_LAMBDA( const int i ) {
            for ( int d = 0; d < 2; ++d )
                vpos( i, d ) = static_cast<Scalar>( i + d );
        } );
    Kokkos::fence();
    const int fails = ( mesh.numVertices() == nv && MeshT::dim == 2 ) ? 0 : 1;
    std::printf( "  [%s] %s\n", label, fails == 0 ? "ok" : "FAIL" );
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
            std::printf( "test_data_model: Mesh construction + field pack\n" );

        // Host (Serial) — covers the SERIAL gate backend.
        fails += run_3d<double, Kokkos::Serial>( "Serial/double" );
        fails += run_3d<float, Kokkos::Serial>( "Serial/float" );
        fails += run_2d<double, Kokkos::Serial>( "Serial/double/Dim2" );

        // Device (default = HIP on Tuolumne) — covers the HIP gate backend.
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
        {
            fails += run_3d<double, Kokkos::DefaultExecutionSpace>(
                "Default/double" );
            fails +=
                run_3d<float, Kokkos::DefaultExecutionSpace>( "Default/float" );
            fails += run_2d<double, Kokkos::DefaultExecutionSpace>(
                "Default/double/Dim2" );
        }
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
