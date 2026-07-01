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

// Unit test: serial icosphere builder + connectivity (Step 3).
//
// Builds icosphere meshes at several subdivision levels and checks the topology
// invariants of a closed genus-0 surface: exact V/E/F counts, Euler
// V - E + F = 2, every edge shared by exactly two faces, every face bounded by
// three valid vertices/edges, the vertex 1-ring CSR relations round-trip, and
// vertices lie on the unit sphere. Runs on both the host (Serial) and device
// (default, HIP on Tuolumne) execution spaces.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Tessera;

// Copy an arbitrary-space mesh's topology to the host and check all invariants
// against the expected V/E/F. Returns the failure count.
template <class MeshT>
int check_mesh( MeshT& mesh, std::size_t expV, std::size_t expE,
                std::size_t expF, const char* label )
{
    int fails = 0;
    const std::size_t nv = mesh.numVertices();
    const std::size_t ne = mesh.numEdges();
    const std::size_t nf = mesh.numFaces();

    if ( nv != expV || ne != expE || nf != expF )
    {
        std::printf( "  [%s] FAIL counts: V=%zu(exp %zu) E=%zu(exp %zu) "
                     "F=%zu(exp %zu)\n",
                     label, nv, expV, ne, expE, nf, expF );
        ++fails;
    }
    // Euler characteristic of a genus-0 closed surface.
    if ( static_cast<long>( nv ) - static_cast<long>( ne ) +
             static_cast<long>( nf ) !=
         2 )
    {
        std::printf( "  [%s] FAIL Euler: V-E+F=%ld\n", label,
                     static_cast<long>( nv ) - static_cast<long>( ne ) +
                         static_cast<long>( nf ) );
        ++fails;
    }

    // Copy AoSoAs to host for inspection.
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", ne );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );

    auto vpos = Cabana::slice<VertexField::Position>( hv );
    auto efaces = Cabana::slice<EdgeField::Faces>( he );
    auto fverts = Cabana::slice<FaceField::Verts>( hf );
    auto fedges = Cabana::slice<FaceField::Edges>( hf );

    // Every vertex on the unit sphere.
    for ( std::size_t i = 0; i < nv; ++i )
    {
        const double r =
            std::sqrt( static_cast<double>( vpos( i, 0 ) ) * vpos( i, 0 ) +
                       static_cast<double>( vpos( i, 1 ) ) * vpos( i, 1 ) +
                       static_cast<double>( vpos( i, 2 ) ) * vpos( i, 2 ) );
        if ( std::abs( r - 1.0 ) > 1e-4 )
        {
            ++fails;
            break;
        }
    }

    // Every edge shared by exactly two faces (closed surface, no boundary).
    for ( std::size_t e = 0; e < ne; ++e )
        if ( efaces( e, 0 ) == invalid_gid || efaces( e, 1 ) == invalid_gid )
        {
            ++fails;
            break;
        }

    // Every face has three distinct valid vertices and three valid edges.
    for ( std::size_t f = 0; f < nf; ++f )
    {
        for ( int k = 0; k < 3; ++k )
        {
            if ( fverts( f, k ) >= nv || fedges( f, k ) >= ne )
            {
                ++fails;
                break;
            }
        }
        if ( fverts( f, 0 ) == fverts( f, 1 ) ||
             fverts( f, 1 ) == fverts( f, 2 ) ||
             fverts( f, 0 ) == fverts( f, 2 ) )
            ++fails;
    }

    // CSR vertex -> faces: total degree == 3F, and every listed face contains
    // the source vertex.
    auto vf_off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().offsets );
    auto vf_nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().neighbors );
    if ( vf_nbr.extent( 0 ) != 3 * nf )
        ++fails;
    for ( std::size_t v = 0; v < nv; ++v )
        for ( int p = vf_off( v ); p < vf_off( v + 1 ); ++p )
        {
            const LocalIndex fidx = vf_nbr( p );
            const bool has = fverts( fidx, 0 ) == v || fverts( fidx, 1 ) == v ||
                             fverts( fidx, 2 ) == v;
            if ( !has )
            {
                ++fails;
                break;
            }
        }

    // CSR vertex -> edges: total degree == 2E, and every listed edge has the
    // source vertex as an endpoint.
    auto ve_off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexEdges().offsets );
    auto ve_nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexEdges().neighbors );
    auto everts = Cabana::slice<EdgeField::Verts>( he );
    if ( ve_nbr.extent( 0 ) != 2 * ne )
        ++fails;
    for ( std::size_t v = 0; v < nv; ++v )
        for ( int p = ve_off( v ); p < ve_off( v + 1 ); ++p )
        {
            const LocalIndex eidx = ve_nbr( p );
            if ( everts( eidx, 0 ) != v && everts( eidx, 1 ) != v )
            {
                ++fails;
                break;
            }
        }

    std::printf( "  [%s] %s (V=%zu E=%zu F=%zu)\n", label,
                 fails == 0 ? "ok" : "FAIL", nv, ne, nf );
    return fails;
}

// Expected V/E/F for an icosphere at subdivision level `s`.
void expected_counts( int s, std::size_t& V, std::size_t& E, std::size_t& F )
{
    V = 12;
    E = 30;
    F = 20;
    for ( int i = 0; i < s; ++i )
    {
        const std::size_t nF = 4 * F;
        const std::size_t nV = V + E;
        const std::size_t nE = 2 * E + 3 * F;
        V = nV;
        E = nE;
        F = nF;
    }
}

template <class Scalar, class Exec>
int run( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    int fails = 0;
    for ( int s = 0; s <= 3; ++s )
    {
        std::size_t V, E, F;
        expected_counts( s, V, E, F );
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, s );
        char label[64];
        std::snprintf( label, sizeof( label ), "%s/subdiv%d", tag, s );
        fails += check_mesh( mesh, V, E, F, label );
    }
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
            std::printf(
                "test_connectivity: icosphere builder + connectivity\n" );

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
