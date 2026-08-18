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

// Unit test: weighted-stencil apply (applyStencil), halo-correct across ranks.
//
// Per the design doc, applyStencil is tested against an ANALYTIC field with
// uniform weights (w == 1) so a failure is unambiguously an apply/halo bug, not
// expected operator inconsistency from a weight scheme. The field is f(p) = p_x.
//
//   reference:  out_ref(v) = sum over 1-ring neighbours j of  f(pos_j)
//               computed directly from (already-haloed) positions, so it does
//               NOT depend on the `in`-field halo path under test.
//   actual:     set in = f(pos) on OWNED vertices only  ->  haloExchange the
//               mesh (fills ghost `in`)  ->  applyStencil  ->  compare on owned.
//
// If the halo were broken, ghost `in` values would be stale and the owned
// results near partition boundaries would diverge from the reference. Runs over
// 1-5 ranks so partition boundaries are exercised; Serial and default (HIP).

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
    // two user vertex fields: [0] = in (f), [1] = out.
    using MeshT = Mesh<Scalar, 3, VertexFields<Scalar, Scalar>, EdgeFields<>,
                       FaceFields<>, mem, Exec>;
    constexpr std::size_t IN = userVertexField<0>();
    constexpr std::size_t OUT = userVertexField<1>();
    int fails = 0;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );
    MeshHalo<mem> halo;
    auto owner = facePartitionByAxis( mesh );
    distribute( mesh, halo, owner );
    haloExchange( mesh, halo );

    const int n_owned = static_cast<int>( mesh.numOwnedVertices() );

    auto stencil = buildVertexStencil( mesh, 1 );

    // Uniform weights (== 1) aligned to the stencil CSR neighbour array.
    const int n_entries = stencil.csr.get().numEntries();
    Kokkos::View<Scalar*, mem> w( "w", n_entries );
    Kokkos::deep_copy( w, Scalar( 1 ) );

    // in = f(pos) = pos_x on OWNED vertices only; ghosts filled by haloExchange.
    {
        auto pos = mesh.template vertexSlice<VertexField::Position>();
        auto in = mesh.template vertexSlice<IN>();
        Kokkos::parallel_for(
            "set_in", Kokkos::RangePolicy<Exec>( 0, n_owned ),
            KOKKOS_LAMBDA( const int i ) { in( i ) = pos( i, 0 ); } );
        Kokkos::fence();
    }
    haloExchange( mesh, halo );

    // apply
    {
        auto in = mesh.template vertexSlice<IN>();
        auto out = mesh.template vertexSlice<OUT>();
        applyStencil( mesh, stencil, w, in, out );
    }

    // ---- reference from positions (independent of the in-field halo path) ---
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto out_slice = Cabana::slice<OUT>( hv );

    const auto& csr = stencil.csr.get();
    auto off =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), csr.offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    csr.neighbors );

    const double tol = 1e-9;
    for ( int v = 0; v < n_owned; ++v )
    {
        double ref = 0.0;
        for ( int p = off( v ); p < off( v + 1 ); ++p )
            ref += static_cast<double>( pos( nbr( p ), 0 ) );
        if ( std::abs( static_cast<double>( out_slice( v ) ) - ref ) > tol )
            ++fails;
    }

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
        std::printf( "  [%s] %s\n", tag, global_fails == 0 ? "ok" : "FAIL" );
    return global_fails;
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
                "test_apply_stencil: weighted stencil apply (halo)\n" );

        fails += run<double, Kokkos::Serial>( "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<double, Kokkos::DefaultExecutionSpace>( "Default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
