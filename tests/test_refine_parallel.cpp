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

// Regression test: distributed 2:1-balanced refinement (Step 6b).
//
// Builds a replicated icosphere, distributes it, and refines the distributed
// mesh's owned faces. Two scenarios:
//   Uniform  (refine every owned face) -> the result is conforming, so the global
//            owned counts equal one subdivision level (V+E, 2E+3F, 4F), the
//            owned-only Euler number is 2, the 2:1 balance holds trivially, every
//            shared boundary-edge midpoint gid agrees across ranks, and the
//            fixpoint needs a single pass.
//   Adaptive (refine a gid-predicate subset) -> the 2:1 mark-propagation fixpoint
//            terminates within the cap and enforces the 2:1 level invariant across
//            partition boundaries, and every shared boundary-edge midpoint gid is
//            bit-identical on both sides (the key guarantee). Euler is not checked
//            (adaptive refinement leaves bounded hanging nodes).
// Runs on host (Serial) and device (default, HIP), np1-5. The post-refinement
// halo rebuild is Step 7; these invariants are all owned-entity properties.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using namespace Tessera;

// Owned-face gids of a distributed mesh (host).
template <class MeshT>
std::vector<GlobalId> ownedFaceGids( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    std::vector<GlobalId> out( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
        out[f] = g( f );
    return out;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // Pinned to the hanging-node mode: this test asserts the 2:1 contract
    // (owned Euler holds only for a UNIFORM mask, a bisected edge survives
    // on its kept side), which is no longer the Mesh default. The conforming
    // counterpart is the refine_conforming test.
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;

    int fails = 0;

    // Coarse-level entity counts for the icosphere used below (subdiv 2).
    const long long V0 = 162, E0 = 480, F0 = 320;

    // ---- Uniform refinement ------------------------------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        std::vector<char> mask( mesh.numOwnedFaces(), 1 );
        auto res = refine( mesh, halo, mask );

        int local = 0;
        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints );
        local += TesseraTest::check21Balance( mesh );
        int glob = 0;
        MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

        const long long gV = TesseraTest::globalOwnedVertices( mesh );
        const long long gE = TesseraTest::globalOwnedEdges( mesh );
        const long long gF = TesseraTest::globalOwnedFaces( mesh );
        const long long euler = TesseraTest::ownedEulerGlobal( mesh );

        if ( glob != 0 )
            ++fails;
        if ( gV != V0 + E0 || gE != 2 * E0 + 3 * F0 || gF != 4 * F0 )
            ++fails; // one conforming subdivision level
        if ( euler != 2 )
            ++fails;
        if ( res.iterations != 1 )
            ++fails; // uniform: nothing to propagate
        if ( rank == 0 )
            std::printf(
                "  [%s] uniform %s (V=%lld E=%lld F=%lld euler=%lld it=%d)\n",
                tag, ( glob == 0 && fails == 0 ) ? "ok" : "FAIL", gV, gE, gF,
                euler, res.iterations );
    }

    // ---- Adaptive refinement -----------------------------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        // Refine a deterministic gid-predicate subset (non-uniform, crosses
        // partition boundaries).
        auto fg = ownedFaceGids( mesh );
        std::vector<char> mask( mesh.numOwnedFaces(), 0 );
        for ( std::size_t f = 0; f < mask.size(); ++f )
            mask[f] = ( fg[f] % 7 == 0 ) ? 1 : 0;
        auto res = refine( mesh, halo, mask );

        int local = 0;
        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints );
        local += TesseraTest::check21Balance( mesh );
        int glob = 0;
        MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

        if ( glob != 0 )
            ++fails;
        if ( res.iterations <= 0 || res.iterations >= 256 )
            ++fails; // fixpoint must terminate within the cap
        if ( rank == 0 )
            std::printf( "  [%s] adaptive %s (it=%d)\n", tag,
                         ( glob == 0 ) ? "ok" : "FAIL", res.iterations );
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
                "test_refine_parallel: distributed 2:1 refinement (size %d)\n",
                size );

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );
    }

    Kokkos::finalize();

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
