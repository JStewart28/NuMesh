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

// Regression test: distributed CONFORMING refinement (Task 4 of
// tasks/conforming-refinement.md). This is the milestone test for the feature.
//
// refine() on a RefinementMode::Conforming mesh un-closes the transient closure
// layer, runs the same 2:1-balanced red engine as the hanging-node mode, and
// re-closes every kept red face whose edges a neighbour bisected. The result is
// a mesh with NO hanging nodes under an ARBITRARY (adaptive) mask. Cases:
//
//   A. THREE SUCCESSIVE ADAPTIVE ROUNDS. After each round:
//        checkConforming        every global edge has exactly two incident faces
//        checkOwnedEuler == 2   the headline criterion -- this is what the
//                               hanging-node mode fails under an adaptive mask
//        checkNoInteriorVertex  geometric: no vertex strictly inside an edge
//        checkOwnershipPartition / checkMidpointAgreement  (unchanged contracts)
//        check21BalanceRed      the 2:1 invariant, applied to the RED layer --
//                               closure children carry their parent's level, so
//                               the balance is a property of the red faces
//        checkClosureInverse    unclose o close reproduces the visible layer and
//                               then the red layer bit-for-bit
//      Plus NON-VACUITY: closure children must actually be emitted at every rank
//      count, and the same three conformity checks are run on a HangingNode2to1
//      mesh with the SAME mask and must FAIL there -- otherwise a mask too weak
//      to create a hanging node would make the whole case prove nothing.
//
//   B. EMPTY MASK. Nothing splits, so nothing needs closing: the visible
//      topology is unchanged and |S| = 0 everywhere.
//
//   C. FULL (UNIFORM) MASK. Every face refines, so there are no kept faces and
//      the closure is inert -- |S| = 0 everywhere, no closure children, and the
//      global V/E/F counts must equal the hanging-node mode's exactly.
//
// Prints the closure-face fraction and the |S|-pattern histogram per round for
// the Task-8 measurement table.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

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

//! Gid of every OWNED face (host snapshot).
template <class MeshT>
std::vector<GlobalId> ownedFaceGids( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    std::vector<GlobalId> out( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < out.size(); ++f )
        out[f] = g( f );
    return out;
}

//! The deterministic adaptive mask used by every case: refine face gids
//! divisible by `m`. Non-uniform, crosses partition boundaries, and (unlike a
//! local-index predicate) rank-count independent.
template <class MeshT>
std::vector<char> gidMask( MeshT& mesh, int m )
{
    const std::vector<GlobalId> g = ownedFaceGids( mesh );
    std::vector<char> mask( g.size(), 0 );
    for ( std::size_t f = 0; f < g.size(); ++f )
        mask[f] = ( g[f] % m == 0 ) ? 1 : 0;
    return mask;
}

//! Globally summed |S| histogram + closure/visible counts for one refine call.
struct ClosureTotals
{
    long long pattern[4] = { 0, 0, 0, 0 };
    long long visible = 0;
    long long closureChildren = 0;
    long long blueLo1 = 0, blueLo2 = 0;
};

inline ClosureTotals reduceClosure( const ClosureStats& s )
{
    long long local[8] = { s.patternCount[0],   s.patternCount[1],
                           s.patternCount[2],   s.patternCount[3],
                           s.nVisible,          s.nClosureChildren,
                           s.nBlueDiagLowFirst, s.nBlueDiagLowSecond };
    long long g[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    MPI_Allreduce( local, g, 8, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    ClosureTotals t;
    for ( int i = 0; i < 4; ++i )
        t.pattern[i] = g[i];
    t.visible = g[4];
    t.closureChildren = g[5];
    t.blueLo1 = g[6];
    t.blueLo2 = g[7];
    return t;
}

//! The three mode-agnostic conformity criteria, as one global figure each.
struct Conformity
{
    long long badIncidence = 0; //!< edges without exactly two incident faces
    long long euler = 0;
    long long interiorVerts = 0;
};

template <class MeshT>
Conformity measureConformity( MeshT& mesh )
{
    Conformity c;
    long long local[2] = {
        static_cast<long long>( TesseraTest::checkConforming( mesh ) ),
        static_cast<long long>( TesseraTest::checkNoInteriorVertex( mesh ) ) };
    long long g[2] = { 0, 0 };
    MPI_Allreduce( local, g, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    c.badIncidence = g[0];
    c.interiorVerts = g[1];
    c.euler = TesseraTest::checkOwnedEuler( mesh );
    return c;
}

//! Sum a LOCAL fail count into a global one.
inline int globalFails( int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return g;
}

// ===========================================================================
// Case A -- three successive adaptive rounds
// ===========================================================================

template <class Exec, class ConfMesh, class HangMesh>
int case_adaptive( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    // The same mask sequence on a HangingNode2to1 mesh, as the non-vacuity
    // control: the conformity checks must FAIL there.
    HangMesh ctrl( MPI_COMM_WORLD );
    buildIcosphere( ctrl, 2 );
    MeshHalo<mem> ctrlHalo;
    {
        auto faceOwner = facePartitionByAxis( ctrl );
        distribute( ctrl, ctrlHalo, faceOwner );
    }

    long long totalClosure = 0;
    long long ctrlBad = 0;

    for ( int round = 0; round < 3; ++round )
    {
        const int m = ( round == 0 ) ? 7 : 5;

        auto res = refine( mesh, halo, gidMask( mesh, m ) );
        auto ctrlRes = refine( ctrl, ctrlHalo, gidMask( ctrl, m ) );
        (void)ctrlRes;

        int local = 0;
        local += TesseraTest::checkConforming( mesh );
        local += TesseraTest::checkNoInteriorVertex( mesh );
        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints );
        local += TesseraTest::check21BalanceRed( mesh );
        local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
        local += TesseraTest::checkOwnershipPartition(
            mesh, TesseraTest::globalOwnedVertices( mesh ),
            TesseraTest::globalOwnedEdges( mesh ),
            TesseraTest::globalOwnedFaces( mesh ) );
        const int glob = globalFails( local );
        if ( glob != 0 )
            ++fails;

        const long long euler = TesseraTest::checkOwnedEuler( mesh );
        if ( euler != 2 )
            ++fails; // THE criterion: conforming under an adaptive mask
        if ( res.iterations <= 0 || res.iterations >= 256 )
            ++fails;

        const ClosureTotals ct = reduceClosure( res.closure );
        totalClosure += ct.closureChildren;
        if ( ct.visible != TesseraTest::globalOwnedFaces( mesh ) )
            ++fails; // the visible count refine() reported is the mesh's

        // Non-vacuity control: the same mask on a hanging-node mesh must leave
        // T-junctions, or the conforming result above proves nothing.
        const Conformity cc = measureConformity( ctrl );
        ctrlBad += cc.badIncidence + cc.interiorVerts;

        if ( rank == 0 )
        {
            const double frac =
                ct.visible > 0 ? static_cast<double>( ct.closureChildren ) /
                                     static_cast<double>( ct.visible )
                               : 0.0;
            std::printf( "  [%s] adaptive round%d %s (it=%d F=%lld euler=%lld "
                         "closure=%lld/%lld=%.3f |S| hist=[%lld,%lld,%lld,%lld]"
                         " blue lo1/lo2=%lld/%lld | control euler=%lld "
                         "badInc=%lld tjunc=%lld)\n",
                         tag, round + 1,
                         ( glob == 0 && euler == 2 ) ? "ok" : "FAIL",
                         res.iterations, ct.visible, euler, ct.closureChildren,
                         ct.visible, frac, ct.pattern[0], ct.pattern[1],
                         ct.pattern[2], ct.pattern[3], ct.blueLo1, ct.blueLo2,
                         cc.euler, cc.badIncidence, cc.interiorVerts );
        }
    }

    if ( totalClosure <= 0 )
        ++fails; // vacuous: the closure never fired
    if ( ctrlBad <= 0 )
        ++fails; // vacuous: the mask never created a hanging node at all

    return fails;
}

// ===========================================================================
// Case B -- empty mask: closure is the identity
// ===========================================================================

template <class Exec, class ConfMesh>
int case_empty( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    const long long V0 = TesseraTest::globalOwnedVertices( mesh );
    const long long E0 = TesseraTest::globalOwnedEdges( mesh );
    const long long F0 = TesseraTest::globalOwnedFaces( mesh );
    unsigned long long cv0, ce0, cf0;
    TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );

    std::vector<char> mask( mesh.numOwnedFaces(), 0 );
    auto res = refine( mesh, halo, mask );

    int local = TesseraTest::checkConforming( mesh );
    local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
    if ( globalFails( local ) != 0 )
        ++fails;

    const ClosureTotals ct = reduceClosure( res.closure );
    unsigned long long cv1, ce1, cf1;
    TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );

    if ( TesseraTest::globalOwnedVertices( mesh ) != V0 ||
         TesseraTest::globalOwnedEdges( mesh ) != E0 ||
         TesseraTest::globalOwnedFaces( mesh ) != F0 )
        ++fails; // an empty mask must not change any count
    if ( cv1 != cv0 || ce1 != ce0 || cf1 != cf0 )
        ++fails; // ... nor any gid
    if ( ct.closureChildren != 0 || ct.pattern[1] || ct.pattern[2] ||
         ct.pattern[3] )
        ++fails; // nothing was split, so nothing may be closed
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;
    if ( !res.midpoints.empty() )
        ++fails; // no edge was bisected

    (void)size;
    if ( rank == 0 )
        std::printf( "  [%s] empty-mask %s (V=%lld E=%lld F=%lld "
                     "closureChildren=%lld)\n",
                     tag, fails == 0 ? "ok" : "FAIL", V0, E0, F0,
                     ct.closureChildren );
    return fails;
}

// ===========================================================================
// Case C -- full mask: no kept faces, so the closure must be inert
// ===========================================================================

template <class Exec, class ConfMesh, class HangMesh>
int case_uniform( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    HangMesh ref( MPI_COMM_WORLD );
    buildIcosphere( ref, 2 );
    MeshHalo<mem> refHalo;
    {
        auto faceOwner = facePartitionByAxis( ref );
        distribute( ref, refHalo, faceOwner );
    }

    std::vector<char> mask( mesh.numOwnedFaces(), 1 );
    auto res = refine( mesh, halo, mask );
    std::vector<char> refMask( ref.numOwnedFaces(), 1 );
    refine( ref, refHalo, refMask );

    int local = TesseraTest::checkConforming( mesh );
    local += TesseraTest::checkNoInteriorVertex( mesh );
    local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                  res.midpoints );
    local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
    if ( globalFails( local ) != 0 )
        ++fails;

    const ClosureTotals ct = reduceClosure( res.closure );
    if ( ct.closureChildren != 0 || ct.pattern[1] || ct.pattern[2] ||
         ct.pattern[3] )
        ++fails; // no kept faces => |S| = 0 everywhere => closure is inert
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;

    const long long gV = TesseraTest::globalOwnedVertices( mesh );
    const long long gE = TesseraTest::globalOwnedEdges( mesh );
    const long long gF = TesseraTest::globalOwnedFaces( mesh );
    if ( gV != TesseraTest::globalOwnedVertices( ref ) ||
         gE != TesseraTest::globalOwnedEdges( ref ) ||
         gF != TesseraTest::globalOwnedFaces( ref ) )
        ++fails; // an inert closure must reproduce the hanging-node counts

    if ( rank == 0 )
        std::printf(
            "  [%s] uniform-mask %s (V=%lld E=%lld F=%lld closureChildren=%lld "
            "|S| hist=[%lld,%lld,%lld,%lld])\n",
            tag, fails == 0 ? "ok" : "FAIL", gV, gE, gF, ct.closureChildren,
            ct.pattern[0], ct.pattern[1], ct.pattern[2], ct.pattern[3] );
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using ConfMesh = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::Conforming>;
    using HangMesh = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::HangingNode2to1>;

    int fails = 0;
    fails += case_adaptive<Exec, ConfMesh, HangMesh>( rank, size, tag );
    fails += case_empty<Exec, ConfMesh>( rank, size, tag );
    fails += case_uniform<Exec, ConfMesh, HangMesh>( rank, size, tag );
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
            std::printf( "test_refine_conforming: distributed conforming "
                         "refinement (size %d)\n",
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
