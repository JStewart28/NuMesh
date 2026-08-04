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

// Test: TRIANGLE SHAPE QUALITY OVER REFINEMENT DEPTH (Task 7 of
// tasks/conforming-refinement.md) -- the guarantee that buys the closure its
// "transient" design.
//
// A closure that PERSISTED would be a disaster for shape: bisecting an already
// bisected green triangle, round after round, drives its min angle to zero with
// no lower bound. Tessera's closure is instead recomputed from scratch on every
// refine call: un-close discards the whole closure layer, the red engine sees
// only red faces, and the closure is rebuilt. Consequently EVERY visible
// triangle is either a red triangle or one of three fixed retriangulations of
// one, so the number of triangle similarity classes is BOUNDED and the worst
// shape is bounded with it -- independent of how many rounds have been run.
//
// This test measures that. It refines a shrinking geodesic cap for >= 8 rounds
// and tracks, per round: the global minimum triangle angle, the global maximum
// radius ratio, and the closure-face fraction. It then asserts each stays inside
// a FIXED bound -- fixed being the whole point: a bound that had to grow with
// the round count would mean the closure is not transient after all.
//
//   MIN ANGLE. On an exactly equilateral parent the four patterns give: |S|=0
//   60 deg, |S|=1 (green, a median cut) 30 deg, |S|=2 (blue) 30 deg (the
//   30-30-120 sliver off the midline is the worst of the three children), |S|=3
//   (red-closure) 60 deg. Real red faces are icosphere triangles, which are
//   near- but not exactly equilateral, so the realised worst case sits somewhat
//   below 30.
//
//   RADIUS RATIO Q = abc(a+b+c) / (16 A^2), which is 1 for an equilateral
//   triangle and grows as the triangle degenerates (it is R/2r, circumradius
//   over inradius, normalised). The ideal-parent worst cases are Q = 1.37 for
//   the 30-60-90 green child and Q = 2.16 for the 30-30-120 blue child.
//
//   CLOSURE FRACTION. The closure covers the level-jump BOUNDARY, an O(perimeter)
//   set, while the mesh grows by area, so the fraction must stay bounded well
//   below 1 rather than tracking the mesh size.
//
// *** THE BOUNDS BELOW ARE PROVISIONAL. *** Per the Task-7 acceptance criteria,
// no test in Tasks 1-7 has been executed, so these are derived from the geometry
// above plus margin -- not from measurement. This test is therefore registered
// at the `unit` tier, NOT in the ship gate. Task 8 runs it, replaces the bounds
// with values justified by the printed per-round data, and promotes it to
// `regression` at ranks 1-5 only if it proves stable across repeated runs. If
// the measured quality turns out to be genuinely unbounded in the round count,
// that is a DESIGN finding about the closure, not a tolerance to loosen.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace Tessera;

// ---- PROVISIONAL quality bounds (see the header note; Task 8 calibrates) ----
//
// Derived, not measured: the ideal-parent worst cases are 30 deg / Q = 2.16,
// and the red layer's own faces are icosphere triangles whose min angle is
// already ~54 deg rather than 60. These allow a further ~1/3 of shape loss for
// that distortion before failing.
static constexpr double kMinAngleDeg = 20.0;    //!< provisional
static constexpr double kMaxRadiusRatio = 4.0;  //!< provisional
static constexpr double kMaxClosureFrac = 0.50; //!< provisional

static constexpr int kRounds = 8;
static constexpr double kPi = 3.14159265358979323846;

static inline long long globalSum( long long v )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    return g;
}
static inline double globalMinD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
    return g;
}
static inline double globalMaxD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return g;
}

//! Vertex gid -> position over every locally held vertex.
template <class MeshT>
static std::unordered_map<GlobalId, std::array<double, 3>>
readPositions( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );
    std::unordered_map<GlobalId, std::array<double, 3>> out;
    out.reserve( mesh.numVertices() * 2 );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
    {
        std::array<double, 3> p = { 0, 0, 0 };
        for ( int d = 0; d < MeshT::dim && d < 3; ++d )
            p[d] = static_cast<double>( vp( i, d ) );
        out[vg( i )] = p;
    }
    return out;
}

template <class MeshT>
static std::vector<VisibleFace> ownedVisible( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    return readVisibleFaces<MeshT>( hf, mesh.numOwnedFaces() );
}

//! One round's measurements, already reduced across ranks.
struct Quality
{
    double minAngleDeg = 180.0;
    double maxRadiusRatio = 0.0;
    double closureFrac = 0.0;
    long long faces = 0;
    long long closureChildren = 0;
    long long marked = 0;
};

//! Shape of every OWNED visible face, from the haloed positions.
template <class MeshT>
static Quality measureQuality( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    const std::vector<VisibleFace> vis = ownedVisible( mesh );

    double minAngle = 180.0, maxQ = 0.0;
    long long nClosure = 0;
    for ( const auto& f : vis )
    {
        if ( f.parent != invalid_gid )
            ++nClosure;

        std::array<double, 3> p[3];
        bool have = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( f.v[k] );
            if ( it == pos.end() )
            {
                have = false;
                break;
            }
            p[k] = it->second;
        }
        if ( !have )
        {
            ++fails; // every corner of an owned face must be held after halo
            continue;
        }

        // Side lengths: s[k] is the side OPPOSITE corner k.
        double s[3];
        for ( int k = 0; k < 3; ++k )
        {
            const std::array<double, 3>& a = p[( k + 1 ) % 3];
            const std::array<double, 3>& b = p[( k + 2 ) % 3];
            double d2 = 0.0;
            for ( int c = 0; c < 3; ++c )
                d2 += ( b[c] - a[c] ) * ( b[c] - a[c] );
            s[k] = std::sqrt( d2 );
        }
        if ( s[0] <= 0.0 || s[1] <= 0.0 || s[2] <= 0.0 )
        {
            ++fails; // a degenerate triangle is a hard failure, not a datum
            continue;
        }

        // Angles by the law of cosines; area by Heron via the cross product.
        for ( int k = 0; k < 3; ++k )
        {
            const double num = s[( k + 1 ) % 3] * s[( k + 1 ) % 3] +
                               s[( k + 2 ) % 3] * s[( k + 2 ) % 3] -
                               s[k] * s[k];
            const double den = 2.0 * s[( k + 1 ) % 3] * s[( k + 2 ) % 3];
            double cosA = num / den;
            cosA = std::max( -1.0, std::min( 1.0, cosA ) );
            minAngle = std::min( minAngle, std::acos( cosA ) * 180.0 / kPi );
        }

        const double u[3] = { p[1][0] - p[0][0], p[1][1] - p[0][1],
                              p[1][2] - p[0][2] };
        const double w[3] = { p[2][0] - p[0][0], p[2][1] - p[0][1],
                              p[2][2] - p[0][2] };
        const double n[3] = { u[1] * w[2] - u[2] * w[1],
                              u[2] * w[0] - u[0] * w[2],
                              u[0] * w[1] - u[1] * w[0] };
        const double area =
            0.5 * std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
        if ( area <= 0.0 )
        {
            ++fails;
            continue;
        }
        const double Q = s[0] * s[1] * s[2] * ( s[0] + s[1] + s[2] ) /
                         ( 16.0 * area * area );
        maxQ = std::max( maxQ, Q );
    }

    Quality q;
    q.minAngleDeg = globalMinD( minAngle );
    q.maxRadiusRatio = globalMaxD( maxQ );
    q.faces = globalSum( static_cast<long long>( vis.size() ) );
    q.closureChildren = globalSum( nClosure );
    q.closureFrac = q.faces > 0 ? static_cast<double>( q.closureChildren ) /
                                      static_cast<double>( q.faces )
                                : 0.0;
    return q;
}

//! Geodesic-cap mask: mark a face iff the direction of its centroid is within
//! `halfAngle` of +z. Purely geometric, so it is the same face set at any rank
//! count; the cap shrinks by 0.6 per round while the faces inside it shrink by
//! 0.5, which keeps the marked set a small, slowly growing patch instead of
//! either dying out or engulfing the sphere.
template <class MeshT>
static std::vector<char>
capMask( MeshT& mesh,
         const std::unordered_map<GlobalId, std::array<double, 3>>& pos,
         double halfAngle, int& fails )
{
    const std::vector<VisibleFace> vis = ownedVisible( mesh );
    const double cosLim = std::cos( halfAngle );
    std::vector<char> mask( vis.size(), 0 );
    for ( std::size_t f = 0; f < vis.size(); ++f )
    {
        double c[3] = { 0, 0, 0 };
        bool have = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( vis[f].v[k] );
            if ( it == pos.end() )
            {
                have = false;
                break;
            }
            for ( int d = 0; d < 3; ++d )
                c[d] += it->second[d] / 3.0;
        }
        if ( !have )
        {
            ++fails;
            continue;
        }
        const double len = std::sqrt( c[0] * c[0] + c[1] * c[1] + c[2] * c[2] );
        mask[f] = ( len > 0.0 && c[2] / len > cosLim ) ? 1 : 0;
    }
    return mask;
}

template <class Exec, class MeshT>
static int case_quality( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    double worstAngle = 180.0, worstQ = 0.0, worstFrac = 0.0;
    long long totalClosure = 0;
    bool properSubsetEveryRound = true;

    for ( int round = 0; round < kRounds; ++round )
    {
        const double halfAngle = 0.35 * std::pow( 0.6, round );
        const auto pos = readPositions( mesh );
        std::vector<char> mask = capMask( mesh, pos, halfAngle, fails );

        long long localMarked = 0;
        for ( char m : mask )
            localMarked += m ? 1 : 0;
        const long long gMarked = globalSum( localMarked );
        const long long gFaces =
            globalSum( static_cast<long long>( mesh.numOwnedFaces() ) );
        if ( gMarked <= 0 || gMarked >= gFaces )
            properSubsetEveryRound = false;

        refine( mesh, halo, mask );

        // refine() leaves an owned-only mesh; the quality measurement and the
        // next round's geometric mask both need corner positions locally.
        {
            std::vector<Rank> dest( mesh.numOwnedFaces(),
                                    static_cast<Rank>( rank ) );
            migrate( mesh, halo, dest );
            haloExchange( mesh, halo );
        }

        const Quality q = measureQuality( mesh, fails );
        if ( q.faces <= 0 )
            ++fails;
        totalClosure += q.closureChildren;
        worstAngle = std::min( worstAngle, q.minAngleDeg );
        worstQ = std::max( worstQ, q.maxRadiusRatio );
        worstFrac = std::max( worstFrac, q.closureFrac );

        // The bound must hold EVERY round, not just on average: a quality that
        // degrades with depth would show up as a late round crossing it.
        if ( q.minAngleDeg < kMinAngleDeg )
            ++fails;
        if ( q.maxRadiusRatio > kMaxRadiusRatio )
            ++fails;
        if ( q.closureFrac > kMaxClosureFrac )
            ++fails;

        // Conformity must survive all eight rounds too -- a quality bound on a
        // mesh that stopped being conforming would prove nothing.
        int local = TesseraTest::checkConforming( mesh );
        local += TesseraTest::check21BalanceRed( mesh );
        local += TesseraTest::checkSiblingCoresidency( mesh );
        int g = 0;
        MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( g != 0 )
            ++fails;
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++fails;

        if ( rank == 0 )
            std::printf( "  [%s] quality round%d marked=%lld F=%lld "
                         "minAngle=%.3f deg maxQ=%.4f closure=%lld (%.4f) "
                         "inv=%d\n",
                         tag, round + 1, gMarked, q.faces, q.minAngleDeg,
                         q.maxRadiusRatio, q.closureChildren, q.closureFrac,
                         g );
    }

    if ( totalClosure <= 0 )
        ++fails; // vacuous: no closure face was ever produced
    if ( !properSubsetEveryRound )
        ++fails; // vacuous: some round refined nothing, or everything

    (void)size;
    if ( rank == 0 )
        std::printf(
            "  [%s] quality over %d rounds %s (worst minAngle=%.3f "
            "deg [bound %.1f], worst Q=%.4f [bound %.1f], worst "
            "closureFrac=%.4f [bound %.2f]) -- BOUNDS ARE PROVISIONAL, "
            "Task 8 calibrates\n",
            tag, kRounds, fails == 0 ? "ok" : "FAIL", worstAngle, kMinAngleDeg,
            worstQ, kMaxRadiusRatio, worstFrac, kMaxClosureFrac );
    return fails;
}

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::Conforming>;
    return case_quality<Exec, MeshT>( rank, size, tag );
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
            std::printf( "test_conforming_quality: triangle shape over %d "
                         "adaptive rounds (size %d)\n",
                         kRounds, size );

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
