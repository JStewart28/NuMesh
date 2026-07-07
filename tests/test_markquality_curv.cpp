/****************************************************************************
 * Copyright (c) 2024, JStewart28                                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Tessera library. Tessera is distributed under a *
 * BSD 3-Clause license. For the licensing terms see the LICENSE file in   *
 * the top-level directory.                                                 *
 ****************************************************************************/

// Regression test: curvature/dihedral quality-based refinement marking
// (Step 10b).
//
// Builds a synthetic sharp-fold fixture -- a subdiv-2 icosphere with one
// original vertex (gid 0) displaced radially outward, so the ring of faces
// around it forms a steep spike while every other face stays near-flat -- and
// drives markByQuality() with a CurvatureCriterion. The dihedral threshold is
// chosen from the reference mesh's own dihedral distribution (the midpoint of
// the largest gap between sorted edge dihedrals), which cleanly separates the
// fold cluster from the flat cluster with a wide margin, so no edge sits near
// the threshold and the marked set is stable across host/device rounding.
//
// Checks:
//   (a) Correctness + rank-count independence: the marked owned-face gid set,
//       BXOR-reduced across ranks, matches a checksum computed from an
//       independent, un-partitioned reference mesh (gid == index) with no MPI
//       at all. Agreement at every rank count 1-5 proves the marked set does
//       not depend on the partition.
//   (b) Boundary-straddle independence: the fold ring around vertex 0 spans the
//       full z-extent of the sphere, so facePartitionByAxis (axis = z) splits
//       the fold across partition bands at np >= 2. The neighbour face across a
//       fold edge is therefore frequently owned by another rank and not present
//       in the vertex-based 1-ring halo -- so a correct result at np >= 2 can
//       only come from the edge-coordinator gather, not the halo. This is the
//       property (a)'s cross-np agreement pins.
//   (c) refine() post-conditions hold (check21Balance, checkMidpointAgreement)
//       via MeshInvariants.hpp. (Euler == 2 is NOT asserted: adaptive marking
//       introduces bounded hanging nodes, as in the Step-6b adaptive case.)
// Runs on host (Serial) and device (default, HIP), ranks 1-5, double and float.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

using namespace Tessera;

// Owned-face gids of a distributed mesh (host).
template <class MeshT>
std::vector<GlobalId> ownedFaceGidsHost( MeshT& mesh )
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

// Displace the vertex with gid `target` radially (scale its position by
// `factor`). Applied on the replicated mesh right after buildIcosphere (gid ==
// index), identically on the reference and the distributed copy, so both see
// exactly the same geometry.
template <class MeshT>
void displaceVertex( MeshT& mesh, GlobalId target,
                     typename MeshT::scalar_type factor )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto gid = Cabana::slice<VertexField::Gid>( hv );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
        if ( gid( i ) == target )
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) *= factor;
    Cabana::deep_copy( mesh.vertices(), hv );
}

template <class Scalar, class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    int fails = 0;

    // Coarse subdiv-2 icosphere entity counts.
    const long long F0 = 320;
    const int subdiv = 2;
    const GlobalId spikeVert = 0;              // an original icosahedron corner
    const Scalar displaceFactor = Scalar( 3 ); // push it well outward

    // Outward unit normal of a face with the CCW-from-outside winding the
    // builder maintains: n = normalize((p1-p0) x (p2-p0)). Mirrors exactly the
    // device kernel in detail::markCurvature so the reference and the
    // distributed run agree bit-for-bit on which faces are sharp.
    auto faceNormal = []( const Scalar* p0, const Scalar* p1, const Scalar* p2,
                          Scalar out[3] )
    {
        const Scalar e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
        const Scalar e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
        Scalar n[3] = { e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0] };
        const Scalar len = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
        const Scalar inv = len > Scalar( 0 ) ? Scalar( 1 ) / len : Scalar( 0 );
        for ( int d = 0; d < 3; ++d )
            out[d] = n[d] * inv;
    };

    // ---- independent reference: replicated (un-partitioned) displaced mesh ---
    // gid == index at this stage, so a per-face expected mark is looked up
    // directly by gid with no partition info. The threshold theta is derived
    // here from the reference dihedral distribution and reused verbatim for the
    // distributed CurvatureCriterion.
    Scalar theta;
    std::vector<char> expectedMark( static_cast<std::size_t>( F0 ), 0 );
    long long refMarked = 0;
    unsigned long long refXor = 0;
    {
        MeshT ref( MPI_COMM_WORLD );
        buildIcosphere( ref, subdiv );
        displaceVertex( ref, spikeVert, displaceFactor );

        const std::size_t nf = ref.numFaces();
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", ref.numVertices() );
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf", nf );
        Cabana::deep_copy( hv, ref.vertices() );
        Cabana::deep_copy( hf, ref.faces() );
        auto pos = Cabana::slice<VertexField::Position>( hv );
        auto fv = Cabana::slice<FaceField::Verts>( hf );

        // per-face outward normal
        std::vector<std::array<Scalar, 3>> nrm( nf );
        for ( std::size_t f = 0; f < nf; ++f )
        {
            Scalar p[3][3];
            for ( int k = 0; k < 3; ++k )
                for ( int d = 0; d < 3; ++d )
                    p[k][d] = pos( static_cast<int>( fv( f, k ) ), d );
            faceNormal( p[0], p[1], p[2], nrm[f].data() );
        }

        // group incident faces per edge
        std::map<EdgeKey, std::vector<int>> byEdge;
        for ( std::size_t f = 0; f < nf; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key =
                    makeEdgeKey( fv( f, k ), fv( f, ( k + 1 ) % 3 ) );
                byEdge[key].push_back( static_cast<int>( f ) );
            }

        // dihedral angle per (closed-surface) edge
        struct EdgeAng
        {
            double ang;
            int fa, fb;
        };
        std::vector<EdgeAng> edges;
        for ( const auto& kv : byEdge )
        {
            if ( kv.second.size() != 2 )
                continue;
            const int fa = kv.second[0], fb = kv.second[1];
            double dot = 0.0;
            for ( int d = 0; d < 3; ++d )
                dot += static_cast<double>( nrm[fa][d] ) * nrm[fb][d];
            if ( dot > 1.0 )
                dot = 1.0;
            if ( dot < -1.0 )
                dot = -1.0;
            edges.push_back( { std::acos( dot ), fa, fb } );
        }

        // theta = midpoint of the largest gap between sorted dihedrals. With a
        // large displacement this gap separates the (few, steep) fold edges from
        // the (many, shallow) flat edges with a wide margin, so no edge sits
        // near theta -> the marked set is insensitive to host/device rounding.
        std::vector<double> sorted;
        sorted.reserve( edges.size() );
        for ( const auto& e : edges )
            sorted.push_back( e.ang );
        std::sort( sorted.begin(), sorted.end() );
        double bestGap = -1.0, thetaD = 0.0;
        for ( std::size_t i = 0; i + 1 < sorted.size(); ++i )
        {
            const double gap = sorted[i + 1] - sorted[i];
            if ( gap > bestGap )
            {
                bestGap = gap;
                thetaD = 0.5 * ( sorted[i] + sorted[i + 1] );
            }
        }
        theta = static_cast<Scalar>( thetaD );

        // expected marks: both faces of any edge whose dihedral exceeds theta.
        for ( const auto& e : edges )
            if ( e.ang > thetaD )
            {
                expectedMark[e.fa] = 1;
                expectedMark[e.fb] = 1;
            }
        for ( std::size_t f = 0; f < expectedMark.size(); ++f )
            if ( expectedMark[f] )
            {
                ++refMarked;
                refXor ^= static_cast<unsigned long long>( f );
            }
    }

    // A meaningful fold must mark a nontrivial proper subset of the faces.
    if ( !( refMarked > 0 && refMarked < F0 ) )
        ++fails;

    // ---- distributed run: CurvatureCriterion at the derived threshold -------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, subdiv );
        displaceVertex( mesh, spikeVert, displaceFactor );
        auto faceOwner = facePartitionByAxis( mesh ); // axis=z: fold straddles
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        auto fgid = ownedFaceGidsHost( mesh );
        CurvatureCriterion<Scalar> crit{ theta };
        auto mask = markByQuality( mesh, crit );

        if ( mask.size() != mesh.numOwnedFaces() )
            ++fails;

        int mismatches = 0;
        long long localMarked = 0;
        unsigned long long localXor = 0;
        for ( std::size_t f = 0; f < mask.size(); ++f )
        {
            const GlobalId g = fgid[f];
            const char expect = expectedMark.at( g );
            if ( mask[f] != expect )
                ++mismatches;
            if ( mask[f] )
            {
                ++localMarked;
                localXor ^= static_cast<unsigned long long>( g );
            }
        }
        unsigned long long globalXor = 0;
        MPI_Allreduce( &localXor, &globalXor, 1, MPI_UNSIGNED_LONG_LONG,
                       MPI_BXOR, MPI_COMM_WORLD );
        long long globalMarked = 0;
        MPI_Allreduce( &localMarked, &globalMarked, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );

        if ( mismatches != 0 )
            ++fails; // must match the partition-independent geometric reference
        if ( globalXor != refXor )
            ++fails; // rank-count- + straddle-independent marked-gid checksum
        if ( globalMarked != refMarked )
            ++fails; // exactly the fold faces, no more, no fewer

        auto res = refine( mesh, halo, mask );

        int local = 0;
        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints );
        local += TesseraTest::check21Balance( mesh );
        int glob = 0;
        MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( glob != 0 )
            ++fails;

        if ( rank == 0 )
            std::printf( "  [%s] curvature %s (theta=%g marked=%lld/%lld "
                         "xor=%llx mismatches=%d)\n",
                         tag, fails == 0 ? "ok" : "FAIL",
                         static_cast<double>( theta ), globalMarked, F0,
                         globalXor, mismatches );
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
            std::printf( "test_markquality_curv: curvature/dihedral quality "
                         "refinement marking (size %d)\n",
                         size );

        fails += run<double, Kokkos::Serial>( rank, size, "double/Serial" );
        fails += run<float, Kokkos::Serial>( rank, size, "float/Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
        {
            fails += run<double, Kokkos::DefaultExecutionSpace>(
                rank, size, "double/Default" );
            fails += run<float, Kokkos::DefaultExecutionSpace>(
                rank, size, "float/Default" );
        }
    }

    Kokkos::finalize();

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
