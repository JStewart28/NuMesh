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

// Regression test: edge-length quality-based refinement marking (Step 10a).
//
// Builds a subdiv-2 icosphere, distributes it, and drives markByQuality() with
// an EdgeLengthCriterion threshold strictly between the coarse mesh's minimum
// and maximum edge length (t = 0.6 * maxEdgeLen). Checks:
//   (a) Correctness + rank-count independence: the marked owned-face gid set,
//       BXOR-reduced across ranks, is compared against a checksum computed
//       from an independent, un-partitioned reference mesh (built once,
//       replicated, before distribute() -- gid == index at that stage). The
//       reference checksum uses no MPI/partition information at all, so
//       agreement proves the marked set does not depend on how the mesh was
//       partitioned.
//   (b) Monotone progress: markByQuality -> refine drops the global max owned
//       edge length strictly below the threshold.
//   (c) refine() post-conditions hold (check21Balance, checkMidpointAgreement)
//       via MeshInvariants.hpp.
//   (d) Degenerate cases: a threshold above the global max edge length yields
//       an empty mask (no-op refine); a threshold below the global min yields
//       a full mask (reproduces one uniform subdivision level, as in the
//       Step-6b uniform test).
// Runs on host (Serial) and device (default, HIP), ranks 1-5, for both double
// and float.

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
#include <limits>
#include <type_traits>
#include <unordered_map>
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

// Global max edge length over every OWNED edge (host; MPI_MAX reduced).
//
// refine() clears the halo (Step 6b): the local mesh holds only owned-first
// entities afterwards, so an owned edge's endpoint may be a vertex this rank
// neither owns nor holds any copy of. In particular a new midpoint's owner
// (min incident *refining-face* owner) and an edge incident to it can be
// owned by a *different* rank (min incident *child-face* owner) -- and
// refine() only ever ships a midpoint's gid to its co-sharers, never its
// position (Tessera_RefineParallel.hpp's Phase-2 `KeyGid` message carries no
// position field). So a purely local (or before/after-snapshotted) vertex
// map is not sufficient in general; missing positions are gathered from
// their true owner via a gid coordinator (`gid % size`), the same idiom
// MeshInvariants.hpp's check21Balance/checkMidpointAgreement use for
// cross-rank edge decisions, applied here to raw position data.
template <class MeshT>
typename MeshT::scalar_type maxOwnedEdgeLength( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();

    const std::size_t nov = mesh.numOwnedVertices();
    const std::size_t noe = mesh.numOwnedEdges();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    auto vgid = Cabana::slice<VertexField::Gid>( hv );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto everts = Cabana::slice<EdgeField::Verts>( he );

    struct PosMsg
    {
        GlobalId gid;
        Scalar pos[Dim];
    };
    struct ReqMsg
    {
        GlobalId gid;
    };

    // Every owned vertex's position, keyed by gid (this rank's authoritative
    // copies -- the only ones guaranteed present post-refine).
    std::unordered_map<GlobalId, std::array<Scalar, Dim>> posByGid;
    posByGid.reserve( nov * 2 );
    for ( std::size_t i = 0; i < nov; ++i )
    {
        std::array<Scalar, Dim> p;
        for ( int d = 0; d < Dim; ++d )
            p[d] = pos( i, d );
        posByGid[vgid( i )] = p;
    }

    // Advertise every owned vertex's position to its gid coordinator.
    std::vector<std::vector<PosMsg>> adv( size );
    for ( const auto& kv : posByGid )
    {
        PosMsg m;
        m.gid = kv.first;
        for ( int d = 0; d < Dim; ++d )
            m.pos[d] = kv.second[d];
        adv[Tessera::detail::gidCoordRank( m.gid, size )].push_back( m );
    }
    auto advGot = allToAllV( comm, adv );
    std::unordered_map<GlobalId, std::array<Scalar, Dim>> coordPos;
    coordPos.reserve( advGot.data.size() * 2 );
    for ( const auto& m : advGot.data )
    {
        std::array<Scalar, Dim> p;
        for ( int d = 0; d < Dim; ++d )
            p[d] = m.pos[d];
        coordPos[m.gid] = p;
    }

    // Request every owned edge endpoint gid not already known locally.
    std::vector<GlobalId> needed;
    for ( std::size_t e = 0; e < noe; ++e )
        for ( int j = 0; j < 2; ++j )
        {
            const GlobalId g = static_cast<GlobalId>( everts( e, j ) );
            if ( posByGid.find( g ) == posByGid.end() )
                needed.push_back( g );
        }
    std::sort( needed.begin(), needed.end() );
    needed.erase( std::unique( needed.begin(), needed.end() ), needed.end() );

    std::vector<std::vector<ReqMsg>> req( size );
    for ( GlobalId g : needed )
        req[Tessera::detail::gidCoordRank( g, size )].push_back( { g } );
    auto reqGot = allToAllV( comm, req );

    // Coordinator replies with the position, grouped back by requester rank.
    std::vector<std::vector<PosMsg>> reply( size );
    for ( int s = 0; s < size; ++s )
    {
        const ReqMsg* p = reqGot.from( s );
        const int cnt = reqGot.count( s );
        for ( int i = 0; i < cnt; ++i )
        {
            PosMsg m;
            m.gid = p[i].gid;
            const auto& pp = coordPos.at( p[i].gid );
            for ( int d = 0; d < Dim; ++d )
                m.pos[d] = pp[d];
            reply[s].push_back( m );
        }
    }
    auto replyGot = allToAllV( comm, reply );
    for ( const auto& m : replyGot.data )
    {
        std::array<Scalar, Dim> p;
        for ( int d = 0; d < Dim; ++d )
            p[d] = m.pos[d];
        posByGid[m.gid] = p;
    }

    Scalar localMax = Scalar( 0 );
    for ( std::size_t e = 0; e < noe; ++e )
    {
        const auto& pa = posByGid.at( static_cast<GlobalId>( everts( e, 0 ) ) );
        const auto& pb = posByGid.at( static_cast<GlobalId>( everts( e, 1 ) ) );
        Scalar lenSq = Scalar( 0 );
        for ( int d = 0; d < Dim; ++d )
        {
            const Scalar diff = pa[d] - pb[d];
            lenSq += diff * diff;
        }
        const Scalar len = std::sqrt( lenSq );
        if ( len > localMax )
            localMax = len;
    }

    double localD = static_cast<double>( localMax );
    double globalD = 0.0;
    MPI_Allreduce( &localD, &globalD, 1, MPI_DOUBLE, MPI_MAX, comm );
    return static_cast<Scalar>( globalD );
}

template <class Scalar, class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // Pinned to the hanging-node mode: the refine() post-conditions asserted
    // below are the 2:1 ones, and the mode is no longer the Mesh default. The
    // conforming counterpart is the markquality_conforming test.
    using MeshT = Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;

    int fails = 0;

    // Coarse subdiv-2 icosphere entity counts (matches the Step-6b fixture).
    const long long V0 = 162, E0 = 480, F0 = 320;

    // ---- independent reference: replicated (un-partitioned) mesh -----------
    // gid == index at this stage (serial builder invariant), so a per-face
    // expected mark can be looked up directly by gid with no partition info.
    Scalar minEdge, maxEdge;
    std::vector<char> expectedMark( static_cast<std::size_t>( F0 ) );
    {
        MeshT ref( MPI_COMM_WORLD );
        buildIcosphere( ref, 2 );
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", ref.numVertices() );
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf", ref.numFaces() );
        Cabana::deep_copy( hv, ref.vertices() );
        Cabana::deep_copy( hf, ref.faces() );
        auto pos = Cabana::slice<VertexField::Position>( hv );
        auto fv = Cabana::slice<FaceField::Verts>( hf );

        minEdge = std::numeric_limits<Scalar>::max();
        maxEdge = Scalar( 0 );
        std::vector<Scalar> faceMaxEdge( ref.numFaces() );
        for ( std::size_t f = 0; f < ref.numFaces(); ++f )
        {
            Scalar fmax = Scalar( 0 );
            for ( int k = 0; k < 3; ++k )
            {
                const int a = static_cast<int>( fv( f, k ) );
                const int b = static_cast<int>( fv( f, ( k + 1 ) % 3 ) );
                Scalar lenSq = Scalar( 0 );
                for ( int d = 0; d < 3; ++d )
                {
                    const Scalar diff = pos( a, d ) - pos( b, d );
                    lenSq += diff * diff;
                }
                const Scalar len = std::sqrt( lenSq );
                if ( len > fmax )
                    fmax = len;
                if ( len < minEdge )
                    minEdge = len;
                if ( len > maxEdge )
                    maxEdge = len;
            }
            faceMaxEdge[f] = fmax;
        }
        const Scalar t = Scalar( 0.6 ) * maxEdge;
        for ( std::size_t f = 0; f < ref.numFaces(); ++f )
            expectedMark[f] = ( faceMaxEdge[f] > t ) ? 1 : 0;
    }
    const Scalar t = Scalar( 0.6 ) * maxEdge;

    // Reference checksum: BXOR of face gids the reference marks, computed with
    // NO MPI and NO partition information at all.
    unsigned long long refXor = 0;
    for ( std::size_t f = 0; f < expectedMark.size(); ++f )
        if ( expectedMark[f] )
            refXor ^= static_cast<unsigned long long>( f );

    // ---- distributed run: main nontrivial threshold -------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        auto fgid = ownedFaceGidsHost( mesh );
        auto mask = markByQuality( mesh, t );

        if ( mask.size() != mesh.numOwnedFaces() )
            ++fails;

        int mismatches = 0;
        unsigned long long localXor = 0;
        for ( std::size_t f = 0; f < mask.size(); ++f )
        {
            const GlobalId g = fgid[f];
            const char expect = expectedMark.at( g );
            if ( mask[f] != expect )
                ++mismatches;
            if ( mask[f] )
                localXor ^= static_cast<unsigned long long>( g );
        }
        unsigned long long globalXor = 0;
        MPI_Allreduce( &localXor, &globalXor, 1, MPI_UNSIGNED_LONG_LONG,
                       MPI_BXOR, MPI_COMM_WORLD );

        if ( mismatches != 0 )
            ++fails; // marking must match the partition-independent reference
        if ( globalXor != refXor )
            ++fails; // rank-count-independent marked-gid checksum

        auto res = refine( mesh, halo, mask );

        int local = 0;
        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints );
        local += TesseraTest::check21Balance( mesh );
        int glob = 0;
        MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( glob != 0 )
            ++fails;

        const Scalar newMax = maxOwnedEdgeLength( mesh );
        if ( !( newMax < t ) )
            ++fails; // monotone progress: previously-over-length edges gone

        if ( rank == 0 )
            std::printf( "  [%s] threshold %s (xor=%llx mismatches=%d "
                         "newMax=%g t=%g)\n",
                         tag, fails == 0 ? "ok" : "FAIL", globalXor, mismatches,
                         static_cast<double>( newMax ),
                         static_cast<double>( t ) );
    }

    // ---- degenerate: threshold above global max -> empty mask, no-op -------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        const Scalar tHigh = maxEdge * Scalar( 10 );
        auto mask = markByQuality( mesh, tHigh );
        bool anyMarked = false;
        for ( char m : mask )
            if ( m )
                anyMarked = true;
        if ( anyMarked )
            ++fails;

        const long long preV = TesseraTest::globalOwnedVertices( mesh );
        const long long preE = TesseraTest::globalOwnedEdges( mesh );
        const long long preF = TesseraTest::globalOwnedFaces( mesh );

        refine( mesh, halo, mask );

        const long long postV = TesseraTest::globalOwnedVertices( mesh );
        const long long postE = TesseraTest::globalOwnedEdges( mesh );
        const long long postF = TesseraTest::globalOwnedFaces( mesh );
        if ( postV != preV || postE != preE || postF != preF )
            ++fails; // empty mask must be a no-op

        if ( rank == 0 )
            std::printf( "  [%s] empty-mask %s\n", tag,
                         !anyMarked && postV == preV ? "ok" : "FAIL" );
    }

    // ---- degenerate: threshold below global min -> full mask, uniform ------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        const Scalar tLow = minEdge * Scalar( 0.1 );
        auto mask = markByQuality( mesh, tLow );
        bool allMarked = true;
        for ( char m : mask )
            if ( !m )
                allMarked = false;
        if ( !allMarked )
            ++fails;

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

        if ( rank == 0 )
            std::printf(
                "  [%s] full-mask %s (V=%lld E=%lld F=%lld euler=%lld)\n", tag,
                ( glob == 0 && allMarked ) ? "ok" : "FAIL", gV, gE, gF, euler );
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
            std::printf( "test_markquality_edge: edge-length quality "
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
