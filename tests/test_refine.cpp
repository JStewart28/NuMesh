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

// Unit test: local single-rank 1->4 red refinement (Step 6a).
//
// Checks:
//   A. Uniform refinement of an icosphere (refine every face) reproduces one
//      subdivision level's topology (V,E,F counts + Euler V-E+F=2), every new
//      midpoint sits at the *plain average* of its edge endpoints (default
//      policy; no sphere projection under AMR), and no midpoint is unit length.
//   B. Partial refinement of a single face inserts exactly three shared
//      midpoints, produces four child faces spanning exactly {corners} U
//      {midpoints}, each parent corner used once and each midpoint shared by
//      more than one sibling child, midpoints at the endpoint average.
//   C. The default policy linearly averages a user vertex field, and a custom
//      per-field policy overrides that one field to a constant while leaving the
//      position (and any other field) on the default rule.
// Runs on host (Serial) and device (default, HIP), single rank.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <type_traits>
#include <vector>

using namespace Tessera;

// Custom policy: force user vertex field 0 (absolute member index UserBegin) to
// a constant, every other field (and position) falls through to the default.
struct ConstFieldPolicy : DefaultRefinePolicy<double>
{
    static constexpr double kConst = 42.0;
    template <std::size_t M>
    double interpolateVertexField( double a, double b ) const
    {
        if constexpr ( M == VertexField::UserBegin + 0 )
            return kConst;
        else
            return DefaultRefinePolicy<double>::template interpolateVertexField<
                M>( a, b );
    }
};

// Expected midpoint (position + optional field average) keyed by the edge's
// structured EdgeKey, built from the pre-refinement mesh.
struct Expected
{
    double pos[3];
    double field; // average of the endpoint user-field values
};

template <bool WithField, class HV>
std::map<EdgeKey, Expected>
build_expected( HV& hv, const std::vector<std::array<int, 3>>& faces )
{
    auto pos = Cabana::slice<VertexField::Position>( hv );
    std::map<EdgeKey, Expected> exp;
    for ( const auto& f : faces )
        for ( int k = 0; k < 3; ++k )
        {
            const int a = f[k];
            const int b = f[( k + 1 ) % 3];
            EdgeKey key = makeEdgeKey( a, b );
            if ( exp.count( key ) )
                continue;
            Expected e;
            for ( int d = 0; d < 3; ++d )
                e.pos[d] = 0.5 * ( static_cast<double>( pos( a, d ) ) +
                                   static_cast<double>( pos( b, d ) ) );
            e.field = 0.0;
            if constexpr ( WithField )
            {
                auto fld = Cabana::slice<VertexField::UserBegin>( hv );
                e.field = 0.5 * ( fld( a ) + fld( b ) );
            }
            exp.emplace( key, e );
        }
    return exp;
}

// Find the expected entry whose position matches p (midpoints are distinct).
const Expected* match_pos( const std::map<EdgeKey, Expected>& exp,
                           const double p[3] )
{
    for ( const auto& kv : exp )
    {
        const Expected& e = kv.second;
        if ( std::abs( e.pos[0] - p[0] ) < 1e-9 &&
             std::abs( e.pos[1] - p[1] ) < 1e-9 &&
             std::abs( e.pos[2] - p[2] ) < 1e-9 )
            return &e;
    }
    return nullptr;
}

// Read the base face->vertex table off the host copy of a mesh.
template <class HF>
std::vector<std::array<int, 3>> face_table( HF& hf, std::size_t nf )
{
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    std::vector<std::array<int, 3>> out( nf );
    for ( std::size_t f = 0; f < nf; ++f )
        for ( int k = 0; k < 3; ++k )
            out[f][k] = static_cast<int>( fv( f, k ) );
    return out;
}

// ---- Case A: uniform refinement reproduces one subdivision level -----------
template <class Exec>
int case_uniform( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 0 ); // V=12 E=30 F=20

    const int nv0 = static_cast<int>( mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv0(
        "hv0", nv0 );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf0(
        "hf0", mesh.numFaces() );
    Cabana::deep_copy( hv0, mesh.vertices() );
    Cabana::deep_copy( hf0, mesh.faces() );
    auto expected =
        build_expected<false>( hv0, face_table( hf0, mesh.numFaces() ) );

    std::vector<char> mask( mesh.numFaces(), 1 );
    refineLocal( mesh, mask );

    int fails = 0;
    const long V = static_cast<long>( mesh.numVertices() );
    const long E = static_cast<long>( mesh.numEdges() );
    const long F = static_cast<long>( mesh.numFaces() );
    if ( V != 42 || E != 120 || F != 80 )
    {
        std::printf( "  [%s] uniform FAIL counts V=%ld E=%ld F=%ld\n", tag, V,
                     E, F );
        ++fails;
    }
    if ( V - E + F != 2 )
    {
        std::printf( "  [%s] uniform FAIL Euler=%ld\n", tag, V - E + F );
        ++fails;
    }
    if ( mesh.numOwnedVertices() != mesh.numVertices() ||
         mesh.numOwnedFaces() != mesh.numFaces() )
        ++fails; // single rank: everything owned

    // Every midpoint (index >= nv0) is the plain endpoint average, interior to
    // the sphere (no projection under AMR).
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    int matched = 0;
    for ( int i = nv0; i < V; ++i )
    {
        const double p[3] = { pos( i, 0 ), pos( i, 1 ), pos( i, 2 ) };
        if ( match_pos( expected, p ) )
            ++matched;
        const double r = std::sqrt( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
        if ( r >= 1.0 - 1e-9 ) // averages of unit vectors are strictly interior
            ++fails;
    }
    if ( matched != static_cast<int>( expected.size() ) || matched != V - nv0 )
        ++fails;

    std::printf( "  [%s] uniform %s (V=%ld E=%ld F=%ld, %d midpoints)\n", tag,
                 fails == 0 ? "ok" : "FAIL", V, E, F, matched );
    return fails;
}

// ---- Case B: single-face refinement -> shared midpoints + 4 children -------
template <class Exec>
int case_partial( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 1 ); // a few faces to leave unrefined neighbours

    const int nv0 = static_cast<int>( mesh.numVertices() );
    const int nf0 = static_cast<int>( mesh.numFaces() );
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv0(
        "hv0", nv0 );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf0(
        "hf0", nf0 );
    Cabana::deep_copy( hv0, mesh.vertices() );
    Cabana::deep_copy( hf0, mesh.faces() );
    auto faces0 = face_table( hf0, nf0 );
    const std::array<int, 3> corners = faces0[0];
    auto expected = build_expected<false>( hv0, { faces0[0] } );

    std::vector<char> mask( nf0, 0 );
    mask[0] = 1;
    refineLocal( mesh, mask );

    int fails = 0;
    if ( static_cast<int>( mesh.numFaces() ) != nf0 - 1 + 4 )
        ++fails; // one face -> four children
    if ( static_cast<int>( mesh.numVertices() ) != nv0 + 3 )
        ++fails; // three new midpoints

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    // The children are exactly the faces referencing a midpoint (index >= nv0).
    std::map<int, int> use; // vertex -> reference count among child faces
    int nChild = 0;
    for ( std::size_t f = 0; f < mesh.numFaces(); ++f )
    {
        bool child = false;
        for ( int k = 0; k < 3; ++k )
            if ( fv( f, k ) >= static_cast<GlobalId>( nv0 ) )
                child = true;
        if ( !child )
            continue;
        ++nChild;
        for ( int k = 0; k < 3; ++k )
            ++use[static_cast<int>( fv( f, k ) )];
    }
    if ( nChild != 4 )
        ++fails;

    // Vertices spanned by the children == 3 corners + 3 midpoints.
    if ( use.size() != 6 )
        ++fails;
    for ( int c : corners )
    {
        if ( use[c] != 1 ) // each parent corner belongs to exactly one child
            ++fails;
    }
    int nMid = 0;
    for ( const auto& kv : use )
    {
        if ( kv.first < nv0 )
            continue;
        ++nMid;
        if ( kv.second < 2 ) // a midpoint is shared by sibling children
            ++fails;
        const double p[3] = { pos( kv.first, 0 ), pos( kv.first, 1 ),
                              pos( kv.first, 2 ) };
        if ( !match_pos( expected, p ) ) // midpoint == endpoint average
            ++fails;
    }
    if ( nMid != 3 )
        ++fails;

    std::printf( "  [%s] partial %s (children=%d midpoints=%d)\n", tag,
                 fails == 0 ? "ok" : "FAIL", nChild, nMid );
    return fails;
}

// ---- Case C: default field averaging vs custom per-field override ----------
template <class Exec>
int case_policy( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<double>, EdgeFields<>,
                       FaceFields<>, mem, Exec>;

    // Build once, snapshot the base topology + a seeded user field.
    auto make = []( MeshT& m )
    {
        buildIcosphere( m, 0 );
        auto fld = m.template vertexSlice<VertexField::UserBegin>();
        Kokkos::parallel_for(
            "seed", Kokkos::RangePolicy<Exec>( 0, m.numVertices() ),
            KOKKOS_LAMBDA( const int i ) {
                fld( i ) = static_cast<double>( i );
            } );
        Kokkos::fence();
    };

    int fails = 0;

    // Default policy: midpoint field == average of endpoint fields.
    {
        MeshT mesh( MPI_COMM_WORLD );
        make( mesh );
        const int nv0 = static_cast<int>( mesh.numVertices() );
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv0( "hv0", nv0 );
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf0(
            "hf0", mesh.numFaces() );
        Cabana::deep_copy( hv0, mesh.vertices() );
        Cabana::deep_copy( hf0, mesh.faces() );
        auto expected =
            build_expected<true>( hv0, face_table( hf0, mesh.numFaces() ) );

        std::vector<char> mask( mesh.numFaces(), 1 );
        refineLocal( mesh, mask ); // default policy

        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", mesh.numVertices() );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto pos = Cabana::slice<VertexField::Position>( hv );
        auto fld = Cabana::slice<VertexField::UserBegin>( hv );
        for ( int i = nv0; i < static_cast<int>( mesh.numVertices() ); ++i )
        {
            const double p[3] = { pos( i, 0 ), pos( i, 1 ), pos( i, 2 ) };
            const Expected* e = match_pos( expected, p );
            if ( !e || std::abs( fld( i ) - e->field ) > 1e-9 )
                ++fails;
        }
    }

    // Custom policy: field 0 forced to a constant; position still averaged.
    {
        MeshT mesh( MPI_COMM_WORLD );
        make( mesh );
        const int nv0 = static_cast<int>( mesh.numVertices() );
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv0( "hv0", nv0 );
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf0(
            "hf0", mesh.numFaces() );
        Cabana::deep_copy( hv0, mesh.vertices() );
        Cabana::deep_copy( hf0, mesh.faces() );
        auto expected =
            build_expected<true>( hv0, face_table( hf0, mesh.numFaces() ) );

        std::vector<char> mask( mesh.numFaces(), 1 );
        refineLocal( mesh, mask, ConstFieldPolicy{} );

        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", mesh.numVertices() );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto pos = Cabana::slice<VertexField::Position>( hv );
        auto fld = Cabana::slice<VertexField::UserBegin>( hv );
        for ( int i = nv0; i < static_cast<int>( mesh.numVertices() ); ++i )
        {
            const double p[3] = { pos( i, 0 ), pos( i, 1 ), pos( i, 2 ) };
            const Expected* e = match_pos( expected, p );
            if ( !e )
                ++fails; // position must still match the default average
            if ( std::abs( fld( i ) - ConstFieldPolicy::kConst ) > 1e-9 )
                ++fails; // field must be the overridden constant
        }
    }

    std::printf( "  [%s] policy %s\n", tag, fails == 0 ? "ok" : "FAIL" );
    return fails;
}

template <class Exec>
int run( const char* tag )
{
    int fails = 0;
    fails += case_uniform<Exec>( tag );
    fails += case_partial<Exec>( tag );
    fails += case_policy<Exec>( tag );
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
            std::printf( "test_refine: local 1->4 red refinement\n" );

        fails += run<Kokkos::Serial>( "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( "Default" );
    }

    Kokkos::finalize();

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
