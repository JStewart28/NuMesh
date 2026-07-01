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

#ifndef TESSERA_ICOSPHERE_HPP
#define TESSERA_ICOSPHERE_HPP

#include "Tessera_Types.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace Tessera
{

// ============================================================================
// Icosphere triangle-soup generator (host, serial, deterministic)
// ============================================================================
//
// Produces a closed genus-0 triangulated unit sphere by recursively 1->4
// subdividing an icosahedron and projecting every vertex onto the unit sphere.
// The result is a *triangle soup*: a flat list of vertex positions and a flat
// list of per-face vertex indices, with shared edge midpoints deduplicated so
// the surface is watertight (each interior edge is shared by exactly two faces).
//
// This is a generation primitive; connectivity (edges, adjacency, CSR 1-ring)
// is derived from the soup by buildFromTriangleSoup() in Tessera_MeshBuilder.hpp.
// It is deterministic and identical on every rank, which is what lets Step 5
// build the same replicated coarse mesh everywhere with no communication.
//
// Entity counts (closed sphere, Euler V - E + F = 2):
//   subdiv 0: V = 12,  E = 30,   F = 20
//   each subdivision: V' = V + E,  E' = 2E + 3F,  F' = 4F
//
// TriangleSoup holds:
//   positions : (num_vertices * 3) scalars, xyz per vertex, |x| == 1.
//   triangles : (num_faces * 3) vertex indices, CCW seen from outside.
template <class Scalar>
struct TriangleSoup
{
    std::vector<Scalar> positions; // size 3 * numVertices
    std::vector<int> triangles;    // size 3 * numFaces

    std::size_t numVertices() const { return positions.size() / 3; }
    std::size_t numFaces() const { return triangles.size() / 3; }
};

namespace detail
{

//! Normalize a 3-vector in place to lie on the unit sphere.
template <class Scalar>
inline void normalize3( Scalar& x, Scalar& y, Scalar& z )
{
    const double inv = 1.0 / std::sqrt( static_cast<double>( x ) * x +
                                        static_cast<double>( y ) * y +
                                        static_cast<double>( z ) * z );
    x = static_cast<Scalar>( x * inv );
    y = static_cast<Scalar>( y * inv );
    z = static_cast<Scalar>( z * inv );
}

} // namespace detail

//! Generate an icosphere triangle soup with `subdivisions` levels of 1->4
//! refinement (subdivisions=0 is the bare icosahedron).
template <class Scalar>
TriangleSoup<Scalar> generateIcosphere( int subdivisions )
{
    TriangleSoup<Scalar> soup;

    // -- base icosahedron: 12 vertices (golden-ratio rectangles) --
    const double t = ( 1.0 + std::sqrt( 5.0 ) ) / 2.0;
    const double base[12][3] = { { -1, t, 0 },  { 1, t, 0 },   { -1, -t, 0 },
                                 { 1, -t, 0 },  { 0, -1, t },  { 0, 1, t },
                                 { 0, -1, -t }, { 0, 1, -t },  { t, 0, -1 },
                                 { t, 0, 1 },   { -t, 0, -1 }, { -t, 0, 1 } };
    for ( int i = 0; i < 12; ++i )
    {
        Scalar x = static_cast<Scalar>( base[i][0] );
        Scalar y = static_cast<Scalar>( base[i][1] );
        Scalar z = static_cast<Scalar>( base[i][2] );
        detail::normalize3( x, y, z );
        soup.positions.push_back( x );
        soup.positions.push_back( y );
        soup.positions.push_back( z );
    }

    // -- base icosahedron: 20 faces (CCW outward) --
    std::vector<int> tris = {
        0, 11, 5,  0, 5,  1, 0, 1, 7, 0, 7,  10, 0, 10, 11, 1, 5, 9, 5, 11,
        4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3,  9,  4, 3,  4,  2, 3, 2, 6, 3,
        6, 8,  3,  8, 9,  4, 9, 5, 2, 4, 11, 6,  2, 10, 8,  6, 7, 9, 8, 1 };

    // -- recursive subdivision with a shared midpoint cache --
    for ( int s = 0; s < subdivisions; ++s )
    {
        std::map<std::pair<int, int>, int> midpoint;
        std::vector<int> out;
        out.reserve( tris.size() * 4 );

        auto midpoint_index = [&]( int a, int b ) -> int
        {
            const std::pair<int, int> key =
                ( a < b ) ? std::make_pair( a, b ) : std::make_pair( b, a );
            auto it = midpoint.find( key );
            if ( it != midpoint.end() )
                return it->second;
            // Create the midpoint vertex, projected to the unit sphere.
            Scalar mx =
                static_cast<Scalar>( 0.5 * ( soup.positions[3 * a + 0] +
                                             soup.positions[3 * b + 0] ) );
            Scalar my =
                static_cast<Scalar>( 0.5 * ( soup.positions[3 * a + 1] +
                                             soup.positions[3 * b + 1] ) );
            Scalar mz =
                static_cast<Scalar>( 0.5 * ( soup.positions[3 * a + 2] +
                                             soup.positions[3 * b + 2] ) );
            detail::normalize3( mx, my, mz );
            const int idx = static_cast<int>( soup.positions.size() / 3 );
            soup.positions.push_back( mx );
            soup.positions.push_back( my );
            soup.positions.push_back( mz );
            midpoint.emplace( key, idx );
            return idx;
        };

        for ( std::size_t f = 0; f < tris.size(); f += 3 )
        {
            const int a = tris[f + 0];
            const int b = tris[f + 1];
            const int c = tris[f + 2];
            const int ab = midpoint_index( a, b );
            const int bc = midpoint_index( b, c );
            const int ca = midpoint_index( c, a );
            const int quad[4][3] = {
                { a, ab, ca }, { b, bc, ab }, { c, ca, bc }, { ab, bc, ca } };
            for ( auto& q : quad )
            {
                out.push_back( q[0] );
                out.push_back( q[1] );
                out.push_back( q[2] );
            }
        }
        tris.swap( out );
    }

    soup.triangles = std::move( tris );
    return soup;
}

} // namespace Tessera

#endif // TESSERA_ICOSPHERE_HPP
