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
#include <stdexcept>
#include <string>
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

// ============================================================================
// Lat/lon (UV) sphere triangle-soup generator (host, serial, deterministic)
// ============================================================================
//
// Why a second generator, when generateIcosphere() already produces a closed
// unit sphere: an icosphere is *too good* a test surface. It is nearly
// isotropic, its triangles are nearly equilateral, and almost every vertex has
// valence 6. A lat/lon sphere is anisotropic by construction -- triangles
// stretch as the rings shrink toward the poles -- and its two pole vertices have
// valence nLon, so it reaches code paths an icosphere never does: quality-based
// marking, cotangent weights at a high-valence vertex, stencil rows of very
// different lengths, and the poles as valence outliers.
//
// This is the same kind of primitive as generateIcosphere(): pure generation,
// handed to buildFromTriangleSoup() for connectivity. A caller could of course
// build this soup itself -- the soup interface exists precisely so it can -- but
// a UV sphere has a handful of easy-to-get-wrong details (duplicate pole
// vertices, a duplicated seam meridian, inconsistent winding), and every
// consumer that writes it writes the same bugs.

//! Generate a unit lat/lon (UV) sphere triangle soup.
//!
//! `nLat` is the number of latitude RINGS **including both poles** (`nLat >= 3`);
//! `nLon` is the number of meridians (`nLon >= 3`). Throws
//! `std::invalid_argument` otherwise.
//!
//! Vertices:
//! \code
//!   theta_j = pi * j / (nLat - 1),  j = 0 .. nLat-1   (0 = north pole)
//!   phi_i   = 2*pi * i / nLon,      i = 0 .. nLon-1   (NOT nLon+1 -- no seam
//!                                                     duplicate; i wraps)
//!   position = ( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) )
//! \endcode
//! The poles (`j = 0` and `j = nLat-1`) are **one vertex each**, written
//! literally as `(0,0,+1)` and `(0,0,-1)` rather than evaluated from the
//! formula: `sin(pi)` is not zero in floating point, so a computed south pole is
//! off the unit sphere in the last bits *and different for different phi*, which
//! is how a duplicate-pole bug hides.
//!
//! Counts (closed surface, `V - E + F = 2`):
//! \code
//!   V = 2 + (nLat - 2) * nLon
//!   F = 2 * nLon * (nLat - 2)        // nLon per pole fan + 2 per interior quad
//!   E = V + F - 2 = 3 * nLon * (nLat - 2)
//! \endcode
//!
//! Ordering: index 0 is the north pole, then ring `j = 1 .. nLat-2` each
//! contributing `nLon` vertices in ascending `i`, then the south pole last.
//! So `ring(j,i) == 1 + (j-1)*nLon + i` and the south pole is `V-1`.
//! Deterministic and documented, so a consumer can address a vertex
//! arithmetically.
//!
//! Winding: CCW seen from OUTSIDE, matching generateIcosphere().
//!
//! Quad diagonal: each interior quad `(i,j)-(i+1,j)-(i+1,j+1)-(i,j+1)` is split
//! along the `(i,j+1)-(i+1,j)` diagonal, the same way around the whole sphere.
//! Fixed, not adaptive -- a caller wanting a different triangulation flips edges.
//!
//! **Reproducibility caveat.** Unlike the icosphere, whose positions come from a
//! rational base table plus `sqrt`, these positions come from `sin`/`cos` at
//! computed angles, which may differ in the last bit across libm
//! implementations and platforms. They are *not* guaranteed bit-reproducible
//! across machines, so a consumer comparing against a gold file generated
//! elsewhere must compare with a tolerance. Within one run they are of course
//! identical on every rank, which is what the replicated coarse build needs.
template <class Scalar>
TriangleSoup<Scalar> generateLatLonSphere( int nLat, int nLon )
{
    if ( nLat < 3 )
        throw std::invalid_argument(
            "Tessera::generateLatLonSphere: nLat (latitude rings INCLUDING "
            "both poles) must be >= 3, got " +
            std::to_string( nLat ) );
    if ( nLon < 3 )
        throw std::invalid_argument(
            "Tessera::generateLatLonSphere: nLon (meridians) must be >= 3, "
            "got " +
            std::to_string( nLon ) );

    // Angles and their sin/cos are computed in double regardless of Scalar and
    // the products cast down, matching detail::normalize3().
    const double pi = 3.14159265358979323846;
    const int nRing = nLat - 2; // interior rings, excluding both poles

    TriangleSoup<Scalar> soup;
    soup.positions.reserve( 3 * static_cast<std::size_t>( 2 + nRing * nLon ) );

    auto push = [&soup]( double x, double y, double z )
    {
        soup.positions.push_back( static_cast<Scalar>( x ) );
        soup.positions.push_back( static_cast<Scalar>( y ) );
        soup.positions.push_back( static_cast<Scalar>( z ) );
    };

    // -- vertices: north pole (exact), interior rings, south pole (exact) -----
    push( 0.0, 0.0, 1.0 );
    for ( int j = 1; j <= nRing; ++j )
    {
        const double theta =
            pi * static_cast<double>( j ) / static_cast<double>( nLat - 1 );
        const double st = std::sin( theta );
        const double ct = std::cos( theta );
        for ( int i = 0; i < nLon; ++i )
        {
            const double phi = 2.0 * pi * static_cast<double>( i ) /
                               static_cast<double>( nLon );
            push( st * std::cos( phi ), st * std::sin( phi ), ct );
        }
    }
    push( 0.0, 0.0, -1.0 );

    const int north = 0;
    const int south = 1 + nRing * nLon;
    // Ring j in [1, nRing], meridian i in [0, nLon) -- i is NOT wrapped here,
    // callers wrap it, so a bad index is a bug rather than a silent alias.
    auto ring = [nLon]( int j, int i ) { return 1 + ( j - 1 ) * nLon + i; };

    soup.triangles.reserve( 3 * static_cast<std::size_t>( 2 * nLon * nRing ) );
    auto tri = [&soup]( int a, int b, int c )
    {
        soup.triangles.push_back( a );
        soup.triangles.push_back( b );
        soup.triangles.push_back( c );
    };

    // -- north pole fan ------------------------------------------------------
    for ( int i = 0; i < nLon; ++i )
        tri( north, ring( 1, i ), ring( 1, ( i + 1 ) % nLon ) );

    // -- interior quads, each cut along the (i,j+1)-(i+1,j) diagonal ---------
    // A = (i,j)  B = (i+1,j)  C = (i+1,j+1)  D = (i,j+1)
    // children (A,D,B) and (D,C,B); both CCW seen from outside.
    for ( int j = 1; j <= nRing - 1; ++j )
        for ( int i = 0; i < nLon; ++i )
        {
            const int ip = ( i + 1 ) % nLon;
            const int A = ring( j, i );
            const int B = ring( j, ip );
            const int C = ring( j + 1, ip );
            const int D = ring( j + 1, i );
            tri( A, D, B );
            tri( D, C, B );
        }

    // -- south pole fan (reversed i order, so the winding is again outward) --
    for ( int i = 0; i < nLon; ++i )
        tri( south, ring( nRing, ( i + 1 ) % nLon ), ring( nRing, i ) );

    return soup;
}

} // namespace Tessera

#endif // TESSERA_ICOSPHERE_HPP
