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

// Unit test: the serial closure kernel (Task 2 of the conforming-refinement
// plan) -- closeFaces / unclose / translateMask and the RefinementMode::
// Conforming path of refineLocal().
//
// Cases:
//
//   A. PATTERNS. A hand-built red parent, each of the four |S| cases, compared
//      against the child triangles written out by hand. Pins:
//        * the exact green / blue / red-closure triangulations;
//        * winding: every child is CCW in the parent's orientation (checked with
//          a planar embedding of the parent, where CCW == positive signed area);
//        * the blue lower-gid diagonal tie-break, exercised in BOTH directions
//          by relabelling the two midpoint gids;
//        * rotation invariance: cyclically relabelling the parent's corners (the
//          same triangle, a different starting corner) yields the SAME set of
//          child triangles. This is why the tie-break is partition-independent:
//          the output is a function of the triangle and the (globally agreed)
//          midpoint gids alone, not of any local ordering;
//        * every child carries the parent's level and parent bookkeeping, and
//          the |S| = 3 red-closure does NOT increment the level;
//        * unclose() restores the parent exactly from any single child.
//
//   B. INVERSE. unclose o close == identity on the red layer, over a random
//      partial mask on buildIcosphere(3). Also pins that unclose() takes the
//      user-field source from the LOWEST-GID child of each parent.
//
//   C. refineLocal(). A partial mask in Conforming mode leaves a CONFORMING
//      mesh: owned Euler V-E+F == 2, every edge with exactly two incident faces,
//      no vertex in the interior of an edge, all faceNormalRaw outward, and the
//      same vertex count as HangingNode2to1 mode (the closure adds no vertices).
//      The same three checks are run on the HangingNode2to1 result with the same
//      mask and must FAIL there -- otherwise the mask is too weak to produce a
//      hanging node and the case would be vacuous.
//
// Runs on host (Serial) and the default execution space (HIP on Tuolumne),
// single rank -- the closure is local by construction.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

using namespace Tessera;

// ===========================================================================
// Case A -- closure patterns
// ===========================================================================

// A planar embedding of the hand-built parent, so "CCW" is checkable by signed
// area. Corner gids 10,20,30 sit at the corners; midpoint gids at the midpoints.
struct PlanarParent
{
    std::map<GlobalId, std::array<double, 2>> xy;
    PlanarParent( GlobalId a, GlobalId b, GlobalId c, GlobalId mab,
                  GlobalId mbc, GlobalId mca )
    {
        xy[a] = { 0.0, 0.0 };
        xy[b] = { 1.0, 0.0 };
        xy[c] = { 0.0, 1.0 };
        xy[mab] = { 0.5, 0.0 };
        xy[mbc] = { 0.5, 0.5 };
        xy[mca] = { 0.0, 0.5 };
    }
    double signedArea( const GlobalId v[3] ) const
    {
        const auto& p0 = xy.at( v[0] );
        const auto& p1 = xy.at( v[1] );
        const auto& p2 = xy.at( v[2] );
        return 0.5 * ( ( p1[0] - p0[0] ) * ( p2[1] - p0[1] ) -
                       ( p1[1] - p0[1] ) * ( p2[0] - p0[0] ) );
    }
};

// Order-independent identity of a triangle *as a triangle*: the sorted corner
// gid triple. Comparing sets of these compares topology, not emission order.
std::set<FaceKey> triangleSet( const std::vector<VisibleFace>& v )
{
    std::set<FaceKey> s;
    for ( const auto& f : v )
        s.insert( makeFaceKey( f.v[0], f.v[1], f.v[2] ) );
    return s;
}

// Emission-order-sensitive comparison against a hand-written expectation.
int expectChildren( const char* what, const std::vector<VisibleFace>& got,
                    const std::vector<std::array<GlobalId, 3>>& want )
{
    if ( got.size() != want.size() )
    {
        std::printf( "    %s FAIL child count %zu != %zu\n", what, got.size(),
                     want.size() );
        return 1;
    }
    int fails = 0;
    for ( std::size_t i = 0; i < want.size(); ++i )
        for ( int k = 0; k < 3; ++k )
            if ( got[i].v[k] != want[i][k] )
            {
                std::printf( "    %s FAIL child %zu corner %d: %llu != %llu\n",
                             what, i, k,
                             static_cast<unsigned long long>( got[i].v[k] ),
                             static_cast<unsigned long long>( want[i][k] ) );
                ++fails;
            }
    return fails;
}

// Close one hand-built parent and check the shared per-child contract.
struct OnePattern
{
    CloseResult cl;
    int fails = 0;
};

OnePattern closeOne( const char* what, const RedFace& parent,
                     const std::map<EdgeKey, GlobalId>& midpointOf,
                     GlobalId firstChildGid, int expectSplit,
                     const PlanarParent* planar )
{
    OnePattern out;
    out.cl = closeFaces( { parent }, midpointOf, firstChildGid );

    if ( out.cl.stats.patternCount[expectSplit] != 1 )
    {
        std::printf( "    %s FAIL |S| histogram\n", what );
        ++out.fails;
    }
    if ( static_cast<int>( out.cl.visible.size() ) !=
         closureChildCount( expectSplit ) )
    {
        std::printf( "    %s FAIL closureChildCount\n", what );
        ++out.fails;
    }

    GlobalId nextGid = firstChildGid;
    for ( const VisibleFace& f : out.cl.visible )
    {
        // Level is the PARENT's -- the |S| = 3 red-closure must not increment it
        // and must not promote the face into the red layer.
        if ( f.level != parent.level )
        {
            std::printf( "    %s FAIL child level %d != parent %d\n", what,
                         static_cast<int>( f.level ),
                         static_cast<int>( parent.level ) );
            ++out.fails;
        }
        if ( expectSplit == 0 )
        {
            // Passed through: keeps its own gid, is NOT a closure child.
            if ( f.gid != parent.gid || f.parent != invalid_gid )
                ++out.fails;
            for ( int k = 0; k < 3; ++k )
                if ( f.parentVerts[k] != invalid_gid )
                    ++out.fails;
        }
        else
        {
            if ( f.gid != nextGid++ )
            {
                std::printf(
                    "    %s FAIL child gid not consecutive from %llu\n", what,
                    static_cast<unsigned long long>( firstChildGid ) );
                ++out.fails;
            }
            if ( f.parent != parent.gid )
                ++out.fails;
            for ( int k = 0; k < 3; ++k )
                if ( f.parentVerts[k] != parent.v[k] )
                    ++out.fails;
        }
        if ( planar && planar->signedArea( f.v ) <= 0.0 )
        {
            std::printf( "    %s FAIL child winding not CCW (signed area %g)\n",
                         what, planar->signedArea( f.v ) );
            ++out.fails;
        }
    }

    // unclose() must restore the parent exactly, from any single child.
    const UncloseResult un = unclose( out.cl.visible );
    if ( un.red.size() != 1 )
    {
        std::printf( "    %s FAIL unclose produced %zu red faces\n", what,
                     un.red.size() );
        ++out.fails;
    }
    else
    {
        const RedFace& r = un.red[0];
        if ( r.gid != parent.gid || r.level != parent.level )
            ++out.fails;
        for ( int k = 0; k < 3; ++k )
            if ( r.v[k] != parent.v[k] )
                ++out.fails;
        // Every child maps back to the single red face.
        for ( int ri : un.redOfVisible )
            if ( ri != 0 )
                ++out.fails;
    }
    return out;
}

int case_patterns( const char* tag )
{
    int fails = 0;

    const GlobalId a = 10, b = 20, c = 30;
    const GlobalId mab = 101, mbc = 102, mca = 103;
    const GlobalId kFirstChild = 5000;

    RedFace parent;
    parent.v[0] = a;
    parent.v[1] = b;
    parent.v[2] = c;
    parent.gid = 7;
    parent.level = 2;

    const PlanarParent planar( a, b, c, mab, mbc, mca );

    // ---- |S| = 0: emitted unchanged ---------------------------------------
    {
        auto r = closeOne( "|S|=0", parent, {}, kFirstChild, 0, &planar );
        fails += r.fails;
        fails += expectChildren( "|S|=0", r.cl.visible, { { a, b, c } } );
        if ( r.cl.stats.nClosureChildren != 0 )
            ++fails; // a passthrough is not a closure child
    }

    // ---- |S| = 1 (green), each of the three edges in turn -----------------
    // Rotating so the split edge is edge 0 of (A,B,C) gives (A,m,C),(m,B,C).
    {
        auto r = closeOne( "green/ab", parent, { { makeEdgeKey( a, b ), mab } },
                           kFirstChild, 1, &planar );
        fails += r.fails;
        fails += expectChildren( "green/ab", r.cl.visible,
                                 { { a, mab, c }, { mab, b, c } } );
    }
    {
        auto r = closeOne( "green/bc", parent, { { makeEdgeKey( b, c ), mbc } },
                           kFirstChild, 1, &planar );
        fails += r.fails;
        fails += expectChildren( "green/bc", r.cl.visible,
                                 { { b, mbc, a }, { mbc, c, a } } );
    }
    {
        auto r = closeOne( "green/ca", parent, { { makeEdgeKey( c, a ), mca } },
                           kFirstChild, 1, &planar );
        fails += r.fails;
        fails += expectChildren( "green/ca", r.cl.visible,
                                 { { c, mca, b }, { mca, a, b } } );
    }

    // ---- |S| = 2 (blue): both diagonals ------------------------------------
    // Unsplit edge (c,a) => (A,B,C) = (a,b,c), q0 = mid(a,b), q1 = mid(b,c).
    // q0 < q1 connects q0 to its opposite corner C = c.
    {
        auto r = closeOne(
            "blue/lowfirst", parent,
            { { makeEdgeKey( a, b ), mab }, { makeEdgeKey( b, c ), mbc } },
            kFirstChild, 2, &planar );
        fails += r.fails;
        fails += expectChildren(
            "blue/lowfirst", r.cl.visible,
            { { a, mab, c }, { mab, b, mbc }, { mab, mbc, c } } );
        if ( r.cl.stats.nBlueDiagLowFirst != 1 ||
             r.cl.stats.nBlueDiagLowSecond != 0 )
            ++fails;
    }
    // Same geometry, midpoint gids swapped so q1 < q0: the diagonal must flip to
    // q1 <-> A. This is the ONLY thing that distinguishes a correct tie-break
    // from one that always picks the first diagonal.
    {
        const GlobalId lo = 101, hi = 102; // mid(a,b) = hi, mid(b,c) = lo
        const PlanarParent pl( a, b, c, hi, lo, mca );
        auto r = closeOne(
            "blue/lowsecond", parent,
            { { makeEdgeKey( a, b ), hi }, { makeEdgeKey( b, c ), lo } },
            kFirstChild, 2, &pl );
        fails += r.fails;
        fails +=
            expectChildren( "blue/lowsecond", r.cl.visible,
                            { { a, hi, lo }, { hi, b, lo }, { a, lo, c } } );
        if ( r.cl.stats.nBlueDiagLowFirst != 0 ||
             r.cl.stats.nBlueDiagLowSecond != 1 )
            ++fails;
    }

    // ---- |S| = 3 (red-closure): the 1->4 convention, level UNCHANGED -------
    {
        auto r = closeOne( "redclosure", parent,
                           { { makeEdgeKey( a, b ), mab },
                             { makeEdgeKey( b, c ), mbc },
                             { makeEdgeKey( c, a ), mca } },
                           kFirstChild, 3, &planar );
        fails += r.fails;
        fails += expectChildren( "redclosure", r.cl.visible,
                                 { { a, mab, mca },
                                   { b, mbc, mab },
                                   { c, mca, mbc },
                                   { mab, mbc, mca } } );
    }

    // ---- rotation invariance ------------------------------------------------
    // A cyclic relabel of the parent's corners is the SAME triangle with a
    // different starting corner. Every pattern rotates its working triple by the
    // split/unsplit edge index rather than by v[0], so the emitted triangle SET
    // must be identical. This is the property that makes the closure a function
    // of global data only.
    {
        const std::vector<std::map<EdgeKey, GlobalId>> maps = {
            {},
            { { makeEdgeKey( a, b ), mab } },
            { { makeEdgeKey( a, b ), mab }, { makeEdgeKey( b, c ), mbc } },
            { { makeEdgeKey( a, b ), mab },
              { makeEdgeKey( b, c ), mbc },
              { makeEdgeKey( c, a ), mca } } };
        const char* names[4] = { "rot|S|=0", "rot|S|=1", "rot|S|=2",
                                 "rot|S|=3" };
        for ( int s = 0; s < 4; ++s )
        {
            std::set<FaceKey> ref;
            for ( int rot = 0; rot < 3; ++rot )
            {
                RedFace p = parent;
                p.v[0] = parent.v[rot];
                p.v[1] = parent.v[( rot + 1 ) % 3];
                p.v[2] = parent.v[( rot + 2 ) % 3];
                const auto cl = closeFaces( { p }, maps[s], kFirstChild );
                const auto got = triangleSet( cl.visible );
                if ( rot == 0 )
                    ref = got;
                else if ( got != ref )
                {
                    std::printf( "    %s FAIL rotation %d changed the child "
                                 "triangle set\n",
                                 names[s], rot );
                    ++fails;
                }
            }
        }
    }

    // ---- translateMask: a parent is marked iff ANY child was ---------------
    {
        const auto cl = closeFaces( { parent },
                                    { { makeEdgeKey( a, b ), mab },
                                      { makeEdgeKey( b, c ), mbc },
                                      { makeEdgeKey( c, a ), mca } },
                                    kFirstChild );
        const UncloseResult un = unclose( cl.visible );
        for ( int which = 0; which < 4; ++which )
        {
            std::vector<char> vm( cl.visible.size(), 0 );
            vm[which] = 1;
            const auto rm = translateMask( vm, un );
            if ( rm.size() != 1 || rm[0] != 1 )
                ++fails; // any single marked child marks the parent
        }
        const auto none =
            translateMask( std::vector<char>( cl.visible.size(), 0 ), un );
        if ( none.size() != 1 || none[0] != 0 )
            ++fails; // and no marked child marks nothing
    }

    std::printf( "  [%s] patterns %s\n", tag, fails == 0 ? "ok" : "FAIL" );
    return fails;
}

// ===========================================================================
// Case B -- unclose o close == identity on the red layer
// ===========================================================================

// Deterministic LCG, so the "random" mask is reproducible across runs, ranks,
// and backends.
struct Lcg
{
    unsigned long long s;
    explicit Lcg( unsigned long long seed )
        : s( seed )
    {
    }
    unsigned next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<unsigned>( s >> 33 );
    }
};

template <class Exec>
int case_inverse( const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::Conforming>;
    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );

    const int nv = static_cast<int>( mesh.numVertices() );
    const int nf = static_cast<int>( mesh.numFaces() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hf, mesh.faces() );

    // The base mesh is entirely red (buildIcosphere calls
    // initClosureFaceMembers), so its visible layer IS its red layer.
    const auto visible0 = readVisibleFaces<MeshT>( hf, nf );
    const UncloseResult un0 = unclose( visible0 );
    int fails = 0;
    if ( static_cast<int>( un0.red.size() ) != nf )
    {
        std::printf( "    inverse FAIL a freshly built mesh is not all-red "
                     "(%zu red faces from %d visible) -- were the closure "
                     "bookkeeping members initialized?\n",
                     un0.red.size(), nf );
        return fails + 1; // the rest of this case indexes un0.red by face
    }

    // Build a red layer by hand: a random partial 1->4 split of the base faces.
    // Every base face is at level 0, so a single split level bisects each edge at
    // most once and the 2:1 closure precondition holds.
    Lcg rng( 0x5eed1234ULL );
    std::map<EdgeKey, GlobalId> midpointOf;
    GlobalId nextMid = static_cast<GlobalId>( nv );
    auto midpoint = [&]( GlobalId x, GlobalId y ) -> GlobalId
    {
        const EdgeKey k = makeEdgeKey( x, y );
        auto it = midpointOf.find( k );
        if ( it != midpointOf.end() )
            return it->second;
        const GlobalId g = nextMid++;
        midpointOf.emplace( k, g );
        return g;
    };

    GlobalId nextGid = static_cast<GlobalId>( nf );
    std::vector<RedFace> red;
    int nMarked = 0;
    for ( int f = 0; f < nf; ++f )
    {
        const RedFace& p = un0.red[f];
        if ( ( rng.next() % 100 ) >= 35 ) // ~35% of faces refine
        {
            red.push_back( p );
            continue;
        }
        ++nMarked;
        const GlobalId ab = midpoint( p.v[0], p.v[1] );
        const GlobalId bc = midpoint( p.v[1], p.v[2] );
        const GlobalId ca = midpoint( p.v[2], p.v[0] );
        const std::array<GlobalId, 3> ch[4] = { { p.v[0], ab, ca },
                                                { p.v[1], bc, ab },
                                                { p.v[2], ca, bc },
                                                { ab, bc, ca } };
        for ( const auto& q : ch )
        {
            RedFace r;
            for ( int k = 0; k < 3; ++k )
                r.v[k] = q[k];
            r.gid = nextGid++;
            r.level = 1;
            red.push_back( r );
        }
    }

    // close -> unclose must reproduce the red layer bit-for-bit.
    const GlobalId firstChild = nextGid;
    const std::size_t predicted = countClosureChildren( red, midpointOf );
    const CloseResult cl = closeFaces( red, midpointOf, firstChild );
    const UncloseResult un = unclose( cl.visible );

    if ( static_cast<std::size_t>( cl.stats.nClosureChildren ) != predicted )
    {
        std::printf(
            "    inverse FAIL countClosureChildren %zu != emitted %d\n",
            predicted, cl.stats.nClosureChildren );
        ++fails;
    }
    if ( un.red.size() != red.size() )
    {
        std::printf( "    inverse FAIL red count %zu != %zu\n", un.red.size(),
                     red.size() );
        ++fails;
    }
    else
    {
        // Compare as gid-keyed sets: unclose() restores in first-encounter order
        // over the visible input, which need not match the input red order.
        std::map<GlobalId, RedFace> want, got;
        for ( const RedFace& r : red )
            want.emplace( r.gid, r );
        for ( const RedFace& r : un.red )
            got.emplace( r.gid, r );
        if ( want.size() != red.size() || got.size() != un.red.size() )
            ++fails; // duplicate gids on either side
        for ( const auto& kv : want )
        {
            auto it = got.find( kv.first );
            if ( it == got.end() )
            {
                ++fails;
                continue;
            }
            if ( it->second.level != kv.second.level )
                ++fails;
            for ( int k = 0; k < 3; ++k )
                if ( it->second.v[k] != kv.second.v[k] )
                    ++fails;
        }
    }

    // The user-field source of each restored red face must be its LOWEST-GID
    // child (or itself, for a passthrough).
    for ( std::size_t r = 0; r < un.red.size(); ++r )
    {
        GlobalId lowest = invalid_gid;
        for ( std::size_t i = 0; i < cl.visible.size(); ++i )
            if ( un.redOfVisible[i] == static_cast<int>( r ) )
                lowest = std::min( lowest, cl.visible[i].gid );
        if ( cl.visible[un.sourceVisible[r]].gid != lowest )
            ++fails;
    }

    // A mask over the visible layer must translate to exactly the set of red
    // faces owning a marked visible face.
    {
        std::vector<char> vm( cl.visible.size(), 0 );
        std::set<int> expect;
        Lcg r2( 0xabcdef01ULL );
        for ( std::size_t i = 0; i < vm.size(); ++i )
            if ( r2.next() % 4 == 0 )
            {
                vm[i] = 1;
                expect.insert( un.redOfVisible[i] );
            }
        const auto rm = translateMask( vm, un );
        for ( std::size_t r = 0; r < rm.size(); ++r )
            if ( ( rm[r] != 0 ) !=
                 ( expect.count( static_cast<int>( r ) ) != 0 ) )
                ++fails;
    }

    std::printf( "  [%s] inverse %s (red=%zu marked=%d visible=%d closure=%d "
                 "|S| hist=[%d,%d,%d,%d] blue diag lo1/lo2=%d/%d)\n",
                 tag, fails == 0 ? "ok" : "FAIL", red.size(), nMarked,
                 cl.stats.nVisible, cl.stats.nClosureChildren,
                 cl.stats.patternCount[0], cl.stats.patternCount[1],
                 cl.stats.patternCount[2], cl.stats.patternCount[3],
                 cl.stats.nBlueDiagLowFirst, cl.stats.nBlueDiagLowSecond );
    return fails;
}

// ===========================================================================
// Case C -- refineLocal() in Conforming mode leaves a conforming mesh
// ===========================================================================

struct Topo
{
    long long V = 0, E = 0, F = 0;
    //! Edges (derived from the face table) whose incident-face count != 2.
    int badIncidence = 0;
    //! Edges with a vertex strictly inside them -- i.e. T-junctions.
    int interiorVerts = 0;
    //! Faces whose raw normal points inward (winding not outward-consistent).
    int inwardNormals = 0;
    long long euler() const { return V - E + F; }
};

// Derive the edge table from the face table, count incidences, and look for a
// vertex in the interior of any edge. A hanging node on edge (u,w) is a vertex
// adjacent to both u and w that is COLLINEAR with and strictly between them --
// the purely topological reading ("some m has edges (u,m) and (m,w)") is true of
// every ordinary triangle, so the test must be geometric.
template <class MeshT>
Topo analyzeTopology( MeshT& mesh )
{
    Topo t;
    const std::size_t nv = mesh.numVertices();
    const std::size_t nf = mesh.numFaces();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    t.V = static_cast<long long>( mesh.numOwnedVertices() );
    t.E = static_cast<long long>( mesh.numOwnedEdges() );
    t.F = static_cast<long long>( mesh.numOwnedFaces() );

    // gid -> local index (gid == index for vertices in the serial path, but do
    // not rely on it).
    std::map<GlobalId, int> lv;
    {
        auto vg = Cabana::slice<VertexField::Gid>( hv );
        for ( std::size_t i = 0; i < nv; ++i )
            lv.emplace( vg( i ), static_cast<int>( i ) );
    }

    std::map<EdgeKey, int> incidence;
    for ( std::size_t f = 0; f < nf; ++f )
        for ( int k = 0; k < 3; ++k )
            ++incidence[makeEdgeKey( fv( f, k ), fv( f, ( k + 1 ) % 3 ) )];
    for ( const auto& kv : incidence )
        if ( kv.second != 2 )
            ++t.badIncidence;

    // vertex -> neighbours, from the derived edge set.
    std::map<GlobalId, std::vector<GlobalId>> nbr;
    for ( const auto& kv : incidence )
    {
        nbr[kv.first.id[0]].push_back( kv.first.id[1] );
        nbr[kv.first.id[1]].push_back( kv.first.id[0] );
    }
    for ( auto& kv : nbr )
        std::sort( kv.second.begin(), kv.second.end() );

    for ( const auto& kv : incidence )
    {
        const GlobalId u = kv.first.id[0];
        const GlobalId w = kv.first.id[1];
        const int iu = lv.at( u );
        const int iw = lv.at( w );
        double d[3], len2 = 0.0;
        for ( int k = 0; k < 3; ++k )
        {
            d[k] = pos( iw, k ) - pos( iu, k );
            len2 += d[k] * d[k];
        }
        const auto& nu = nbr[u];
        const auto& nw = nbr[w];
        std::vector<GlobalId> common;
        std::set_intersection( nu.begin(), nu.end(), nw.begin(), nw.end(),
                               std::back_inserter( common ) );
        for ( GlobalId m : common )
        {
            const int im = lv.at( m );
            double e[3], proj = 0.0;
            for ( int k = 0; k < 3; ++k )
            {
                e[k] = pos( im, k ) - pos( iu, k );
                proj += e[k] * d[k];
            }
            const double cx = e[1] * d[2] - e[2] * d[1];
            const double cy = e[2] * d[0] - e[0] * d[2];
            const double cz = e[0] * d[1] - e[1] * d[0];
            const double area2 = cx * cx + cy * cy + cz * cz;
            // Collinear with (u,w) AND strictly between the endpoints.
            if ( area2 <= 1e-20 * len2 * len2 && proj > 1e-12 * len2 &&
                 proj < ( 1.0 - 1e-12 ) * len2 )
                ++t.interiorVerts;
        }
    }

    // Winding: the icosphere and every midpoint-averaged refinement of it stay
    // star-shaped about the origin, so an outward normal has a positive dot
    // product with the face centroid. Evaluated through faceNormalRaw on the
    // mesh's own execution space, as the acceptance criterion specifies.
    {
        auto geom = buildMeshGeometry( mesh );
        int inward = 0;
        Kokkos::parallel_reduce(
            "winding",
            Kokkos::RangePolicy<typename MeshT::execution_space>( 0, nf ),
            KOKKOS_LAMBDA( const int f, int& acc ) {
                typename MeshT::scalar_type n[3];
                faceNormalRaw( geom, f, n );
                typename MeshT::scalar_type dot = 0;
                for ( int k = 0; k < 3; ++k )
                {
                    const auto ctr =
                        ( geom.pos( geom.faceVerts( f, 0 ), k ) +
                          geom.pos( geom.faceVerts( f, 1 ), k ) +
                          geom.pos( geom.faceVerts( f, 2 ), k ) ) /
                        static_cast<typename MeshT::scalar_type>( 3 );
                    dot += n[k] * ctr;
                }
                if ( !( dot > 0 ) )
                    ++acc;
            },
            inward );
        Kokkos::fence();
        t.inwardNormals = inward;
    }
    return t;
}

// The mask must actually create a level jump, or case C proves nothing: refine
// every third face of an icosphere(2).
inline std::vector<char> adaptiveMask( std::size_t nf )
{
    std::vector<char> m( nf, 0 );
    for ( std::size_t f = 0; f < nf; f += 3 )
        m[f] = 1;
    return m;
}

// Seed face user field 0 to the face's own local index, so a closure child's
// inherited value names the base face it descends from.
template <class Exec, class MeshT>
void seedFaceField( MeshT& mesh )
{
    auto u = mesh.template faceSlice<userFaceField<0>()>();
    Kokkos::parallel_for(
        "seed_faces", Kokkos::RangePolicy<Exec>( 0, mesh.numFaces() ),
        KOKKOS_LAMBDA( const int f ) { u( f ) = static_cast<double>( f ); } );
    Kokkos::fence();
}

template <class Exec>
int case_refine_local( const char* tag )
{
    using mem = typename Exec::memory_space;
    using VF = VertexFields<>;
    using EF = EdgeFields<>;
    using FF = FaceFields<double>; // exercise face user-field inheritance
    using MeshH =
        Mesh<double, 3, VF, EF, FF, mem, Exec, RefinementMode::HangingNode2to1>;
    using MeshC =
        Mesh<double, 3, VF, EF, FF, mem, Exec, RefinementMode::Conforming>;

    int fails = 0;

    // ---- hanging-node reference (and the non-vacuity guard) ----------------
    Topo th;
    long long nvH = 0;
    {
        MeshH mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        seedFaceField<Exec>( mesh );
        refineLocal( mesh, adaptiveMask( mesh.numFaces() ) );
        th = analyzeTopology( mesh );
        nvH = static_cast<long long>( mesh.numVertices() );
    }
    if ( th.badIncidence == 0 || th.interiorVerts == 0 || th.euler() == 2 )
    {
        std::printf(
            "    refineLocal FAIL the mask produces no hanging node in "
            "HangingNode2to1 mode (Euler=%lld bad=%d interior=%d) -- "
            "the conforming check below would be vacuous\n",
            th.euler(), th.badIncidence, th.interiorVerts );
        ++fails;
    }

    // ---- conforming ---------------------------------------------------------
    Topo tc;
    long long nvC = 0;
    int nClosure = 0;
    {
        MeshC mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        seedFaceField<Exec>( mesh );
        const std::size_t nf0 = mesh.numFaces();
        refineLocal( mesh, adaptiveMask( nf0 ) );
        tc = analyzeTopology( mesh );
        nvC = static_cast<long long>( mesh.numVertices() );

        // Face user fields: every closure child inherited its parent's value, so
        // all children of one parent agree, and every value is one of the base
        // faces' seeded values.
        const std::size_t nf = mesh.numFaces();
        Cabana::AoSoA<typename MeshC::face_member_types, Kokkos::HostSpace> hf(
            "hf", nf );
        Cabana::deep_copy( hf, mesh.faces() );
        auto u = Cabana::slice<userFaceField<0>()>( hf );
        auto cp = Cabana::slice<MeshC::closure_parent_field>( hf );
        std::map<GlobalId, double> byParent;
        for ( std::size_t f = 0; f < nf; ++f )
        {
            const double val = u( f );
            if ( !( val >= 0.0 && val < static_cast<double>( nf0 ) &&
                    val == std::floor( val ) ) )
                ++fails; // not one of the seeded base-face values
            if ( cp( f ) == invalid_gid )
                continue;
            ++nClosure;
            auto it = byParent.find( cp( f ) );
            if ( it == byParent.end() )
                byParent.emplace( cp( f ), val );
            else if ( it->second != val )
                ++fails; // siblings disagree: user fields were not inherited
        }
        if ( nClosure == 0 )
        {
            std::printf(
                "    refineLocal FAIL no closure faces were emitted\n" );
            ++fails;
        }
        // The visible layer must un-close back to a consistent red layer (this
        // also exercises the duplicate-red-gid guard inside unclose()).
        const UncloseResult un = unclose( readVisibleFaces<MeshC>( hf, nf ) );
        if ( un.red.size() >= nf )
            ++fails; // closure children must collapse
    }

    if ( tc.euler() != 2 )
    {
        std::printf( "    refineLocal FAIL conforming Euler V-E+F = %lld "
                     "(V=%lld E=%lld F=%lld)\n",
                     tc.euler(), tc.V, tc.E, tc.F );
        ++fails;
    }
    if ( tc.badIncidence != 0 )
    {
        std::printf( "    refineLocal FAIL %d edges without exactly two "
                     "incident faces\n",
                     tc.badIncidence );
        ++fails;
    }
    if ( tc.interiorVerts != 0 )
    {
        std::printf( "    refineLocal FAIL %d T-junctions (vertex interior to "
                     "an edge)\n",
                     tc.interiorVerts );
        ++fails;
    }
    if ( tc.inwardNormals != 0 )
    {
        std::printf( "    refineLocal FAIL %d faces wound inward\n",
                     tc.inwardNormals );
        ++fails;
    }
    if ( nvC != nvH )
    {
        std::printf( "    refineLocal FAIL vertex count %lld != hanging-node "
                     "%lld -- the closure must add no vertices\n",
                     nvC, nvH );
        ++fails;
    }

    std::printf( "  [%s] refineLocal %s (conforming V=%lld E=%lld F=%lld "
                 "Euler=%lld closure faces=%d | hanging Euler=%lld bad=%d "
                 "T-junctions=%d)\n",
                 tag, fails == 0 ? "ok" : "FAIL", tc.V, tc.E, tc.F, tc.euler(),
                 nClosure, th.euler(), th.badIncidence, th.interiorVerts );
    return fails;
}

// ===========================================================================

template <class Exec>
int run( const char* tag )
{
    int fails = 0;
    fails += case_patterns( tag );
    fails += case_inverse<Exec>( tag );
    fails += case_refine_local<Exec>( tag );
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
                "test_refine_closure: transient red-green-blue closure "
                "(closeFaces / unclose / translateMask)\n" );

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
