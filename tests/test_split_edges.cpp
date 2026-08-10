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

// Regression test: caller-driven edge split (tasks/edge-split.md).
//
// splitEdges() bisects EXACTLY the marked edges and subdivides every incident
// face into 2, 3 or 4 children on the bit pattern of its bisected edges. It is
// the first operation of the REMESH editing family; refine() is the whole of the
// HIERARCHICAL one, and a mesh belongs to one family only.
//
// EVERY COUNT ASSERTION BELOW IS A CLOSED-FORM IDENTITY, so the test checks
// arithmetic rather than checking Tessera against itself, and all of them are
// asserted through the global owned-count reductions so they hold at every rank
// count:
//
//   1. ONE EDGE. The globally smallest EdgeKey (a rank-count-invariant choice).
//      V+1, E+3, F+2 -- Euler delta 1 - 3 + 2 == 0, so V-E+F is still 2. The two
//      incident faces each become 2 children and nothing else moves.
//   2. THREE EDGES OF ONE FACE. V+3, E+9, F+6: that face becomes 4 children and
//      each of its three neighbours becomes 2. Euler delta 0.
//   3. ALL EDGES -- EQUIVALENCE WITH refine(). Every face then has |S| = 3, so
//      the result must be a uniform red refinement: V' = V+E = 642,
//      E' = 2E+3F = 1920, F' = 4F = 1280. Stronger than the counts, the two
//      meshes' vertex position multisets and face corner-position triple
//      multisets must be BITWISE identical (compared through raw IEEE bit
//      patterns, order-independently, so gid numbering is irrelevant). This is
//      the sharpest available check on the |S| = 3 path because it pins the new
//      code against machinery the gate already verifies.
//   4. TWO-EDGE PATTERN DETERMINISM. A rank-count-invariant mask (EdgeKey.id[0]
//      even) under which faces with all of |S| = 1, 2 and 3 occur -- asserted
//      non-vacuously. The V/E/F and the face corner-position triple multiset must
//      match a reference the same run computes alone on MPI_COMM_SELF, which is
//      what pins the two-edge diagonal tie-break as local and rank-count
//      invariant. (Nothing gid-keyed is comparable across rank counts: new gids
//      come from an MPI_Exscan.)
//   5. MIDPOINT AGREEMENT. checkMidpointAgreement() on every case, plus the
//      globally-decided ground truth shared with test_refine_splitedges: an edge
//      is in the map iff some rank touching it reports it, and no rank reports an
//      edge it does not touch.
//   6. MIDPOINT POSITIONS. Each midpoint is the exact (bitwise) average of its
//      endpoints, and is NOT on the unit sphere -- asserting the negative pins
//      the decision that AMR does not project.
//   7. USER-FIELD TRANSFER. A `double` and a `double[3]` vertex user field seeded
//      to linear functions of position are reproduced at every midpoint to 1e-15
//      relative.
//   8. REPEATED ROUNDS. Five successive splitEdges() with a length-threshold mask
//      and NO intervening migrate(): Euler == 2 and conformity after each, and
//      the global minimum triangle radius ratio (inradius/circumradius, 0.5 for
//      an equilateral triangle) stays above a floor MEASURED in the first
//      implementation run rather than guessed.
//   9. EMPTY MASK. A no-op: V/E/F and the topology checksum unchanged.
//  10. HALO VALID ON RETURN. Corrupting every ghost position and exchanging
//      restores the owners' values, and a SECOND splitEdges() with nothing in
//      between succeeds.
//  11. FAMILY GUARD. refine() on a splitEdges()-edited mesh throws naming both
//      families, and splitEdges() on a refine()d mesh throws.
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace Tessera;

// Coarse subdiv-2 icosphere entity counts (the Step-6b fixture).
static const long long V0 = 162, E0 = 480, F0 = 320;

// Floor on the global minimum triangle radius ratio (inradius/circumradius; 0.5
// exactly for an equilateral triangle) over case 8's five length-driven rounds.
//
// MEASURED, not guessed. The first implementation run reported, per round,
//   0.3780  0.3780  0.2815  0.3780  0.3780
// byte-identically at np1-5 on both backends and in both execution spaces, so
// the worst any round reaches is 0.2815 and the floor is set just below it. Note
// the sequence does not drift downward -- rounds 4 and 5 recover to the value
// round 1 had -- which is the substantive statement: a length-threshold mask
// driven through splitEdges() does not degrade triangle shape without bound.
// A regression that starts emitting slivers -- e.g. a two-edge diagonal chosen
// as the LONGER one -- drops this immediately.
static const double kMinRadiusRatioFloor = 0.25;

// ---------------------------------------------------------------------------
// Order-independent multiset checksum (same idiom as test_conforming_determinism)
// ---------------------------------------------------------------------------
//
// count + SUM + BXOR over a 64-bit hash per item. All three combiners are
// commutative, so the value depends on neither visit order, nor local index, nor
// how the items are spread over ranks -- which is what "the same mesh, however it
// is decomposed" needs.
struct Chk
{
    long long n = 0;
    unsigned long long sum = 0;
    unsigned long long x = 0;

    void add( unsigned long long h )
    {
        ++n;
        sum += h;
        x ^= h;
    }
    bool operator==( const Chk& o ) const
    {
        return n == o.n && sum == o.sum && x == o.x;
    }
    bool operator!=( const Chk& o ) const { return !( *this == o ); }

    void reduce( MPI_Comm comm )
    {
        long long gn = 0;
        unsigned long long gs = 0, gx = 0;
        MPI_Allreduce( &n, &gn, 1, MPI_LONG_LONG, MPI_SUM, comm );
        MPI_Allreduce( &sum, &gs, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm );
        MPI_Allreduce( &x, &gx, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, comm );
        n = gn;
        sum = gs;
        x = gx;
    }
};

static inline unsigned long long mix64( unsigned long long z )
{
    z += 0x9e3779b97f4a7c15ULL;
    z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
    z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
    return z ^ ( z >> 31 );
}
static inline unsigned long long hashCombine( unsigned long long a,
                                              unsigned long long b )
{
    return mix64( a ^ ( b + 0x9e3779b97f4a7c15ULL + ( a << 6 ) + ( a >> 2 ) ) );
}

//! BITWISE position hash: the raw IEEE bit patterns, not a quantisation. Both
//! sides of every comparison in this test compute their positions with the same
//! arithmetic on the same inputs (an existing vertex is copied verbatim, a
//! midpoint is one 0.5*(a+b)), so exact equality is the right assertion and a
//! tolerance would only hide a real difference.
static inline unsigned long long posHashExact( const std::array<double, 3>& p )
{
    unsigned long long h = 0xcbf29ce484222325ULL;
    for ( int d = 0; d < 3; ++d )
    {
        unsigned long long u = 0;
        std::memcpy( &u, &p[d], sizeof( u ) );
        h = hashCombine( h, u );
    }
    return h;
}

//! Canonical hash of a triangle from its three corner hashes: sorted, then
//! folded, so it is independent of which corner is stored at v[0].
static inline unsigned long long
triHash( unsigned long long a, unsigned long long b, unsigned long long c )
{
    unsigned long long t[3] = { a, b, c };
    std::sort( t, t + 3 );
    return hashCombine( hashCombine( t[0], t[1] ), t[2] );
}

static inline int gsum( MPI_Comm comm, int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
    return g;
}
static inline long long gsumll( MPI_Comm comm, long long local )
{
    long long g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return g;
}

// ---------------------------------------------------------------------------
// Mesh readers
// ---------------------------------------------------------------------------

//! Vertex gid -> position, over every locally held vertex (owned + ghost).
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

//! Corner vertex gids of every OWNED face.
template <class MeshT>
static std::vector<std::array<GlobalId, 3>> ownedFaceVerts( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    std::vector<std::array<GlobalId, 3>> out( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
        for ( int k = 0; k < 3; ++k )
            out[f][k] = fv( f, k );
    return out;
}

//! EdgeKey of every OWNED edge, in owned-edge local index order -- i.e. exactly
//! the indexing splitEdges()' mask uses.
template <class MeshT>
static std::vector<EdgeKey> ownedEdgeKeys( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto ev = Cabana::slice<EdgeField::Verts>( he );
    std::vector<EdgeKey> out( mesh.numOwnedEdges() );
    for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
        out[e] = makeEdgeKey( ev( e, 0 ), ev( e, 1 ) );
    return out;
}

//! The global mesh's V, E, F from the owned-count reductions.
template <class MeshT>
static void counts( MeshT& mesh, long long& V, long long& E, long long& F )
{
    V = TesseraTest::globalOwnedVertices( mesh );
    E = TesseraTest::globalOwnedEdges( mesh );
    F = TesseraTest::globalOwnedFaces( mesh );
}

//! Multiset signature of the OWNED vertex positions and of the OWNED face
//! corner-position triples, both reduced over `comm`. Bitwise (see
//! posHashExact) and gid-free, so two meshes agree iff they are the same
//! geometry however they are numbered or decomposed.
template <class MeshT>
static void geoSignature( MeshT& mesh, Chk& verts, Chk& faces, int& fails )
{
    const auto pos = readPositions( mesh );

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vp = Cabana::slice<VertexField::Position>( hv );
    for ( std::size_t i = 0; i < mesh.numOwnedVertices(); ++i )
    {
        std::array<double, 3> p = { 0, 0, 0 };
        for ( int d = 0; d < MeshT::dim && d < 3; ++d )
            p[d] = static_cast<double>( vp( i, d ) );
        verts.add( posHashExact( p ) );
    }

    for ( const auto& t : ownedFaceVerts( mesh ) )
    {
        unsigned long long h[3];
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( t[k] );
            if ( it == pos.end() )
            {
                ok = false; // every corner of an owned face must be held
                h[k] = 0;
            }
            else
                h[k] = posHashExact( it->second );
        }
        if ( !ok )
            ++fails;
        faces.add( triHash( h[0], h[1], h[2] ) );
    }

    verts.reduce( mesh.comm() );
    faces.reduce( mesh.comm() );
}

//! Global minimum inradius/circumradius over the owned faces. 0.5 exactly for an
//! equilateral triangle, 0 for a degenerate one. Reduced over mesh.comm().
template <class MeshT>
static double minRadiusRatio( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    double worst = 1.0;
    for ( const auto& t : ownedFaceVerts( mesh ) )
    {
        std::array<double, 3> p[3];
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( t[k] );
            if ( it == pos.end() )
                ok = false;
            else
                p[k] = it->second;
        }
        if ( !ok )
        {
            ++fails;
            continue;
        }
        double side[3] = { 0, 0, 0 };
        for ( int k = 0; k < 3; ++k )
        {
            double s = 0.0;
            for ( int d = 0; d < 3; ++d )
            {
                const double dd = p[( k + 1 ) % 3][d] - p[k][d];
                s += dd * dd;
            }
            side[k] = std::sqrt( s );
        }
        // Area from the cross product of two edge vectors.
        double u[3], v[3];
        for ( int d = 0; d < 3; ++d )
        {
            u[d] = p[1][d] - p[0][d];
            v[d] = p[2][d] - p[0][d];
        }
        const double cx = u[1] * v[2] - u[2] * v[1];
        const double cy = u[2] * v[0] - u[0] * v[2];
        const double cz = u[0] * v[1] - u[1] * v[0];
        const double area = 0.5 * std::sqrt( cx * cx + cy * cy + cz * cz );
        const double abc = side[0] * side[1] * side[2];
        if ( abc <= 0.0 )
        {
            ++fails;
            continue;
        }
        const double s = 0.5 * ( side[0] + side[1] + side[2] );
        // r/R = (area/s) / (abc/(4 area)) = 4 area^2 / (s abc)
        worst = std::min( worst, 4.0 * area * area / ( s * abc ) );
    }
    double global = worst;
    MPI_Allreduce( &worst, &global, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
    return global;
}

// ---------------------------------------------------------------------------
// Rank-count-invariant edge selections
// ---------------------------------------------------------------------------

//! The globally smallest EdgeKey over all owned edges. A pure function of the
//! global mesh, so every rank count picks the same edge.
template <class MeshT>
static EdgeKey globalMinEdgeKey( MeshT& mesh )
{
    EdgeKey best;
    best.id[0] = ~0ULL;
    best.id[1] = ~0ULL;
    for ( const EdgeKey& k : ownedEdgeKeys( mesh ) )
        if ( k < best )
            best = k;
    const int size = mesh.commSize();
    std::vector<unsigned long long> all( 2 * size, 0 );
    unsigned long long mine[2] = { best.id[0], best.id[1] };
    MPI_Allgather( mine, 2, MPI_UNSIGNED_LONG_LONG, all.data(), 2,
                   MPI_UNSIGNED_LONG_LONG, mesh.comm() );
    for ( int r = 0; r < size; ++r )
    {
        EdgeKey k;
        k.id[0] = all[2 * r];
        k.id[1] = all[2 * r + 1];
        if ( k < best )
            best = k;
    }
    return best;
}

//! The three EdgeKeys of the face with global gid 0, on every rank. Exactly one
//! rank owns that face and contributes; the others contribute zeros, so an
//! MPI_MAX carries the owner's values everywhere.
template <class MeshT>
static void faceZeroEdgeKeys( MeshT& mesh, EdgeKey out[3] )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    unsigned long long mine[6] = { 0, 0, 0, 0, 0, 0 };
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
        if ( fg( f ) == 0 )
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key =
                    makeEdgeKey( fv( f, k ), fv( f, ( k + 1 ) % 3 ) );
                mine[2 * k] = key.id[0];
                mine[2 * k + 1] = key.id[1];
            }
    unsigned long long all[6] = { 0, 0, 0, 0, 0, 0 };
    MPI_Allreduce( mine, all, 6, MPI_UNSIGNED_LONG_LONG, MPI_MAX, mesh.comm() );
    for ( int k = 0; k < 3; ++k )
    {
        out[k].id[0] = all[2 * k];
        out[k].id[1] = all[2 * k + 1];
    }
}

//! Mask marking this rank's owned edges whose key is in `want`.
template <class MeshT>
static std::vector<char> maskFromKeys( MeshT& mesh,
                                       const std::set<EdgeKey>& want )
{
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    std::vector<char> mask( keys.size(), 0 );
    for ( std::size_t e = 0; e < keys.size(); ++e )
        mask[e] = want.count( keys[e] ) ? 1 : 0;
    return mask;
}

//! Mask marking every owned edge whose lower endpoint gid is even. Defined on
//! the EdgeKey alone, so it selects the same global edge set at any rank count,
//! and it is irregular enough that faces with all of |S| = 1, 2 and 3 occur.
template <class MeshT>
static std::vector<char> parityMask( MeshT& mesh )
{
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    std::vector<char> mask( keys.size(), 0 );
    for ( std::size_t e = 0; e < keys.size(); ++e )
        mask[e] = ( keys[e].id[0] % 2 == 0 ) ? 1 : 0;
    return mask;
}

//! Mask marking every owned edge longer than the global mean owned edge length.
template <class MeshT>
static std::vector<char> aboveMeanLengthMask( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    std::vector<double> len( keys.size(), 0.0 );
    double localSum = 0.0;
    for ( std::size_t e = 0; e < keys.size(); ++e )
    {
        auto ia = pos.find( keys[e].id[0] );
        auto ib = pos.find( keys[e].id[1] );
        if ( ia == pos.end() || ib == pos.end() )
        {
            ++fails; // an owned edge's endpoints must be held locally
            continue;
        }
        double s = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            const double dd = ia->second[d] - ib->second[d];
            s += dd * dd;
        }
        len[e] = std::sqrt( s );
        localSum += len[e];
    }
    double totalLen = 0.0;
    MPI_Allreduce( &localSum, &totalLen, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );
    const long long nE = TesseraTest::globalOwnedEdges( mesh );
    const double mean = nE > 0 ? totalLen / static_cast<double>( nE ) : 0.0;

    std::vector<char> mask( keys.size(), 0 );
    for ( std::size_t e = 0; e < keys.size(); ++e )
        mask[e] = ( len[e] > mean ) ? 1 : 0;
    return mask;
}

// ---------------------------------------------------------------------------
// Shared post-conditions
// ---------------------------------------------------------------------------

//! Everything that must hold of ANY splitEdges() result: conformity, Euler,
//! ownership a partition, the owned 1-ring local, and the split-edge map both
//! agreed cross-rank and exactly covering the edges this rank touches.
//! `preVerts` is the pre-split owned face corner list. Returns LOCAL fails.
template <class MeshT>
static int checkAll( MeshT& mesh, const SplitResult& res,
                     const std::vector<std::array<GlobalId, 3>>& preVerts,
                     bool deepGeometry )
{
    int fails = 0;
    long long V, E, F;
    counts( mesh, V, E, F );
    fails += TesseraTest::checkOwnershipPartition( mesh, V, E, F );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += TesseraTest::checkConforming( mesh );
    fails += TesseraTest::checkMidpointAgreement( mesh.comm(), mesh.commSize(),
                                                  res.midpoints );
    fails += TesseraTest::checkSplitEdgeCoverage( mesh.comm(), mesh.commSize(),
                                                  preVerts, res.midpoints );
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;
    if ( deepGeometry )
        fails += TesseraTest::checkNoInteriorVertex( mesh );
    return fails;
}

//! Build -> partition -> distribute the subdiv-2 fixture on `comm`.
template <class MeshT, class Exec>
static void setup( MeshT& mesh, MeshHalo<typename Exec::memory_space>& halo )
{
    buildIcosphere( mesh, 2 );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );
}

// ===========================================================================
// Cases
// ===========================================================================

//! Cases 1 and 2: the two closed-form count identities, plus the |S| histogram
//! each of them implies. `nEdges` is 1 (one edge) or 3 (one face's edges).
template <class MeshT, class Exec>
static int caseCounts( int rank, const char* tag, int nEdges )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    long long V, E, F;
    counts( mesh, V, E, F );
    const auto preVerts = ownedFaceVerts( mesh );

    std::set<EdgeKey> want;
    if ( nEdges == 1 )
    {
        want.insert( globalMinEdgeKey( mesh ) );
    }
    else
    {
        EdgeKey k[3];
        faceZeroEdgeKeys( mesh, k );
        for ( int i = 0; i < 3; ++i )
            want.insert( k[i] );
    }

    const SplitResult res =
        splitEdges( mesh, halo, maskFromKeys( mesh, want ) );

    int local = checkAll( mesh, res, preVerts, true );
    long long V2, E2, F2;
    counts( mesh, V2, E2, F2 );

    // Closed forms. One edge: V+1, E+3, F+2 (both incident faces -> 2 children,
    // the bisected edge -> 2 halves, one new median per child pair). Three edges
    // of one face: V+3, E+9, F+6 (that face -> 4, its three neighbours -> 2).
    const long long dV = nEdges == 1 ? 1 : 3;
    const long long dE = nEdges == 1 ? 3 : 9;
    const long long dF = nEdges == 1 ? 2 : 6;
    if ( V2 != V + dV || E2 != E + dE || F2 != F + dF )
        ++local;
    if ( res.requested != nEdges || res.split != nEdges )
        ++local;
    if ( res.facesBefore != F || res.facesAfter != F2 )
        ++local;

    // |S| histogram, also closed form: one edge -> two |S|=1 faces; three edges
    // of one face -> that face at |S|=3 and its three neighbours at |S|=1.
    const long long wantS1 = nEdges == 1 ? 2 : 3;
    const long long wantS3 = nEdges == 1 ? 0 : 1;
    if ( res.pattern[1] != wantS1 || res.pattern[2] != 0 ||
         res.pattern[3] != wantS3 )
        ++local;
    if ( res.pattern[0] != F - wantS1 - wantS3 )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case%d (%s edge%s) %s: V %lld->%lld E %lld->%lld "
                     "F %lld->%lld |S|=(%lld,%lld,%lld,%lld)\n",
                     tag, nEdges == 1 ? 1 : 2, nEdges == 1 ? "one" : "three",
                     nEdges == 1 ? "" : "s", glob == 0 ? "ok" : "FAIL", V, V2,
                     E, E2, F, F2, res.pattern[0], res.pattern[1],
                     res.pattern[2], res.pattern[3] );
    return glob == 0 ? 0 : 1;
}

//! Case 3: marking EVERY edge must reproduce a uniform refine() exactly --
//! counts, vertex position multiset, and face corner-position triple multiset,
//! all bitwise.
template <class MeshT, class Exec>
static int caseAllEdges( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    Chk splitV, splitF;
    long long sV, sE, sF;
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        const auto preVerts = ownedFaceVerts( mesh );
        std::vector<char> mask( mesh.numOwnedEdges(), 1 );
        const SplitResult res = splitEdges( mesh, halo, mask );
        local += checkAll( mesh, res, preVerts, true );
        counts( mesh, sV, sE, sF );
        if ( res.pattern[0] != 0 || res.pattern[1] != 0 ||
             res.pattern[2] != 0 || res.pattern[3] != F0 )
            ++local; // every face must be a |S| = 3 red split
        if ( res.split != E0 )
            ++local;
        geoSignature( mesh, splitV, splitF, local );
    }

    Chk refV, refF;
    long long rV, rE, rF;
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        std::vector<char> mask( mesh.numOwnedFaces(), 1 );
        refine( mesh, halo, mask );
        counts( mesh, rV, rE, rF );
        geoSignature( mesh, refV, refF, local );
    }

    // V' = V+E, E' = 2E+3F, F' = 4F on the subdiv-2 fixture.
    if ( sV != V0 + E0 || sE != 2 * E0 + 3 * F0 || sF != 4 * F0 )
        ++local;
    if ( sV != rV || sE != rE || sF != rF )
        ++local;
    if ( splitV != refV )
        ++local; // vertex position multisets differ
    if ( splitF != refF )
        ++local; // face corner-position triple multisets differ

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case3 (all edges == uniform refine) %s: "
                     "V=%lld E=%lld F=%lld (refine V=%lld E=%lld F=%lld)\n",
                     tag, glob == 0 ? "ok" : "FAIL", sV, sE, sF, rV, rE, rF );
    return glob == 0 ? 0 : 1;
}

//! Case 4: the parity mask, against a reference the same run computes alone on
//! MPI_COMM_SELF. At np1 the comparison is a tautology; at np2..5 it is the real
//! assertion that the edit -- including the two-edge diagonal -- is rank-count
//! invariant.
template <class MeshT, class Exec>
static int caseDeterminism( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    auto measure = [&]( MPI_Comm comm, Chk& cv, Chk& cf, long long& V,
                        long long& E, long long& F, SplitResult& res )
    {
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
        const auto preVerts = ownedFaceVerts( mesh );
        res = splitEdges( mesh, halo, parityMask( mesh ) );
        local += checkAll( mesh, res, preVerts, false );
        counts( mesh, V, E, F );
        geoSignature( mesh, cv, cf, local );
    };

    Chk wv, wf, sv, sf;
    long long wV, wE, wF, xV, xE, xF;
    SplitResult wres, sres;
    measure( MPI_COMM_WORLD, wv, wf, wV, wE, wF, wres );
    measure( MPI_COMM_SELF, sv, sf, xV, xE, xF, sres );

    if ( wV != xV || wE != xE || wF != xF )
        ++local;
    if ( wv != sv || wf != sf )
        ++local;
    // Non-vacuity: all three bit patterns must actually occur, or the case
    // proves nothing about the two-edge diagonal.
    if ( wres.pattern[1] <= 0 || wres.pattern[2] <= 0 || wres.pattern[3] <= 0 )
        ++local;
    // The |S| histogram is itself a function of the global mesh alone.
    for ( int i = 0; i < 4; ++i )
        if ( wres.pattern[i] != sres.pattern[i] )
            ++local;
    if ( wres.diagTies != sres.diagTies )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case4 (two-edge determinism, np%d vs SELF) %s: "
                     "V=%lld E=%lld F=%lld |S|=(%lld,%lld,%lld,%lld) "
                     "diagTies=%lld\n",
                     tag, size, glob == 0 ? "ok" : "FAIL", wV, wE, wF,
                     wres.pattern[0], wres.pattern[1], wres.pattern[2],
                     wres.pattern[3], wres.diagTies );
    return glob == 0 ? 0 : 1;
}

//! Case 6: every midpoint is the exact average of its endpoints and is NOT on
//! the unit sphere.
template <class MeshT, class Exec>
static int caseMidpointPositions( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );
    const auto preVerts = ownedFaceVerts( mesh );
    const SplitResult res = splitEdges( mesh, halo, parityMask( mesh ) );

    int local = checkAll( mesh, res, preVerts, false );
    const auto pos = readPositions( mesh );
    long long checked = 0, offSphere = 0;
    for ( const auto& kv : res.midpoints )
    {
        auto im = pos.find( kv.second );
        auto ia = pos.find( kv.first.id[0] );
        auto ib = pos.find( kv.first.id[1] );
        if ( im == pos.end() || ia == pos.end() || ib == pos.end() )
        {
            // Every edge in the map is an edge of one of this rank's faces, so
            // after rebuildHalo() all three vertices are held.
            ++local;
            continue;
        }
        ++checked;
        double norm = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            const double want = 0.5 * ( ia->second[d] + ib->second[d] );
            if ( im->second[d] != want )
                ++local; // not the exact (bitwise) average
            norm += im->second[d] * im->second[d];
        }
        // The midpoint of a chord of the unit sphere is strictly inside it: AMR
        // does NOT project, so asserting the negative pins that decision.
        if ( std::sqrt( norm ) < 1.0 - 1e-9 )
            ++offSphere;
    }
    if ( checked != static_cast<long long>( res.midpoints.size() ) )
        ++local;
    if ( offSphere != checked )
        ++local; // some midpoint landed on the unit sphere

    const long long gChecked = gsumll( MPI_COMM_WORLD, checked );
    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case6 (midpoint positions) %s: %lld midpoints "
                     "checked, all exact averages and all off-sphere\n",
                     tag, glob == 0 ? "ok" : "FAIL", gChecked );
    return glob == 0 ? 0 : 1;
}

//! Case 7: a `double` and a `double[3]` vertex user field, seeded to linear
//! functions of position, are reproduced at every midpoint. The default policy
//! averages, and the average of a linear function is that function of the
//! average, so the identity is exact up to rounding.
template <class Exec>
static int caseUserFields( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    using UMeshT = Mesh<double, 3, VertexFields<double, double[3]>,
                        EdgeFields<>, FaceFields<>, mem, Exec>;
    constexpr std::size_t F0Idx = userVertexField<0>();
    constexpr std::size_t F1Idx = userVertexField<1>();

    auto f0 = []( const std::array<double, 3>& p )
    { return 0.25 + 2.0 * p[0] - 3.0 * p[1] + 5.0 * p[2]; };
    auto f1 = []( const std::array<double, 3>& p, int c )
    { return 1.5 * p[c] - 0.25 * ( c + 1 ) * p[( c + 1 ) % 3] + 0.125 * c; };

    UMeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<UMeshT, Exec>( mesh, halo );

    // Seed the OWNED vertices, then exchange so a midpoint owner blending across
    // a partition boundary reads the owner's value from its ghost.
    {
        Cabana::AoSoA<typename UMeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", mesh.numVertices() );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto vp = Cabana::slice<VertexField::Position>( hv );
        auto u0 = Cabana::slice<F0Idx>( hv );
        auto u1 = Cabana::slice<F1Idx>( hv );
        for ( std::size_t i = 0; i < mesh.numOwnedVertices(); ++i )
        {
            std::array<double, 3> p = { vp( i, 0 ), vp( i, 1 ), vp( i, 2 ) };
            u0( i ) = f0( p );
            for ( int c = 0; c < 3; ++c )
                u1( i, c ) = f1( p, c );
        }
        Cabana::deep_copy( mesh.vertices(), hv );
        haloExchange( mesh, halo );
    }

    const auto preVerts = ownedFaceVerts( mesh );
    const SplitResult res = splitEdges( mesh, halo, parityMask( mesh ) );
    int local = checkAll( mesh, res, preVerts, false );

    // Every OWNED vertex -- original or midpoint -- must satisfy the identity.
    long long checked = 0;
    double worst = 0.0;
    {
        Cabana::AoSoA<typename UMeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", mesh.numVertices() );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto vp = Cabana::slice<VertexField::Position>( hv );
        auto u0 = Cabana::slice<F0Idx>( hv );
        auto u1 = Cabana::slice<F1Idx>( hv );
        auto rel = [&]( double got, double want )
        {
            const double scale = std::max( 1.0, std::fabs( want ) );
            return std::fabs( got - want ) / scale;
        };
        for ( std::size_t i = 0; i < mesh.numOwnedVertices(); ++i )
        {
            std::array<double, 3> p = { vp( i, 0 ), vp( i, 1 ), vp( i, 2 ) };
            ++checked;
            worst = std::max( worst, rel( u0( i ), f0( p ) ) );
            for ( int c = 0; c < 3; ++c )
                worst = std::max( worst, rel( u1( i, c ), f1( p, c ) ) );
        }
    }
    double gWorst = worst;
    MPI_Allreduce( &worst, &gWorst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    if ( gWorst > 1e-15 )
        ++local;

    const long long gChecked = gsumll( MPI_COMM_WORLD, checked );
    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case7 (user-field transfer) %s: %lld owned "
                     "vertices, worst relative error %.3e\n",
                     tag, glob == 0 ? "ok" : "FAIL", gChecked, gWorst );
    return glob == 0 ? 0 : 1;
}

//! Case 8: five successive length-driven rounds with NOTHING in between.
template <class MeshT, class Exec>
static int caseRepeatedRounds( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    for ( int round = 0; round < 5; ++round )
    {
        const auto preVerts = ownedFaceVerts( mesh );
        const std::vector<char> mask = aboveMeanLengthMask( mesh, local );
        const SplitResult res = splitEdges( mesh, halo, mask );
        local += checkAll( mesh, res, preVerts, false );
        if ( res.requested <= 0 )
            ++local; // vacuous round
        long long V, E, F;
        counts( mesh, V, E, F );
        const double q = minRadiusRatio( mesh, local );
        if ( q < kMinRadiusRatioFloor )
            ++local;
        if ( rank == 0 )
            std::printf( "  [%s] case8 round%d: split=%lld V=%lld E=%lld "
                         "F=%lld |S|=(%lld,%lld,%lld,%lld) minRadiusRatio="
                         "%.4f\n",
                         tag, round + 1, res.split, V, E, F, res.pattern[0],
                         res.pattern[1], res.pattern[2], res.pattern[3], q );
    }
    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case8 (five rounds) %s\n", tag,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Case 9: an empty mask is the identity, and case 10: the halo is valid on
//! return -- a ghost corrupt/resync round-trips and a SECOND splitEdges()
//! succeeds with nothing in between.
template <class MeshT, class Exec>
static int caseEmptyAndHalo( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    // ---- case 9: empty mask ------------------------------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        long long V, E, F;
        counts( mesh, V, E, F );
        unsigned long long cv0, ce0, cf0, cv1, ce1, cf1;
        TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );

        std::vector<char> mask( mesh.numOwnedEdges(), 0 );
        const SplitResult res = splitEdges( mesh, halo, mask );

        long long V2, E2, F2;
        counts( mesh, V2, E2, F2 );
        TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );
        if ( V != V2 || E != E2 || F != F2 )
            ++local;
        if ( cv0 != cv1 || ce0 != ce1 || cf0 != cf1 )
            ++local;
        if ( res.requested != 0 || res.split != 0 || !res.midpoints.empty() ||
             res.facesBefore != res.facesAfter )
            ++local;
    }

    // ---- case 10: halo valid on return -------------------------------------
    long long corrupted = 0, planSize = 0;
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        auto preVerts = ownedFaceVerts( mesh );
        SplitResult res = splitEdges( mesh, halo, parityMask( mesh ) );
        local += checkAll( mesh, res, preVerts, false );

        for ( const auto* p : { &halo.vplan, &halo.eplan, &halo.fplan } )
            planSize += static_cast<long long>( p->totalSend() ) +
                        static_cast<long long>( p->totalRecv() );
        planSize = gsumll( MPI_COMM_WORLD, planSize );

        // Corrupt every ghost position; haloExchange() must restore the owners'.
        {
            Cabana::AoSoA<typename MeshT::vertex_member_types,
                          Kokkos::HostSpace>
                hv( "hv", mesh.numVertices() );
            Cabana::deep_copy( hv, mesh.vertices() );
            auto pos = Cabana::slice<VertexField::Position>( hv );
            for ( std::size_t i = mesh.numOwnedVertices();
                  i < mesh.numVertices(); ++i )
            {
                ++corrupted;
                for ( int d = 0; d < MeshT::dim; ++d )
                    pos( i, d ) = -1234.5;
            }
            Cabana::deep_copy( mesh.vertices(), hv );
        }
        corrupted = gsumll( MPI_COMM_WORLD, corrupted );
        haloExchange( mesh, halo );
        {
            Cabana::AoSoA<typename MeshT::vertex_member_types,
                          Kokkos::HostSpace>
                hv( "hv", mesh.numVertices() );
            Cabana::deep_copy( hv, mesh.vertices() );
            auto pos = Cabana::slice<VertexField::Position>( hv );
            for ( std::size_t i = mesh.numOwnedVertices();
                  i < mesh.numVertices(); ++i )
                if ( pos( i, 0 ) == -1234.5 )
                    ++local; // haloExchange did not restore the owner's value
        }

        // A SECOND splitEdges() with nothing in between must work: its Phase B
        // needs the positions of both endpoints of every midpoint it owns, and
        // across a partition boundary one of them is a ghost.
        preVerts = ownedFaceVerts( mesh );
        res = splitEdges( mesh, halo, parityMask( mesh ) );
        local += checkAll( mesh, res, preVerts, false );
        if ( res.requested <= 0 )
            ++local;

        if ( size > 1 && ( planSize <= 0 || corrupted <= 0 ) )
            ++local; // vacuous: an empty plan passes every structural check
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case9+10 (empty mask, halo valid on return) %s: "
                     "plan=%lld ghostsCorrupted=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", planSize, corrupted );
    return glob == 0 ? 0 : 1;
}

//! Case 11: the two editing families are disjoint and the guard says so.
template <class MeshT, class Exec>
static int caseFamilyGuard( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    auto namesBothFamilies = []( const std::string& what )
    {
        return what.find( "Hierarchical" ) != std::string::npos &&
               what.find( "Remesh" ) != std::string::npos;
    };

    // refine() on a mesh splitEdges() has edited.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        splitEdges( mesh, halo,
                    maskFromKeys( mesh, { globalMinEdgeKey( mesh ) } ) );
        bool threw = false, named = false;
        try
        {
            std::vector<char> mask( mesh.numOwnedFaces(), 1 );
            refine( mesh, halo, mask );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            named = namesBothFamilies( e.what() );
        }
        if ( !threw || !named )
            ++local;
    }

    // splitEdges() on a refine()d mesh.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        std::vector<char> fmask( mesh.numOwnedFaces(), 1 );
        refine( mesh, halo, fmask );
        bool threw = false, named = false;
        try
        {
            std::vector<char> emask( mesh.numOwnedEdges(), 1 );
            splitEdges( mesh, halo, emask );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            named = namesBothFamilies( e.what() );
        }
        if ( !threw || !named )
            ++local;
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case11 (editing-family guard) %s\n", tag,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    int fails = 0;
    fails += caseCounts<MeshT, Exec>( rank, tag, 1 );
    fails += caseCounts<MeshT, Exec>( rank, tag, 3 );
    fails += caseAllEdges<MeshT, Exec>( rank, tag );
    fails += caseDeterminism<MeshT, Exec>( rank, size, tag );
    fails += caseMidpointPositions<MeshT, Exec>( rank, tag );
    fails += caseUserFields<Exec>( rank, tag );
    fails += caseRepeatedRounds<MeshT, Exec>( rank, tag );
    fails += caseEmptyAndHalo<MeshT, Exec>( rank, size, tag );
    fails += caseFamilyGuard<MeshT, Exec>( rank, tag );
    std::fflush( stdout );
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
                "test_split_edges: caller-driven edge split (size %d)\n",
                size );
        std::fflush( stdout );

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
