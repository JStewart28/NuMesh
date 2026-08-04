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

// Regression test: the conforming closure is DETERMINISTIC (Task 7 of
// tasks/conforming-refinement.md). Three independent senses:
//
//   A. RANK-COUNT INDEPENDENCE. The same geometrically-defined refinement of
//      the same starting mesh must produce the same conforming mesh however
//      many ranks compute it. The reference is computed IN THE SAME RUN on
//      MPI_COMM_SELF -- every rank redundantly refines the whole mesh by
//      itself -- and compared against the MPI_COMM_WORLD result. At np1 the
//      comparison is a tautology; at np2..5 it is the real thing.
//
//      WHY THE COMPARISON IS BY POSITION, NOT BY GID. New vertex and face gids
//      come from an MPI_Exscan over ranks (Phase 2c / step 3c), so WHICH gid a
//      given midpoint receives is a function of the partition: at np1 the
//      midpoints are numbered in global EdgeKey order, at np>1 they are grouped
//      by owner rank first. Nothing gid-keyed is therefore comparable across
//      rank counts, while everything positional is -- a midpoint's position is
//      a function of its edge alone. So each face is canonicalised as the
//      sorted triple of its corners' quantised positions.
//
//      This case is what pins the BLUE-DIAGONAL TIE-BREAK, and it is worth
//      being precise about what it can prove. The tie-break connects the
//      lower-GID midpoint of the quad to its opposite corner. Its inputs are
//      globally agreed within a run, so the closure never depends on which rank
//      owns the face -- but they are gid-valued, and gids are rank-count
//      dependent as above. The red layer, the |S| histogram, the closure-vertex
//      set, and V/E/F are all provably rank-count independent; the VISIBLE
//      layer is too only if the diagonal choice is. All of them are asserted,
//      and the breakdown is printed component by component (plus a direct count
//      of blue parents whose diagonal differs from the serial reference) so a
//      failure names its own cause instead of just "the checksums differ".
//
//   B. CLOSURE IDEMPOTENCE. Re-refining an already-closed mesh with an EMPTY
//      mask is un-close -> no red split -> re-close, which must be the identity
//      on the visible topology. This is the sharpest single probe of the
//      closure's most delicate premise: that the split-edge information the
//      re-closure needs is recoverable from the mesh as it stands. Face gids of
//      closure children are NOT expected to survive (the re-closure allocates a
//      fresh block above the global max), so the comparison is over (sorted
//      corner gids, level, parent gid, parent corners) -- everything but the
//      child's own gid -- plus the red layer by gid and the V/E/F counts.
//
//   C. CROSS-MODE EQUIVALENCE UNDER A UNIFORM MASK. With every face marked
//      there are no kept faces, so |S| = 0 everywhere and no closure child is
//      emitted: the two modes run the same red engine over the same exscans and
//      must produce bit-identical topology -- the same face gids, the same
//      corner gids, the same levels, not merely the same counts. A mismatch
//      means the closure fired when it should have been inert.
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
#include <map>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace Tessera;

// ---------------------------------------------------------------------------
// Order-independent multiset checksum
// ---------------------------------------------------------------------------
//
// count + SUM + BXOR over a 64-bit hash per item. All three combiners are
// commutative, so the value does not depend on the order items are visited in,
// on the local index of anything, or on how the items are spread over ranks --
// which is exactly what "the same mesh, however it is decomposed" needs.
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

    //! Reduce this rank's contribution into the global value on every rank.
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

//! splitmix64 finalizer -- a cheap avalanche so that summing and XOR-ing the
//! hashes of distinct items does not collide structurally.
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

//! Quantised position hash. 1e-9 is far below the smallest vertex separation
//! these meshes reach (a level-8 icosphere edge is ~2e-3) and far above any
//! rounding difference between two runs that compute the same midpoint with the
//! same arithmetic, so it is a faithful identity for a vertex POSITION.
static inline unsigned long long posHash( const std::array<double, 3>& p )
{
    unsigned long long h = 0xcbf29ce484222325ULL;
    for ( int d = 0; d < 3; ++d )
        h = hashCombine(
            h, static_cast<unsigned long long>( std::llround( p[d] * 1e9 ) ) );
    return h;
}

//! Canonical (order-independent) hash of a triangle given its three corner
//! hashes: sorted, then folded. Sorting is what makes it independent of which
//! corner happens to be stored at v[0].
static inline unsigned long long
triHash( unsigned long long a, unsigned long long b, unsigned long long c )
{
    unsigned long long t[3] = { a, b, c };
    std::sort( t, t + 3 );
    return hashCombine( hashCombine( t[0], t[1] ), t[2] );
}

static inline long long globalSum( long long v, MPI_Comm comm )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return g;
}

// ---------------------------------------------------------------------------
// Mesh readers
// ---------------------------------------------------------------------------

//! Vertex gid -> position, over every locally held vertex.
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

//! Visible OWNED faces with their closure bookkeeping.
template <class MeshT>
static std::vector<VisibleFace> ownedVisible( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    return readVisibleFaces<MeshT>( hf, mesh.numOwnedFaces() );
}

// ---------------------------------------------------------------------------
// Case A -- rank-count independence against an MPI_COMM_SELF reference
// ---------------------------------------------------------------------------

//! Everything about a conforming mesh that is provably a function of the
//! GLOBAL mesh alone (no gid, no local index, no owner rank).
struct GeoSig
{
    long long V = 0, E = 0, F = 0;
    long long hist[4] = { 0, 0, 0, 0 }; //!< |S| histogram, summed over rounds
    Chk red;                            //!< red layer, by position triple
    Chk vis;                            //!< visible layer, by position triple
    Chk closureVerts;                   //!< former hanging nodes, by position
    //! Blue parents: canonical parent hash -> canonical diagonal hash. Only the
    //! serial reference fills this globally; the distributed run fills its own
    //! local share and looks each entry up in the reference.
    std::map<unsigned long long, unsigned long long> blueDiag;
};

//! Geometric refine mask: mark a face iff its centroid's z lies above `zmin`.
//! Defined purely by position, so it selects the SAME faces of the same global
//! mesh at any rank count -- unlike a gid predicate, whose target set moves as
//! soon as the exscan renumbers the faces.
template <class MeshT>
static std::vector<char>
capMask( MeshT& mesh,
         const std::unordered_map<GlobalId, std::array<double, 3>>& pos,
         double zmin, int& fails )
{
    const std::vector<VisibleFace> vis = ownedVisible( mesh );
    std::vector<char> mask( vis.size(), 0 );
    for ( std::size_t f = 0; f < vis.size(); ++f )
    {
        double cz = 0.0;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( vis[f].v[k] );
            if ( it == pos.end() )
            {
                ++fails; // a corner position must be locally held here
                cz = -1e30;
                break;
            }
            cz += it->second[2] / 3.0;
        }
        mask[f] = ( cz > zmin ) ? 1 : 0;
    }
    return mask;
}

//! Build -> distribute -> two geometric conforming rounds on `comm`, and
//! measure. Every returned Chk is reduced over `comm`; blueDiag is this rank's
//! own share, which for the MPI_COMM_SELF reference is the whole mesh.
template <class MeshT, class Exec>
static GeoSig refineAndMeasure( MPI_Comm comm, int& fails )
{
    using mem = typename Exec::memory_space;
    GeoSig sig;

    MeshT mesh( comm );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    int rank = 0;
    MPI_Comm_rank( comm, &rank );

    const double zmin[2] = { 0.35, 0.65 };
    for ( int round = 0; round < 2; ++round )
    {
        auto pos = readPositions( mesh );
        auto res =
            refine( mesh, halo, capMask( mesh, pos, zmin[round], fails ) );
        for ( int i = 0; i < 4; ++i )
            sig.hist[i] += res.closure.patternCount[i];

        // refine() leaves an owned-only mesh; the next round's geometric mask
        // and the measurement below both need corner positions locally.
        std::vector<Rank> dest( mesh.numOwnedFaces(),
                                static_cast<Rank>( rank ) );
        migrate( mesh, halo, dest );
        haloExchange( mesh, halo );
    }

    long long h[4] = { sig.hist[0], sig.hist[1], sig.hist[2], sig.hist[3] };
    long long gh[4] = { 0, 0, 0, 0 };
    MPI_Allreduce( h, gh, 4, MPI_LONG_LONG, MPI_SUM, comm );
    for ( int i = 0; i < 4; ++i )
        sig.hist[i] = gh[i];

    // ---- positional signatures -------------------------------------------
    const auto pos = readPositions( mesh );
    const std::vector<VisibleFace> vis = ownedVisible( mesh );
    auto ph = [&]( GlobalId g ) -> unsigned long long
    {
        auto it = pos.find( g );
        if ( it == pos.end() )
        {
            ++fails; // every corner of an owned face must be held after halo
            return 0;
        }
        return posHash( it->second );
    };

    for ( const auto& f : vis )
        sig.vis.add( triHash( ph( f.v[0] ), ph( f.v[1] ), ph( f.v[2] ) ) );

    const UncloseResult un = unclose( vis );
    for ( const auto& r : un.red )
        sig.red.add( triHash( ph( r.v[0] ), ph( r.v[1] ), ph( r.v[2] ) ) );

    // Closure vertices: a child corner that is not a parent corner.
    std::map<unsigned long long, int> cverts;
    for ( const auto& f : vis )
    {
        if ( f.parent == invalid_gid )
            continue;
        for ( int k = 0; k < 3; ++k )
            if ( f.v[k] != f.parentVerts[0] && f.v[k] != f.parentVerts[1] &&
                 f.v[k] != f.parentVerts[2] )
                cverts[ph( f.v[k] )] = 1;
    }
    for ( const auto& kv : cverts )
        sig.closureVerts.add( kv.first );

    // ---- blue parents and their chosen diagonal ---------------------------
    //
    // A blue parent has exactly three children. Two edges are internal to the
    // group; the DIAGONAL is the one with exactly one endpoint among the
    // parent's corners (the other internal edge joins the two midpoints). That
    // characterisation holds for both branches of the tie-break, so it reads
    // the CHOICE without assuming which branch was taken.
    {
        std::map<GlobalId, std::vector<const VisibleFace*>> byParent;
        for ( const auto& f : vis )
            if ( f.parent != invalid_gid )
                byParent[f.parent].push_back( &f );

        for ( const auto& kv : byParent )
        {
            if ( kv.second.size() != 3 )
                continue; // 2 = green, 4 = red-closure
            std::map<std::pair<GlobalId, GlobalId>, int> edgeCount;
            for ( const VisibleFace* f : kv.second )
                for ( int k = 0; k < 3; ++k )
                {
                    GlobalId a = f->v[k], b = f->v[( k + 1 ) % 3];
                    if ( b < a )
                        std::swap( a, b );
                    ++edgeCount[{ a, b }];
                }
            const GlobalId* pv = kv.second[0]->parentVerts;
            auto isCorner = [&]( GlobalId g )
            { return g == pv[0] || g == pv[1] || g == pv[2]; };

            for ( const auto& ec : edgeCount )
            {
                if ( ec.second != 2 )
                    continue;
                const int nCorner = ( isCorner( ec.first.first ) ? 1 : 0 ) +
                                    ( isCorner( ec.first.second ) ? 1 : 0 );
                if ( nCorner != 1 )
                    continue;
                const unsigned long long parentHash =
                    triHash( ph( pv[0] ), ph( pv[1] ), ph( pv[2] ) );
                unsigned long long d0 = ph( ec.first.first );
                unsigned long long d1 = ph( ec.first.second );
                if ( d1 < d0 )
                    std::swap( d0, d1 );
                sig.blueDiag[parentHash] = hashCombine( d0, d1 );
            }
        }
    }

    sig.V = TesseraTest::globalOwnedVertices( mesh );
    sig.E = TesseraTest::globalOwnedEdges( mesh );
    sig.F = TesseraTest::globalOwnedFaces( mesh );
    sig.red.reduce( comm );
    sig.vis.reduce( comm );
    sig.closureVerts.reduce( comm );
    return sig;
}

template <class Exec, class ConfMesh>
static int case_rank_count( int rank, int size, const char* tag )
{
    int fails = 0;

    // The distributed result, and the SAME refinement recomputed by this rank
    // alone on MPI_COMM_SELF. Both are the same global mesh under the same
    // geometric mask, so every quantity below is comparable.
    GeoSig dist = refineAndMeasure<ConfMesh, Exec>( MPI_COMM_WORLD, fails );
    GeoSig ref = refineAndMeasure<ConfMesh, Exec>( MPI_COMM_SELF, fails );

    const bool okCounts =
        ( dist.V == ref.V && dist.E == ref.E && dist.F == ref.F );
    bool okHist = true;
    for ( int i = 0; i < 4; ++i )
        okHist = okHist && ( dist.hist[i] == ref.hist[i] );
    const bool okRed = ( dist.red == ref.red );
    const bool okVis = ( dist.vis == ref.vis );
    const bool okCv = ( dist.closureVerts == ref.closureVerts );

    // Direct diagnosis of the tie-break: how many blue parents that BOTH runs
    // produced chose a different diagonal, and how many the reference does not
    // contain at all (which would mean the red layer itself diverged).
    long long diagMismatch = 0, parentMissing = 0;
    for ( const auto& kv : dist.blueDiag )
    {
        auto it = ref.blueDiag.find( kv.first );
        if ( it == ref.blueDiag.end() )
            ++parentMissing;
        else if ( it->second != kv.second )
            ++diagMismatch;
    }
    diagMismatch = globalSum( diagMismatch, MPI_COMM_WORLD );
    parentMissing = globalSum( parentMissing, MPI_COMM_WORLD );

    if ( !okCounts || !okHist || !okRed || !okVis || !okCv )
        ++fails;
    if ( diagMismatch != 0 || parentMissing != 0 )
        ++fails;
    // Non-vacuity: an all-|S|=0 refinement would make every comparison above
    // trivially true.
    if ( dist.hist[1] + dist.hist[2] + dist.hist[3] <= 0 )
        ++fails;
    if ( dist.closureVerts.n <= 0 )
        ++fails;

    if ( rank == 0 )
        std::printf(
            "  [%s] rank-count-independence %s (np=%d V=%lld E=%lld F=%lld "
            "|S| hist=[%lld,%lld,%lld,%lld] counts=%s hist=%s red=%s vis=%s "
            "closureVerts=%s(%lld) blueDiagMismatch=%lld parentMissing=%lld)\n",
            tag, fails == 0 ? "ok" : "FAIL", size, dist.V, dist.E, dist.F,
            dist.hist[0], dist.hist[1], dist.hist[2], dist.hist[3],
            okCounts ? "ok" : "DIFF", okHist ? "ok" : "DIFF",
            okRed ? "ok" : "DIFF", okVis ? "ok" : "DIFF", okCv ? "ok" : "DIFF",
            dist.closureVerts.n, diagMismatch, parentMissing );
    return fails;
}

// ---------------------------------------------------------------------------
// Case B -- closure idempotence
// ---------------------------------------------------------------------------

//! Multiset checksum of the visible layer EXCLUDING each face's own gid: the
//! sorted corner gids, the level, the parent gid, and the parent's corners.
//! Everything the closure decides, and nothing the gid exscan does.
static Chk visibleSigNoGid( const std::vector<VisibleFace>& vis )
{
    Chk c;
    for ( const auto& f : vis )
    {
        unsigned long long h =
            triHash( mix64( f.v[0] ), mix64( f.v[1] ), mix64( f.v[2] ) );
        h = hashCombine( h,
                         mix64( static_cast<unsigned long long>( f.level ) ) );
        h = hashCombine( h, mix64( f.parent ) );
        h = hashCombine( h, triHash( mix64( f.parentVerts[0] ),
                                     mix64( f.parentVerts[1] ),
                                     mix64( f.parentVerts[2] ) ) );
        c.add( h );
    }
    return c;
}

//! Multiset checksum of the RED layer, by gid: gid, corners IN ORDER (so a
//! winding change fails), and level.
static Chk redSigGid( const std::vector<RedFace>& red )
{
    Chk c;
    for ( const auto& r : red )
    {
        unsigned long long h = mix64( r.gid );
        for ( int k = 0; k < 3; ++k )
            h = hashCombine( h, mix64( r.v[k] ) );
        h = hashCombine( h,
                         mix64( static_cast<unsigned long long>( r.level ) ) );
        c.add( h );
    }
    return c;
}

template <class Exec, class ConfMesh>
static int case_idempotence( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    ConfMesh mesh( comm );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    {
        auto gidOf = [&]( int m )
        {
            const std::vector<VisibleFace> v = ownedVisible( mesh );
            std::vector<char> mask( v.size(), 0 );
            for ( std::size_t f = 0; f < v.size(); ++f )
                mask[f] = ( v[f].gid % m == 0 ) ? 1 : 0;
            return mask;
        };
        refine( mesh, halo, gidOf( 7 ) );
        refine( mesh, halo, gidOf( 5 ) );
    }

    auto snapshot =
        [&]( Chk& vs, Chk& rs, long long& V, long long& E, long long& F )
    {
        const std::vector<VisibleFace> vis = ownedVisible( mesh );
        vs = visibleSigNoGid( vis );
        rs = redSigGid( unclose( vis ).red );
        vs.reduce( comm );
        rs.reduce( comm );
        V = TesseraTest::globalOwnedVertices( mesh );
        E = TesseraTest::globalOwnedEdges( mesh );
        F = TesseraTest::globalOwnedFaces( mesh );
    };

    Chk vs0, rs0;
    long long V0, E0, F0;
    snapshot( vs0, rs0, V0, E0, F0 );
    const long long closure0 =
        globalSum( TesseraTest::closureSiblingGroups( mesh ), comm );
    if ( closure0 <= 0 )
        ++fails; // vacuous: no closure to be idempotent about

    // Two empty-mask rounds. Each is un-close -> nothing splits -> re-close, so
    // each must reproduce the visible layer it started from.
    for ( int pass = 0; pass < 2; ++pass )
    {
        std::vector<char> empty( mesh.numOwnedFaces(), 0 );
        auto res = refine( mesh, halo, empty );

        Chk vs1, rs1;
        long long V1, E1, F1;
        snapshot( vs1, rs1, V1, E1, F1 );

        if ( vs1 != vs0 || rs1 != rs0 )
            ++fails;
        if ( V1 != V0 || E1 != E0 || F1 != F0 )
            ++fails;
        if ( !res.midpoints.empty() )
            ++fails; // an empty mask bisects nothing

        int local = TesseraTest::checkConforming( mesh );
        local += TesseraTest::check21BalanceRed( mesh );
        local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
        int g = 0;
        MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
        if ( g != 0 )
            ++fails;
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++fails;

        std::vector<Rank> dest( mesh.numOwnedFaces(),
                                static_cast<Rank>( rank ) );
        migrate( mesh, halo, dest );
        haloExchange( mesh, halo );
    }

    (void)size;
    if ( rank == 0 )
        std::printf( "  [%s] closure-idempotence %s (V=%lld E=%lld F=%lld "
                     "siblingGroups=%lld)\n",
                     tag, fails == 0 ? "ok" : "FAIL", V0, E0, F0, closure0 );
    return fails;
}

// ---------------------------------------------------------------------------
// Case C -- cross-mode equivalence under a uniform mask
// ---------------------------------------------------------------------------

//! Full topology checksum BY GID: (face gid, sorted corner gids, level) over
//! owned faces, plus the owned vertex/edge gid checksums. Valid to compare
//! across the two MODES (they share the red engine and its exscans), unlike
//! across rank counts.
template <class MeshT>
static void modeSig( MeshT& mesh, Chk& faceChk, unsigned long long& cv,
                     unsigned long long& ce, unsigned long long& cf,
                     long long& V, long long& E, long long& F )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    auto fl = Cabana::slice<FaceField::Level>( hf );

    faceChk = Chk{};
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        unsigned long long h = mix64( fg( f ) );
        h = hashCombine( h, triHash( mix64( fv( f, 0 ) ), mix64( fv( f, 1 ) ),
                                     mix64( fv( f, 2 ) ) ) );
        h = hashCombine( h,
                         mix64( static_cast<unsigned long long>( fl( f ) ) ) );
        faceChk.add( h );
    }
    faceChk.reduce( mesh.comm() );
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    V = TesseraTest::globalOwnedVertices( mesh );
    E = TesseraTest::globalOwnedEdges( mesh );
    F = TesseraTest::globalOwnedFaces( mesh );
}

template <class Exec, class ConfMesh, class HangMesh>
static int case_cross_mode( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    ConfMesh conf( comm );
    buildIcosphere( conf, 2 );
    MeshHalo<mem> confHalo;
    {
        auto faceOwner = facePartitionByAxis( conf );
        distribute( conf, confHalo, faceOwner );
    }
    HangMesh hang( comm );
    buildIcosphere( hang, 2 );
    MeshHalo<mem> hangHalo;
    {
        auto faceOwner = facePartitionByAxis( hang );
        distribute( hang, hangHalo, faceOwner );
    }

    long long closureChildren = 0;
    for ( int round = 0; round < 2; ++round )
    {
        std::vector<char> mc( conf.numOwnedFaces(), 1 );
        auto res = refine( conf, confHalo, mc );
        closureChildren += globalSum(
            static_cast<long long>( res.closure.nClosureChildren ), comm );

        std::vector<char> mh( hang.numOwnedFaces(), 1 );
        refine( hang, hangHalo, mh );

        Chk fc, fh;
        unsigned long long cvc, cec, cfc, cvh, ceh, cfh;
        long long Vc, Ec, Fc, Vh, Eh, Fh;
        modeSig( conf, fc, cvc, cec, cfc, Vc, Ec, Fc );
        modeSig( hang, fh, cvh, ceh, cfh, Vh, Eh, Fh );

        if ( fc != fh )
            ++fails; // face gids / corners / levels must be identical
        if ( cvc != cvh || cec != ceh || cfc != cfh )
            ++fails;
        if ( Vc != Vh || Ec != Eh || Fc != Fh )
            ++fails;

        if ( rank == 0 )
            std::printf( "  [%s] cross-mode round%d %s (V=%lld E=%lld F=%lld "
                         "closureChildren=%lld)\n",
                         tag, round + 1, fails == 0 ? "ok" : "FAIL", Vc, Ec, Fc,
                         closureChildren );
    }

    // With no kept faces the closure has nothing to do; if it emitted anything
    // the two modes cannot be equivalent and the comparison above was hiding it.
    if ( closureChildren != 0 )
        ++fails;

    (void)size;
    return fails;
}

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using ConfMesh = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::Conforming>;
    using HangMesh = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::HangingNode2to1>;

    int fails = 0;
    fails += case_rank_count<Exec, ConfMesh>( rank, size, tag );
    fails += case_idempotence<Exec, ConfMesh>( rank, size, tag );
    fails += case_cross_mode<Exec, ConfMesh, HangMesh>( rank, size, tag );
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
            std::printf( "test_conforming_determinism: rank-count "
                         "independence, idempotence, cross-mode (size %d)\n",
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
