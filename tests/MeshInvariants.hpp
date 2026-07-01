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

#ifndef TESSERA_TEST_MESH_INVARIANTS_HPP
#define TESSERA_TEST_MESH_INVARIANTS_HPP

// Shared distributed-mesh invariant checks, reused by the Step 5/6/7/8 regression
// tests. Every helper returns a LOCAL failure count whose SUM across ranks is the
// true global failure count (collective helpers place each check on exactly one
// rank), so a caller can MPI_Allreduce(SUM) the total and treat non-zero as fail.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TesseraTest
{

using Tessera::GlobalId;
using Tessera::LocalIndex;
using Tessera::Rank;

// Owned gids [begin,end) of an AoSoA (gid is member 0 for every entity kind).
template <class AoSoAType>
std::vector<GlobalId> ownedGids( const AoSoAType& a, std::size_t begin,
                                 std::size_t end )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "h", a.size() );
    Cabana::deep_copy( h, a );
    auto g = Cabana::slice<0>( h );
    std::vector<GlobalId> out;
    out.reserve( end - begin );
    for ( std::size_t i = begin; i < end; ++i )
        out.push_back( g( i ) );
    return out;
}

// One entity kind's partition check: (1) total owned across ranks == Nglobal
// (checked on rank 0 only), (2) no gid owned by two ranks (each gid routed to a
// coordinator rank = gid % size, which flags duplicates). Returns LOCAL fails.
inline int checkKindPartition( MPI_Comm comm, int rank, int comm_size,
                               const std::vector<GlobalId>& owned,
                               long long Nglobal )
{
    long long local = static_cast<long long>( owned.size() );
    long long total = 0;
    MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, comm );

    int fails = 0;
    if ( rank == 0 && total != Nglobal )
        ++fails;

    std::vector<std::vector<GlobalId>> send( comm_size );
    for ( GlobalId g : owned )
        send[g % comm_size].push_back( g );
    auto res = Tessera::allToAllV( comm, send );
    std::set<GlobalId> seen;
    for ( GlobalId g : res.data )
        if ( !seen.insert( g ).second )
            ++fails; // this gid was owned by more than one rank
    return fails;
}

// Ownership is a partition of every entity kind. LOCAL fails (sum == global).
template <class MeshT>
int checkOwnershipPartition( MeshT& mesh, long long NvG, long long NeG,
                             long long NfG )
{
    MPI_Comm comm = mesh.comm();
    const int rank = mesh.rank();
    const int size = mesh.commSize();
    int fails = 0;
    fails += checkKindPartition(
        comm, rank, size,
        ownedGids( mesh.vertices(), 0, mesh.numOwnedVertices() ), NvG );
    fails += checkKindPartition(
        comm, rank, size, ownedGids( mesh.edges(), 0, mesh.numOwnedEdges() ),
        NeG );
    fails += checkKindPartition(
        comm, rank, size, ownedGids( mesh.faces(), 0, mesh.numOwnedFaces() ),
        NfG );
    return fails;
}

// Every owned vertex holds its full 1-ring locally: each incident face contains
// the vertex, and that face's three vertices and three edges are all present
// locally. LOCAL fails (no communication; sum across ranks == global).
template <class MeshT>
int owned1RingLocal( MeshT& mesh )
{
    const std::size_t nv = mesh.numVertices();
    const std::size_t nf = mesh.numFaces();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vgid = Cabana::slice<Tessera::VertexField::Gid>( hv );
    auto egid = Cabana::slice<Tessera::EdgeField::Gid>( he );
    auto fverts = Cabana::slice<Tessera::FaceField::Verts>( hf );
    auto fedges = Cabana::slice<Tessera::FaceField::Edges>( hf );

    std::unordered_set<GlobalId> vset, eset;
    for ( std::size_t i = 0; i < nv; ++i )
        vset.insert( vgid( i ) );
    for ( std::size_t i = 0; i < mesh.numEdges(); ++i )
        eset.insert( egid( i ) );

    auto off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().neighbors );

    int fails = 0;
    for ( std::size_t v = 0; v < mesh.numOwnedVertices(); ++v )
    {
        const GlobalId vg = vgid( v );
        for ( int p = off( v ); p < off( v + 1 ); ++p )
        {
            const LocalIndex lf = nbr( p );
            bool has = false;
            for ( int k = 0; k < 3; ++k )
            {
                if ( fverts( lf, k ) == vg )
                    has = true;
                if ( vset.find( fverts( lf, k ) ) == vset.end() )
                    ++fails; // adjacent vertex not held locally
                if ( eset.find( fedges( lf, k ) ) == eset.end() )
                    ++fails; // incident edge not held locally
            }
            if ( !has )
                ++fails; // CSR listed a face not incident to this vertex
        }
    }
    return fails;
}

// Rank-count-independent topology checksum: XOR of owned gids per kind, reduced
// with MPI_BXOR so every rank gets the same value. Two distributions of the same
// global mesh (e.g. at different rank counts, or before/after I/O) agree iff their
// checksums agree. Used by Steps 6b/7/8.
template <class MeshT>
void topologyChecksum( MeshT& mesh, unsigned long long& cv,
                       unsigned long long& ce, unsigned long long& cf )
{
    auto xorKind = [&]( const std::vector<GlobalId>& owned )
    {
        unsigned long long local = 0;
        for ( GlobalId g : owned )
            local ^= static_cast<unsigned long long>( g );
        unsigned long long global = 0;
        MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                       mesh.comm() );
        return global;
    };
    cv = xorKind( ownedGids( mesh.vertices(), 0, mesh.numOwnedVertices() ) );
    ce = xorKind( ownedGids( mesh.edges(), 0, mesh.numOwnedEdges() ) );
    cf = xorKind( ownedGids( mesh.faces(), 0, mesh.numOwnedFaces() ) );
}

// ---------------------------------------------------------------------------
// Step 6b (distributed refinement) invariants
// ---------------------------------------------------------------------------

// Global owned-only Euler number Σ_ranks( ownedV - ownedE + ownedF ). For a
// conforming closed genus-0 surface (e.g. after a UNIFORM refine) this is 2;
// adaptive refinement introduces bounded hanging nodes and does not preserve it.
template <class MeshT>
long long ownedEulerGlobal( MeshT& mesh )
{
    long long local = static_cast<long long>( mesh.numOwnedVertices() ) -
                      static_cast<long long>( mesh.numOwnedEdges() ) +
                      static_cast<long long>( mesh.numOwnedFaces() );
    long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    return global;
}

// Sum of an owned count across ranks (owned entities partition the global mesh).
template <class MeshT>
long long globalOwnedVertices( MeshT& mesh )
{
    long long l = static_cast<long long>( mesh.numOwnedVertices() ), g = 0;
    MPI_Allreduce( &l, &g, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    return g;
}
template <class MeshT>
long long globalOwnedEdges( MeshT& mesh )
{
    long long l = static_cast<long long>( mesh.numOwnedEdges() ), g = 0;
    MPI_Allreduce( &l, &g, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    return g;
}
template <class MeshT>
long long globalOwnedFaces( MeshT& mesh )
{
    long long l = static_cast<long long>( mesh.numOwnedFaces() ), g = 0;
    MPI_Allreduce( &l, &g, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    return g;
}

// 2:1 balance: no edge's two incident (owned) faces differ by more than one
// refinement level. Each face advertises (edge, level) to the edge's coordinator,
// which compares the two incidences. Returns LOCAL fails (sum == global).
template <class MeshT>
int check21Balance( MeshT& mesh )
{
    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();
    const std::size_t nof = mesh.numOwnedFaces();

    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<Tessera::FaceField::Verts>( hf );
    auto fl = Cabana::slice<Tessera::FaceField::Level>( hf );

    struct LvMsg
    {
        Tessera::EdgeKey key;
        Tessera::Level level;
    };
    std::vector<std::vector<LvMsg>> send( size );
    for ( std::size_t f = 0; f < nof; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const Tessera::EdgeKey key =
                Tessera::makeEdgeKey( fv( f, k ), fv( f, ( k + 1 ) % 3 ) );
            send[Tessera::detail::edgeCoordRank( key, size )].push_back(
                { key, fl( f ) } );
        }
    auto got = Tessera::allToAllV( comm, send );

    std::map<Tessera::EdgeKey, std::vector<Tessera::Level>> byEdge;
    for ( const auto& m : got.data )
        byEdge[m.key].push_back( m.level );

    int fails = 0;
    for ( const auto& kv : byEdge )
        if ( kv.second.size() == 2 )
        {
            const int d = static_cast<int>( kv.second[0] ) -
                          static_cast<int>( kv.second[1] );
            if ( d > 1 || d < -1 )
                ++fails;
        }
    return fails;
}

// Cross-rank midpoint-gid agreement (the key Step-6b guarantee): every rank that
// creates a midpoint for a shared edge must use the same gid. Each (edge, gid)
// pair is routed to the edge's coordinator, which flags any edge seen with two
// distinct gids. Returns LOCAL fails (sum == global).
inline int checkMidpointAgreement(
    MPI_Comm comm, int size,
    const std::vector<std::pair<Tessera::EdgeKey, Tessera::GlobalId>>& mids )
{
    struct KG
    {
        Tessera::EdgeKey key;
        Tessera::GlobalId gid;
    };
    std::vector<std::vector<KG>> send( size );
    for ( const auto& kg : mids )
        send[Tessera::detail::edgeCoordRank( kg.first, size )].push_back(
            { kg.first, kg.second } );
    auto got = Tessera::allToAllV( comm, send );

    std::map<Tessera::EdgeKey, Tessera::GlobalId> seen;
    int fails = 0;
    for ( const auto& m : got.data )
    {
        auto it = seen.find( m.key );
        if ( it == seen.end() )
            seen.emplace( m.key, m.gid );
        else if ( it->second != m.gid )
            ++fails; // same edge, different midpoint gid across ranks
    }
    return fails;
}

} // namespace TesseraTest

#endif // TESSERA_TEST_MESH_INVARIANTS_HPP
