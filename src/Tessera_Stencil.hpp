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

#ifndef TESSERA_STENCIL_HPP
#define TESSERA_STENCIL_HPP

#include "Tessera_CsrAdjacency.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_GenerationGuard.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Tessera
{

// ============================================================================
// VertexStencil — k-ring vertex neighbour topology
// ============================================================================
//
// A CSR of the k-ring vertex neighbours of every local vertex, in local
// indices, plus the ring order k it was built for. This is pure TOPOLOGY: it
// carries no weights and no convention. A caller builds its weight View over
// this same sparsity pattern (RBF-FD, cotangent, uniform, ...) — the weight
// scheme is the caller's, never Tessera's.
//
// INVALIDATION: the CSR is generation-guarded (GenerationHandle) exactly like
// Mesh::vertexFacesHandle(): it is stamped with the mesh generation at build
// time, and applyStencil() re-checks it against the mesh's live generation
// before launching. Rebuild with buildVertexStencil() after any topology op.
template <class MemorySpace>
struct VertexStencil
{
    using memory_space = MemorySpace;

    GenerationHandle<CsrAdjacency<MemorySpace>> csr;
    int k = 0;
};

// ============================================================================
// buildVertexStencil — k-ring vertex neighbour CSR via edge BFS
// ============================================================================
//
// Builds the k-ring (k>=1) vertex neighbourhood of every local vertex by BFS
// over the existing vertex->edge connectivity: the 1-ring is the set of
// opposite endpoints of a vertex's incident edges; the 2-ring adds the 1-rings
// of those, and so on to distance k. The source vertex itself is excluded.
// Runs on the host with the mesh's own (small, replicated-coarse or migrated)
// connectivity — the same locus as the coarse builder and the halo builder —
// then deep-copies the CSR to the device. k=1 and k=2 both cover the operator
// families the milestone targets.
//
// Neighbours are stored ascending by local index within each row for
// determinism. Coverage is complete for OWNED vertices provided the local halo
// depth is >= k, and that is now CHECKED rather than merely documented: a
// k-ring stencil on a mesh whose halo is shallower than k throws
// std::invalid_argument naming both numbers. Discharging it is the caller's
// responsibility and is expressible — build the mesh with
// `distribute( mesh, halo, faceOwner, k )` (or `rebuildHalo( mesh, halo, k )`)
// and refine()/migrate() preserve that depth thereafter.
//
// The check is skipped when mesh.haloDepth() == 0, which means "never
// distributed": a replicated mesh straight out of the builder holds every
// entity, so no ring can be missing at any k.
//
// Without the check a short CSR row looks exactly like a correct one, and an
// operator built on it produces a plausible field with a small error localized
// on partition boundaries that moves when the rank count changes.
template <class MeshT>
VertexStencil<typename MeshT::memory_space> buildVertexStencil( MeshT& mesh,
                                                                int k )
{
    using memory_space = typename MeshT::memory_space;

    if ( mesh.haloDepth() > 0 && k > mesh.haloDepth() )
        throw std::invalid_argument(
            std::string( "Tessera::buildVertexStencil: k=" ) +
            std::to_string( k ) + " exceeds the mesh halo depth " +
            std::to_string( mesh.haloDepth() ) +
            "; the k-rings of owned vertices on a partition boundary would be "
            "silently short. Rebuild the mesh with a halo of depth >= " +
            std::to_string( k ) +
            " (distribute(mesh, halo, faceOwner, depth) or rebuildHalo(mesh, "
            "halo, depth))." );

    const int nv = static_cast<int>( mesh.numVertices() );
    const int ne = static_cast<int>( mesh.numEdges() );

    // Host copy of edge endpoints (gids) + the vertex gid table for gid->local.
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "stencil_hv", nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "stencil_he", ne );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto e_v = Cabana::slice<EdgeField::Verts>( he );

    std::unordered_map<GlobalId, int> gid2local;
    gid2local.reserve( static_cast<std::size_t>( nv ) * 2 );
    for ( int i = 0; i < nv; ++i )
        gid2local[v_gid( i )] = i;

    // vertex 1-ring adjacency (local indices) from vertex->edges CSR.
    auto ve_off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexEdges().offsets );
    auto ve_nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexEdges().neighbors );

    std::vector<std::vector<int>> onering( nv );
    for ( int v = 0; v < nv; ++v )
        for ( int p = ve_off( v ); p < ve_off( v + 1 ); ++p )
        {
            const int e = ve_nbr( p );
            const int a = gid2local.count( e_v( e, 0 ) )
                              ? gid2local[e_v( e, 0 )]
                              : invalid_local;
            const int b = gid2local.count( e_v( e, 1 ) )
                              ? gid2local[e_v( e, 1 )]
                              : invalid_local;
            const int other = ( a == v ) ? b : a;
            if ( other != invalid_local && other != v )
                onering[v].push_back( other );
        }

    // BFS to distance k from each vertex (self excluded).
    std::vector<int> off( nv + 1, 0 );
    std::vector<LocalIndex> nbr;
    std::vector<int> dist( nv, -1 );
    std::vector<int> frontier, next_frontier, ring;
    for ( int v = 0; v < nv; ++v )
    {
        ring.clear();
        dist[v] = 0;
        frontier.assign( 1, v );
        for ( int d = 0; d < k; ++d )
        {
            next_frontier.clear();
            for ( int u : frontier )
                for ( int w : onering[u] )
                    if ( dist[w] < 0 )
                    {
                        dist[w] = d + 1;
                        ring.push_back( w );
                        next_frontier.push_back( w );
                    }
            frontier.swap( next_frontier );
        }
        // reset visited marks for the next source
        dist[v] = -1;
        for ( int w : ring )
            dist[w] = -1;
        std::sort( ring.begin(), ring.end() );
        for ( int w : ring )
            nbr.push_back( static_cast<LocalIndex>( w ) );
        off[v + 1] = static_cast<int>( nbr.size() );
    }

    CsrAdjacency<memory_space> csr;
    detail::fillCsr( csr, off, nbr, "vertex_stencil_k" + std::to_string( k ) );

    VertexStencil<memory_space> stencil;
    stencil.csr = GenerationHandle<CsrAdjacency<memory_space>>(
        csr, mesh.generation(), mesh.generationPtr() );
    stencil.k = k;
    return stencil;
}

// ============================================================================
// applyStencil — weighted stencil apply over a generic per-vertex field
// ============================================================================
//
// Computes, for every OWNED vertex i:  out(i) = sum_{p in row i} w(p)*in(nbr(p))
// where nbr/row come from the stencil CSR and w is a caller-supplied flat View
// aligned to the CSR neighbour array (length == stencil.csr.numEntries()). The
// weights ARE the convention (cotangent, RBF-FD, uniform, ...) and are the
// caller's to build; Tessera only applies them. Scalar-per-vertex fields
// (rank-1 slices); apply per component for vector fields.
//
// HALO CORRECTNESS: a row of an owned vertex may reference ghost neighbours, so
// `in` must have current ghost values — the caller must haloExchange() the
// `in` field before calling. Only owned rows [0, numOwnedVertices) are written;
// ghost `out` entries are left untouched (refresh them with a subsequent
// haloExchange if needed). GPU-resident: one device kernel, no host copy.
template <class MeshT, class WeightsView, class InSlice, class OutSlice>
void applyStencil( MeshT& mesh,
                   const VertexStencil<typename MeshT::memory_space>& stencil,
                   WeightsView w, InSlice in, OutSlice out )
{
    using execution_space = typename MeshT::execution_space;
    using Scalar = typename MeshT::scalar_type;

    // Host-side staleness check against the mesh's live generation.
    const CsrAdjacency<typename MeshT::memory_space>& csr = stencil.csr.get();

    const int n_owned = static_cast<int>( mesh.numOwnedVertices() );
    auto offsets = csr.offsets;
    auto neighbors = csr.neighbors;

    Kokkos::parallel_for(
        "tessera_apply_stencil",
        Kokkos::RangePolicy<execution_space>( 0, n_owned ),
        KOKKOS_LAMBDA( const int i ) {
            Scalar acc = Scalar( 0 );
            for ( int p = offsets( i ); p < offsets( i + 1 ); ++p )
                acc += w( p ) * in( neighbors( p ) );
            out( i ) = acc;
        } );
    Kokkos::fence();
}

} // namespace Tessera

#endif // TESSERA_STENCIL_HPP
