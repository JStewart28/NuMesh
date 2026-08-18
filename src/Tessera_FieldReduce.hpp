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

#ifndef TESSERA_FIELD_REDUCE_HPP
#define TESSERA_FIELD_REDUCE_HPP

#include "Tessera_Geometry.hpp"

#include <Kokkos_Core.hpp>

namespace Tessera
{

// ============================================================================
// reduceVertexFromFaces — face -> vertex reduction with a caller-supplied op
// ============================================================================
//
// For every OWNED vertex v, Tessera walks the faces incident to v (the
// vertex->faces CSR, local indices) and invokes the caller's device functor
//     op( v, f, geom, faceSlice, vertSlice )
// once per incident face f. Tessera owns ONLY the iteration; the op owns the
// accumulation and the convention. This is the primitive behind vertex normals
// (area-weighted vs cross-product average) and vertex areas (barycentric /
// Voronoi / one-third) — Tessera picks none of them.
//
// Each owned vertex is handled by a single thread that visits its faces in
// sequence, so an op may accumulate into vertSlice(v, ...) directly with no
// atomics. The op should initialise vertSlice(v, ...) on the first visit (or
// the caller should zero it beforehand). `geom` (a MeshGeometry) is passed so
// the op can use the raw geometric primitives (faceArea, faceNormalRaw, ...);
// capturing it here also revalidates it against the mesh generation.
//
// The vertex->faces CSR is read fresh from the mesh at call time (local
// indices), so this call has no stale-handle hazard of its own; `geom` must be
// current (rebuilt after the last topology op).
template <class MeshT, class FaceSlice, class VertSlice, class ReduceOp>
void reduceVertexFromFaces( MeshT& mesh, const MeshGeometry<MeshT>& geom,
                            FaceSlice faceSlice, VertSlice vertSlice,
                            ReduceOp op )
{
    using execution_space = typename MeshT::execution_space;

    const int n_owned = static_cast<int>( mesh.numOwnedVertices() );
    auto offsets = mesh.vertexFaces().offsets;
    auto neighbors = mesh.vertexFaces().neighbors;
    MeshGeometry<MeshT> g = geom; // copy -> revalidates against mesh generation

    Kokkos::parallel_for(
        "tessera_reduce_vertex_from_faces",
        Kokkos::RangePolicy<execution_space>( 0, n_owned ),
        KOKKOS_LAMBDA( const int v ) {
            for ( int p = offsets( v ); p < offsets( v + 1 ); ++p )
            {
                const int f = neighbors( p );
                op( v, f, g, faceSlice, vertSlice );
            }
        } );
    Kokkos::fence();
}

} // namespace Tessera

#endif // TESSERA_FIELD_REDUCE_HPP
