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

#ifndef TESSERA_MESH_HPP
#define TESSERA_MESH_HPP

#include "Tessera_CsrAdjacency.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <string>

namespace Tessera
{

// ============================================================================
// Mesh
// ============================================================================
//
// A distributed unstructured triangle mesh: vertices, edges, and faces stored
// in three Cabana AoSoAs with full both-direction connectivity for 1-ring
// stencils. This header defines the DATA MODEL (Step 2): the templated
// container, its AoSoA/member types, slice accessors, and the canonical-key
// side tables. The serial builder (Step 3), distribution/halo (Step 5),
// refinement (Step 6), migration/load-balancing (Step 7), and I/O (Step 8) are
// added on top.
//
// Template parameters:
//   Scalar          - Floating-point precision for coordinates and user fields
//                     (user's choice; never hard-coded).
//   Dim             - Coordinate embedding dimension (default 3). Milestone 1
//                     uses Dim=3 (a closed surface in R^3); Dim=2 (planar mesh)
//                     is expressible with no retrofit.
//   VertexUserFields- Compile-time user field pack for vertices, e.g.
//                     VertexFields<Scalar[2]>, appended after the core members.
//   EdgeUserFields  - Compile-time user field pack for edges.
//   FaceUserFields  - Compile-time user field pack for faces.
//   MemorySpace     - Kokkos memory space for entity storage.
//   ExecutionSpace  - Kokkos execution space for kernels.
//
template <class Scalar, int Dim = 3, class VertexUserFields = VertexFields<>,
          class EdgeUserFields = EdgeFields<>,
          class FaceUserFields = FaceFields<>,
          class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace>
class Mesh
{
  public:
    // -- public type aliases --------------------------------------------------
    using scalar_type = Scalar;
    static constexpr int dim = Dim;
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    using vertex_member_types =
        VertexMemberTypes<Scalar, Dim, VertexUserFields>;
    using edge_member_types = EdgeMemberTypes<EdgeUserFields>;
    using face_member_types = FaceMemberTypes<FaceUserFields>;

    using vertex_aosoa_type = Cabana::AoSoA<vertex_member_types, MemorySpace>;
    using edge_aosoa_type = Cabana::AoSoA<edge_member_types, MemorySpace>;
    using face_aosoa_type = Cabana::AoSoA<face_member_types, MemorySpace>;

    using csr_type = CsrAdjacency<MemorySpace>;

    // -- construction ---------------------------------------------------------
    //! Construct an empty mesh on the given MPI communicator. No entities are
    //! allocated; the builder / distribution steps populate the AoSoAs.
    explicit Mesh( MPI_Comm comm )
        : _comm( comm )
        , _vertices( "tessera_vertices" )
        , _edges( "tessera_edges" )
        , _faces( "tessera_faces" )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    }

    // -- MPI ------------------------------------------------------------------
    MPI_Comm comm() const { return _comm; }
    int rank() const { return _rank; }
    int commSize() const { return _comm_size; }

    // -- sizes (owned + ghost entities currently held locally) ----------------
    std::size_t numVertices() const { return _vertices.size(); }
    std::size_t numEdges() const { return _edges.size(); }
    std::size_t numFaces() const { return _faces.size(); }

    // -- resize (used by builder / distribution / refinement) -----------------
    void resizeVertices( std::size_t n ) { _vertices.resize( n ); }
    void resizeEdges( std::size_t n ) { _edges.resize( n ); }
    void resizeFaces( std::size_t n ) { _faces.resize( n ); }

    // -- AoSoA accessors ------------------------------------------------------
    vertex_aosoa_type& vertices() { return _vertices; }
    const vertex_aosoa_type& vertices() const { return _vertices; }
    edge_aosoa_type& edges() { return _edges; }
    const edge_aosoa_type& edges() const { return _edges; }
    face_aosoa_type& faces() { return _faces; }
    const face_aosoa_type& faces() const { return _faces; }

    // -- slice accessors ------------------------------------------------------
    //
    // Core fields via the named indices in Tessera::{Vertex,Edge,Face}Field,
    // user fields via Tessera::userVertexField<M>() etc. Example:
    //   auto pos  = mesh.vertexSlice<Tessera::VertexField::Position>();
    //   auto vort = mesh.vertexSlice<Tessera::userVertexField<0>()>();
    template <std::size_t M>
    auto vertexSlice()
    {
        return Cabana::slice<M>( _vertices );
    }
    template <std::size_t M>
    auto edgeSlice()
    {
        return Cabana::slice<M>( _edges );
    }
    template <std::size_t M>
    auto faceSlice()
    {
        return Cabana::slice<M>( _faces );
    }

    // -- vertex 1-ring adjacency (CSR; built in Step 3, rebuilt on topo change)
    csr_type& vertexEdges() { return _vertexEdges; }
    const csr_type& vertexEdges() const { return _vertexEdges; }
    csr_type& vertexFaces() { return _vertexFaces; }
    const csr_type& vertexFaces() const { return _vertexFaces; }

    // -- canonical-key side tables --------------------------------------------
    //
    // The 128-bit structured keys (EdgeKey = pair of endpoint gids, FaceKey =
    // triple of corner gids) are the cross-rank matching identity, kept
    // parallel to the edge/face AoSoAs but OUT of them (they are only consulted
    // during halo/ghost matching and migration, so the per-entity AoSoA
    // footprint stays small). Allocated on demand by the steps that need them.
    Kokkos::View<EdgeKey*, MemorySpace>& edgeKeys() { return _edgeKeys; }
    const Kokkos::View<EdgeKey*, MemorySpace>& edgeKeys() const
    {
        return _edgeKeys;
    }
    Kokkos::View<FaceKey*, MemorySpace>& faceKeys() { return _faceKeys; }
    const Kokkos::View<FaceKey*, MemorySpace>& faceKeys() const
    {
        return _faceKeys;
    }

  private:
    MPI_Comm _comm;
    int _rank = 0;
    int _comm_size = 1;

    vertex_aosoa_type _vertices;
    edge_aosoa_type _edges;
    face_aosoa_type _faces;

    csr_type _vertexEdges;
    csr_type _vertexFaces;

    Kokkos::View<EdgeKey*, MemorySpace> _edgeKeys;
    Kokkos::View<FaceKey*, MemorySpace> _faceKeys;
};

} // namespace Tessera

#endif // TESSERA_MESH_HPP
