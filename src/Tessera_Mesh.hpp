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
#include "Tessera_GenerationGuard.hpp"
#include "Tessera_RefinementMode.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
//   Mode            - Which conformity contract refine()/refineLocal() obey
//                     (Tessera_RefinementMode.hpp). Appended LAST so every
//                     existing seven-argument Mesh<...> spelling still compiles.
//                     DEFAULTS TO Conforming: a mesh with no hanging nodes is
//                     what a surface operator (applyStencil,
//                     reduceVertexFromFaces) needs to be consistent, so it is
//                     the safe default and HangingNode2to1 is the opt-in
//                     cheaper mode. In Conforming the face member list carries
//                     two extra closure-bookkeeping members after the user pack;
//                     FaceField::UserBegin and userFaceField<M>() are identical
//                     in both modes.
//
template <class Scalar, int Dim = 3, class VertexUserFields = VertexFields<>,
          class EdgeUserFields = EdgeFields<>,
          class FaceUserFields = FaceFields<>,
          class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace,
          RefinementMode Mode = RefinementMode::Conforming>
class Mesh
{
  public:
    // -- public type aliases --------------------------------------------------
    using scalar_type = Scalar;
    static constexpr int dim = Dim;
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    //! Refinement conformity contract; refine()/refineLocal() dispatch on this.
    static constexpr RefinementMode refinement_mode = Mode;

    //! The user field packs, re-exported so generic code can size a user-field
    //! loop without assuming the tuple ends at the user pack (it does not in
    //! Conforming mode -- see numFaceUserFields<>() in Tessera_Fields.hpp).
    using vertex_user_fields = VertexUserFields;
    using edge_user_fields = EdgeUserFields;
    using face_user_fields = FaceUserFields;

    using vertex_member_types =
        VertexMemberTypes<Scalar, Dim, VertexUserFields>;
    using edge_member_types = EdgeMemberTypes<EdgeUserFields>;
    using face_member_types = FaceMemberTypes<FaceUserFields, Mode>;

    //! Slice indices of the Conforming-mode closure members. Slicing these on a
    //! HangingNode2to1 mesh is a compile error (the members do not exist);
    //! guard uses with `if constexpr ( refinement_mode == ... )`.
    static constexpr std::size_t closure_parent_field =
        closureParentField<FaceUserFields>();
    static constexpr std::size_t closure_parent_verts_field =
        closureParentVertsField<FaceUserFields>();

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

    // -- distributed-state counts ---------------------------------------------
    //
    // Convention: entities are stored owned-first. Local indices [0, numOwnedX)
    // are owned by this rank; [numOwnedX, numX) are ghosts (owned by a neighbour
    // and kept locally to complete the 1-deep halo). For a replicated / serial
    // mesh every entity is owned. distribute() (Step 5) sets these; migration and
    // refinement (Steps 6-7) update them.
    //
    // INVALIDATION: any change to the local entity count or the ghost set
    // (distribution, migration, refinement) invalidates every slice/CSR/
    // key-View handed out before the change -- setOwnedCounts() bumps
    // generation() (see below) so GenerationHandle-wrapped handles taken
    // beforehand fail loudly on next use instead of silently reading stale
    // storage.
    std::size_t numOwnedVertices() const { return _n_owned_v; }
    std::size_t numOwnedEdges() const { return _n_owned_e; }
    std::size_t numOwnedFaces() const { return _n_owned_f; }
    void setOwnedCounts( std::size_t nv, std::size_t ne, std::size_t nf )
    {
        _n_owned_v = nv;
        _n_owned_e = ne;
        _n_owned_f = nf;
        bumpGeneration();
    }

    // -- generation counter ----------------------------------------------------
    //
    // Monotonically increases every time this mesh's local entity count or
    // storage is reallocated (resize*, setOwnedCounts, the key-View/CSR
    // rebuild methods below). Slice/CSR/key-View accessors stamp the handles
    // they return with the generation at creation time (see
    // Tessera_GenerationGuard.hpp); haloExchange() never bumps this counter,
    // since it is topology-preserving. Free functions that reallocate mesh
    // storage without going through this class (e.g. Tessera_Migrate.hpp's
    // mesh-agnostic migrate() primitive, if ever invoked directly on
    // mesh.vertices()/edges()/faces()) must call the public bumpGeneration()
    // themselves -- see the note at Tessera_Migrate.hpp's reassignment site.
    std::size_t generation() const { return _generation; }
    void bumpGeneration() { ++_generation; }

    //! Pointer to the live generation counter. For stamping externally-built
    //! generation-guarded handles (Tessera_GenerationGuard.hpp) whose backing
    //! storage is derived from this mesh but not handed out by it directly --
    //! e.g. the k-ring stencil CSR from buildVertexStencil(). The pointer stays
    //! valid for the mesh's lifetime; the counter it addresses is bumped by the
    //! same reallocation sites that invalidate slices.
    const std::size_t* generationPtr() const { return &_generation; }

    // -- resize (used by builder / distribution / refinement) -----------------
    void resizeVertices( std::size_t n )
    {
        _vertices.resize( n );
        bumpGeneration();
    }
    void resizeEdges( std::size_t n )
    {
        _edges.resize( n );
        bumpGeneration();
    }
    void resizeFaces( std::size_t n )
    {
        _faces.resize( n );
        bumpGeneration();
    }

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
    //
    // The returned handle is a GenerationHandle wrapping the Cabana::slice: it
    // forwards operator() unchanged (no device-side cost) but aborts with a
    // diagnostic if copied (e.g. captured into a KOKKOS_LAMBDA) after this
    // mesh's generation() has advanced past the slice's capture point -- see
    // Tessera_GenerationGuard.hpp for the INVALIDATION contract.
    template <std::size_t M>
    auto vertexSlice()
    {
        using slice_type = decltype( Cabana::slice<M>( _vertices ) );
        return GenerationHandle<slice_type>( Cabana::slice<M>( _vertices ),
                                             _generation, &_generation );
    }
    template <std::size_t M>
    auto edgeSlice()
    {
        using slice_type = decltype( Cabana::slice<M>( _edges ) );
        return GenerationHandle<slice_type>( Cabana::slice<M>( _edges ),
                                             _generation, &_generation );
    }
    template <std::size_t M>
    auto faceSlice()
    {
        using slice_type = decltype( Cabana::slice<M>( _faces ) );
        return GenerationHandle<slice_type>( Cabana::slice<M>( _faces ),
                                             _generation, &_generation );
    }

    //! Convenience re-slice helpers: re-derive a whole set of generation-
    //! stamped slices in one structural call, e.g. at the top of every solver
    //! stage, rather than re-slicing each field individually.
    //!   auto [pos, vort] =
    //!       mesh.vertexSlices<VertexField::Position, userVertexField<0>()>();
    template <std::size_t... M>
    auto vertexSlices()
    {
        return std::make_tuple( vertexSlice<M>()... );
    }
    template <std::size_t... M>
    auto edgeSlices()
    {
        return std::make_tuple( edgeSlice<M>()... );
    }
    template <std::size_t... M>
    auto faceSlices()
    {
        return std::make_tuple( faceSlice<M>()... );
    }

    // -- vertex 1-ring adjacency (CSR; built in Step 3, rebuilt on topo change)
    //
    // INVALIDATION: rebuildVertexFaces()/rebuildVertexEdges() below replace the
    // CSR's internal offsets/neighbors Views wholesale and bump generation();
    // these bare accessors are for internal, same-scope use that does not
    // outlive a topology-changing call. External/solver code that wants a
    // guarded handle should use vertexFacesHandle()/vertexEdgesHandle().
    csr_type& vertexEdges() { return _vertexEdges; }
    const csr_type& vertexEdges() const { return _vertexEdges; }
    csr_type& vertexFaces() { return _vertexFaces; }
    const csr_type& vertexFaces() const { return _vertexFaces; }

    //! Generation-stamped snapshot of the CSR relation (a cheap copy: two
    //! Kokkos::Views). Use this to hold a guarded handle across a call that
    //! might reallocate the CSR.
    GenerationHandle<csr_type> vertexEdgesHandle() const
    {
        return GenerationHandle<csr_type>( _vertexEdges, _generation,
                                           &_generation );
    }
    GenerationHandle<csr_type> vertexFacesHandle() const
    {
        return GenerationHandle<csr_type>( _vertexFaces, _generation,
                                           &_generation );
    }

    //! Rebuild the vertex->faces / vertex->edges CSR from host offset/neighbor
    //! vectors, replacing the CSR's storage and bumping generation(). This is
    //! the sanctioned replacement for calling detail::fillCsr() directly on
    //! vertexFaces()/vertexEdges().
    void rebuildVertexFaces( const std::vector<int>& offsets,
                             const std::vector<LocalIndex>& neighbors,
                             const std::string& label )
    {
        detail::fillCsr( _vertexFaces, offsets, neighbors, label );
        bumpGeneration();
    }
    void rebuildVertexEdges( const std::vector<int>& offsets,
                             const std::vector<LocalIndex>& neighbors,
                             const std::string& label )
    {
        detail::fillCsr( _vertexEdges, offsets, neighbors, label );
        bumpGeneration();
    }

    // -- canonical-key side tables --------------------------------------------
    //
    // The 128-bit structured keys (EdgeKey = pair of endpoint gids, FaceKey =
    // triple of corner gids) are the cross-rank matching identity, kept
    // parallel to the edge/face AoSoAs but OUT of them (they are only consulted
    // during halo/ghost matching and migration, so the per-entity AoSoA
    // footprint stays small). Allocated on demand by the steps that need them.
    //
    // INVALIDATION: setEdgeKeys()/setFaceKeys() below replace these Views
    // wholesale and bump generation(); the bare reference accessors are for
    // internal, same-scope use (and for filling the table right after
    // construction, before any generation-stamped handle exists yet). External/
    // solver code that wants a guarded handle should use
    // edgeKeysHandle()/faceKeysHandle().
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

    //! Replace the edge/face key-View wholesale, bumping generation().
    void setEdgeKeys( Kokkos::View<EdgeKey*, MemorySpace> keys )
    {
        _edgeKeys = std::move( keys );
        bumpGeneration();
    }
    void setFaceKeys( Kokkos::View<FaceKey*, MemorySpace> keys )
    {
        _faceKeys = std::move( keys );
        bumpGeneration();
    }

    GenerationHandle<Kokkos::View<EdgeKey*, MemorySpace>> edgeKeysHandle() const
    {
        return GenerationHandle<Kokkos::View<EdgeKey*, MemorySpace>>(
            _edgeKeys, _generation, &_generation );
    }
    GenerationHandle<Kokkos::View<FaceKey*, MemorySpace>> faceKeysHandle() const
    {
        return GenerationHandle<Kokkos::View<FaceKey*, MemorySpace>>(
            _faceKeys, _generation, &_generation );
    }

  private:
    MPI_Comm _comm;
    int _rank = 0;
    int _comm_size = 1;

    std::size_t _n_owned_v = 0;
    std::size_t _n_owned_e = 0;
    std::size_t _n_owned_f = 0;

    std::size_t _generation = 0;

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
