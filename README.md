# Tessera

A distributed, MPI- and GPU-aware **unstructured triangle-mesh library** built on
[Cabana](https://github.com/ECP-copa/Cabana) and
[Kokkos](https://github.com/kokkos/kokkos).

Tessera provides the local, halo-able mesh machinery for evolving closed surfaces:
entity storage, full connectivity, a hand-built 1-deep MPI halo, split-based
adaptive refinement, optional migration/load-balancing, and parallel
Paraview-readable I/O. It is a dependency of the [Canopy](https://github.com/) FMM
solver and the Beatnik/rocketrig Rayleigh–Taylor problem; the global
Birkhoff–Rott/FMM velocity solve lives downstream in Canopy and is **not** part of
Tessera.

> **Status: early development — design reference.** This README documents the
> *design* of milestone 1 (a single rising bubble). The API below is the target the
> implementation is being built against (see `plans/compiled-hopping-torvalds.md`
> and the `tasks/milestone1_mesh.md` handoff contract). Sections marked *(planned)*
> are not yet implemented.

---

## Design

### Concepts

A mesh is a set of three entity kinds — **vertices**, **edges**, **faces**
(triangles) — with full, both-direction connectivity sufficient for 1-ring
(one-deep-halo) stencils:

```
face  → 3 vertices, 3 edges
edge  → 2 vertices, ≤2 faces
vertex→ N edges, N faces      (variable valence, CSR)
```

The mesh is **distributed**: each MPI rank owns a subset of faces and the vertices/
edges that ownership implies, plus a **1-deep ghost layer** so every owned entity
has its entire 1-ring held locally (owned or ghosted) after a halo exchange.

The structure is **component-agnostic**: it is purely entities + connectivity, so a
surface that pinches into two or more disjoint surfaces is represented with no
special case. Traversal follows connectivity and therefore respects component
boundaries for free. (Topology surgery — pinch-off, edge collapse/flip — is
milestone 2+ and out of scope here, but the data model leaves room for it, including
an optional per-entity `component_id`.)

### Templated precision and embedding dimension

The mesh is templated on its **scalar type** and its **coordinate embedding
dimension** — the user chooses the floating-point precision (Tessera never
hard-codes `double`) and the ambient dimension (`Dim`, default 3):

```cpp
template <class Scalar, int Dim = 3,
          class VertexFields, class EdgeFields, class FaceFields,
          class MemorySpace, class ExecutionSpace>
class Mesh;
```

Coordinates and all user field data use `Scalar`; `position` is `Scalar[Dim]`.
Milestone 1 uses `Dim=3` (a closed surface lives in ℝ³); templating `Dim` keeps a
planar 2D triangle mesh (`Dim=2`) expressible later with no retrofit. All
coordinate-dependent paths (Zoltan2 centroid, normals/area, I/O) are written against
`Dim`. Global identifiers and local indices are integer-typed independent of `Scalar`
and `Dim` (see *Global IDs*).

### Data model — AoSoA layout

Per entity kind, Tessera stores two Cabana AoSoAs:

1. A **fixed core topology AoSoA** (integer-typed), holding identity, ownership, and
   connectivity:

   | Entity | Core fields |
   |---|---|
   | Vertex | `gid`, `owner`, `flags`, `position` (`Scalar[Dim]`, mandatory) |
   | Edge | `gid`, `owner`, `level`, `v[2]` (vertex gids), `f[2]` (face gids) |
   | Face | `gid`, `owner`, `level`, `v[3]` (vertex gids), `e[3]` (edge gids) |

2. A **templated user-state AoSoA** holding the compile-time **field pack** for that
   entity kind (see below).

Variable-valence vertex adjacency (`vertex → edges/faces`) is held in **CSR side
arrays** (offsets + neighbor lists), not in the AoSoA, so the AoSoA slices stay
fixed-arity and vectorizable. `position` is a core vertex field (not user data)
because the partitioner centroid, I/O, and local operators all require it. (For a
`Dim`-dimensional embedding, `position` is `Scalar[Dim]`.)

This design is sized for **100M+ entities** and leaves room for collapse/flip later.

### Per-entity data — compile-time field pack

Arbitrary user data of arbitrary type may live on any vertex, edge, or face. Fields
are declared at **compile time** as a Cabana field pack per entity kind (idiomatic
Cabana, matching downstream Canopy/Beatnik usage; fastest, typed slices in kernels):

```cpp
// e.g. positions (3 scalars) + vorticity (2 scalars) on vertices,
//      mean curvature (1 scalar) on faces, nothing extra on edges:
using VFields = Tessera::VertexFields<Tessera::Field::Vorticity /*Scalar[2]*/>;
using FFields = Tessera::FaceFields  <Tessera::Field::Curvature  /*Scalar*/>;
using EFields = Tessera::EdgeFields  <>;

using MeshT = Tessera::Mesh<double, /*Dim=*/3, VFields, EFields, FFields,
                            Kokkos::HIPSpace, Kokkos::HIP>;
```

**Every field in the pack is haloed and migrated with its entity automatically** —
the halo field-sync and the migration primitive are generic over the whole AoSoA
tuple, so adding a field requires no communication-layer changes. After any halo
exchange, `mesh.haloExchange()` has synced the full pack into the ghost entities.

### Global IDs — structured 128-bit keys

Cross-rank entity identity uses **structured keys, not hashes** (a 64-bit hash has a
~27% collision probability at 10⁸ entities, and a single collision would silently
merge two distinct entities across a partition boundary):

- **Vertex gid:** globally unique 64-bit.
- **Edge canonical key:** the ordered pair `(min(v0,v1), max(v0,v1))` of its endpoint
  vertex gids — a 128-bit key (`uint64[2]`) that is **identical on every rank with no
  communication**.
- **Refinement midpoint vertex gid:** derived deterministically from the edge's
  128-bit key (a reserved midpoint namespace), so a shared edge's midpoint vertex is
  **bit-identical on both sides** of a partition boundary, again with no comm.
- **Face child gids:** derived from `(parent key, child index)`.

The AoSoA carries a dense 64-bit local index for kernel use; the 128-bit canonical
key lives in a side table consulted only during halo/ghost matching and migration.

### Ownership

A face is owned uniquely by the rank the partition assigns it to. A **shared vertex
or edge is owned by the lowest rank** in its *sharing set* — the set of ranks that
hold it (owned or ghost) after the halo is built. This rule is deterministic and
needs no vote/tie-break. The sharing set is materialized by the ghost-build neighbor
exchange (neighbor-bounded, **not** all-to-all).

### 1-deep halo

After `haloExchange()`, every owned vertex has all incident edges/faces and the
opposite vertices of its 1-ring locally; every owned face has its 3 vertices + 3
edges locally. Halo pack/unpack is **GPU-resident** (device buffers handed to
GPU-aware MPI) using persistent, registration-bounded buffer pools ported from
Canopy. The halo is described by a **`HaloExchangePlan`** (per-peer index maps +
buffer pools); any operation that changes the local entity count or ghost set
invalidates the plan, which is rebuilt before the next sync.

### Adaptive refinement

Refinement is **split-based**: a face is refined "red" 1→4 by inserting a midpoint
vertex on each of its 3 edges and connecting them into 4 child triangles. Policy is
**conforming, 2:1-balanced**: hanging nodes (T-junctions) are permitted but bounded
to at most one refinement-level difference across any edge. A parallel
mark-and-propagate fixpoint enforces the 2:1 bound across partition boundaries; the
refine decision is a pure function of gids + levels (synced each iteration), so all
ranks agree, and the structured midpoint keys guarantee a shared edge refines
identically on both sides. (Edge collapse/flip is out of scope.)

**Midpoint placement is a pluggable interpolation policy.** The default sets a new
midpoint vertex's `position` to the linear edge midpoint `0.5·(p(v0)+p(v1))` and
every user field to the linear average of the two endpoints. A **per-field override
hook** lets a curvature-aware geometric scheme (e.g. modified Butterfly, which the
1-deep halo's 1-ring makes available) or a physics-correct field rule (e.g. a
vorticity/sheet-strength conservation rule from the reference solver) be substituted
without touching the topology/refinement machinery. Initial icosphere *generation*
separately projects new vertices onto the sphere — that is a generation step, not
AMR.

The distributed driver is `Tessera::refine(mesh, halo, ownedFaceMask, policy)`. It
enforces the 2:1 balance and assigns midpoint gids entirely through **edge
coordinators** (the deterministic rank `hash(EdgeKey) % nranks` gathers an edge's
two incident faces from whichever ranks own them) rather than the vertex-based
halo, which does not guarantee a face sees its cross-boundary edge-neighbour. Three
coordinator phases run: (1) a monotone mark-propagation fixpoint (`MPI_Allreduce`
on the changed-count, hard iteration cap) that flags the coarser face of any edge
whose incident final levels would differ by >1; (2) midpoint-gid assignment — each
split edge's midpoint is owned by the lowest incident refining-face owner, gids are
allocated in a global block via `MPI_Exscan` over the pre-refinement global vertex
count, and the owner **sends** the gid to co-sharers, so a shared edge's midpoint is
bit-identical on every side with no reliance on matching local order; (3) edge
ownership (lowest incident child-face owner) so owned counts stay a global
partition. The refined mesh is left holding each rank's owned entities; the 1-deep
halo is **rebuilt in Step 7** (shared with migration), so `haloExchange()` must not
run on a freshly-refined mesh until then.

The single-rank building block is the free function
`Tessera::refineLocal(mesh, faceMask, policy = DefaultRefinePolicy)`: it red-splits
every flagged face, deduplicates each edge's midpoint by the edge's `EdgeKey`, emits
the four children under the convention `{a,ab,ca}, {b,bc,ab}, {c,ca,bc}, {ab,bc,ca}`
(matching the icosphere subdivision, so a uniform mask reproduces one subdivision
level and preserves Euler), propagates `level = parent + 1` to child faces/edges, and
re-derives edges/CSR/key tables. A partial mask leaves bounded hanging nodes for the
parallel 2:1 balance to resolve. The interpolation policy exposes two hooks —
`interpolatePosition(mid, a, b, dim)` and `template<std::size_t M>
interpolateVertexField(a, b)` (absolute member index `M`, called per component) —
overridden per field with `if constexpr` on `M`.

### Load balancing — optional, external-first

Load balancing is **optional**; building, haloing, and refining never require it.
The public contract is **migration**, not partitioning:

- `Tessera::migrate(mesh, halo, dest_rank_per_owned_face)` applies an externally-
  computed assignment through the hand-built comm layer: faces move to their
  destination ranks, their vertices/edges follow, the whole field pack rides along,
  ownership is recomputed (lowest-rank), and a fresh 1-deep ghost layer + halo plans
  are rebuilt. Read accessors (`ownedFaceCentroids(mesh)`, `ownedFaceGids(mesh)`,
  `ownedFaceWeights(mesh)`) let an external partitioner compute the assignment. This
  is the path **Canopy** uses to drive redistribution from its FMM-tree partition,
  avoiding an intermediate Tessera↔Canopy migration. Unlike `distribute()`,
  `migrate()` assumes **no replicated knowledge** — every rank holds only its own
  entities, so ownership and the ghost set are discovered by communication (whole
  Cabana tuples travel over `allToAllV`; ownership and ghost-face discovery route
  through per-gid vertex/edge coordinators). This is the **general (non-replicated)
  ghost builder** that distributed `refine()` (Step 6b) deferred.
- `Tessera::loadBalance(mesh, halo, imbalanceTolerance=0.05)` is a thin convenience
  wrapper (**Step 7b**): `Tessera::computeLoadBalance(mesh, imbalanceTolerance)`
  gathers every rank's owned-face centroids/weights/gids to rank 0 (the mesh is
  **not** replicated, so the geometric input must be assembled before Zoltan2 can
  see it), runs Zoltan2 geometric **MultiJagged** over a `Teuchos::SerialComm`
  (solve on rank 0 only — MultiJagged is not guaranteed deterministic across ranks
  — then `MPI_Scatterv` the per-face part assignment back; RCB is never used, it
  breaks on Tuolumne), and returns a `dest` in the same order as
  `ownedFaceCentroids/Gids/Weights`. `loadBalance()` hands that `dest` to the
  **same** `migrate()`. There is no separate internal-vs-external migration code
  path. The Zoltan2 adapter (`Zoltan2::BasicVectorAdapter`) is built with its
  generic multivector constructor (per-dimension arrays), not a fixed 3D x/y/z
  one, so it works for both `Dim=2` and `Dim=3`.

### Parallel I/O

Mesh connectivity and any per-vertex/edge/face fields are written with **manual
parallel HDF5** (MPI-IO collective) plus an **XDMF** sidecar that Paraview reads
directly. Each rank writes its owned entities into a contiguous hyperslab; dense
global indices are assigned by an `MPI_Exscan` over owned-only counts (ghosts are not
written). A **full round-trip reader** reconstructs a mesh from the file; tests do
write→read→compare of topology and field checksums, and assert the on-disk global
checksum is independent of rank count.

---

## Usage / API

> *(Planned — the API is being implemented against this specification; signatures may
> shift as steps land. See `tasks/milestone1_mesh.md` for status.)*

```cpp
#include <Tessera_Mesh.hpp>

using MeshT = Tessera::Mesh<double, /*Dim=*/3, VFields, EFields, FFields,
                            MemSpace, ExecSpace>;

MeshT mesh( MPI_COMM_WORLD );
mesh.buildIcosphere( /*subdivisions=*/3 );   // initial coarse closed surface
mesh.haloExchange();                          // fill the 1-deep ghost layer

auto vort = mesh.slice<Tessera::Field::Vorticity>();   // typed Cabana slice
// ... fill/evolve owned vertices ...
mesh.haloExchange();                          // refresh ghosts (whole field pack)

Tessera::refine( mesh, halo, face_refine_mask ); // conforming 2:1 split refinement

Tessera::loadBalance( mesh, halo );           // internal Zoltan2 (optional, Step 7b)
// or, external (e.g. Canopy-driven):
//   auto c = Tessera::ownedFaceCentroids( mesh );
//   std::vector<Tessera::Rank> dest = external_partition( c );
//   Tessera::migrate( mesh, halo, dest );

Tessera::writeHDF5( mesh, "bubble_0000" );    // bubble_0000.h5 + bubble_0000.xmf
```

### Example programs

| Example | Directory | Arguments | Description |
|---|---|---|---|
| `hello_tessera` | `examples/01_hello_tessera/` | *(none)* | Prints the active Kokkos backend and MPI rank count. Build verification only. |

*(End-to-end mesh example added in Step 9.)*

---

## Dependencies and Build Notes

### Dependencies

| Dependency | Notes |
|---|---|
| [Kokkos](https://github.com/kokkos/kokkos) | ≥ 4.0; Serial, OpenMP, and HIP execution spaces |
| [Cabana](https://github.com/ECP-copa/Cabana) | MPI-enabled build required |
| [Trilinos](https://github.com/trilinos/Trilinos) | Zoltan2 (geometric MultiJagged) for internal load balancing |
| [HDF5](https://www.hdfgroup.org/) | Parallel (`+mpi`) build, for mesh I/O *(added to the spack env in Step 0)* |
| MPI | Cray-MPICH on Tuolumne (GPU-aware); OpenMPI on local workstations |
| CMake | ≥ 3.21 |
| clang-format | For `make format` / `make format-check` |

### Build (Tuolumne — AMD MI300A)

```bash
spack env activate ~/spack_envs/tuolumne_trilinos/
mkdir build-tuolumne && cd build-tuolumne
bash ../run_cmake_toulumne.sh
make -j $(nproc)
```

### Build (local workstation)

```bash
mkdir build-local && cd build-local
bash ../run_cmake.sh -DCMAKE_PREFIX_PATH=<deps-install>
make -j $(nproc)
```

### Design limitations

- Milestone 1 implements **split-based refinement only** (no edge collapse/flip).
- Topology surgery (pinch-off / merger of surfaces) is not implemented; the data
  model is component-agnostic to leave room for it.

---

## Future Optimizations

- Incrementally patch the `HaloExchangePlan` index maps after refinement instead of
  a full `rebuild()` each 2:1-balance iteration.

---

## Known Issues

- **A distributed mesh must be re-haloed after `refine()` before `haloExchange()`.**
  Distributed `refine()` (Step 6b) leaves each rank holding only its refined *owned*
  entities and clears the halo. The general (non-replicated) halo rebuild now exists
  (Step 7, inside `migrate()`), but it is currently coupled to `migrate()` and is not
  yet invoked automatically at the end of `refine()`; calling `haloExchange()` on a
  freshly-refined-but-not-migrated mesh is a no-op on an empty plan, not a synced
  ghost layer. Factoring the rebuild into a standalone `rebuildHalo()` that `refine()`
  also calls is a tracked follow-up.
- **Adaptive `refine()` is non-conforming (bounded hanging nodes).** A partial
  refine mask leaves T-junctions bounded to a 2:1 level jump; the owned-only Euler
  number equals 2 only for a uniform (conforming) refine. Green-closure to a fully
  conforming triangulation is a future enhancement.
- **Edge user fields are reset by `refine()`/`refineLocal()`.** Edges are re-derived
  from the new face connectivity, so any per-edge user data is re-initialized (M1
  carries no edge user state through AMR). Vertex and face user fields are preserved
  (interpolated / inherited).
