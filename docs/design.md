# Tessera — Design

Detailed descriptions of the algorithms and design decisions behind Tessera:
entity storage and layout, connectivity, the 1-deep MPI halo, adaptive
refinement, load balancing, and parallel I/O — including *why* each choice was
made.

For the public API, build instructions, and example arguments, see
[README.md](../README.md).

---

## Concepts

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

## Templated precision and embedding dimension

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

## Data model — AoSoA layout

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

## Per-entity data — compile-time field pack

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

## Global IDs — structured 128-bit keys

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

## Ownership

A face is owned uniquely by the rank the partition assigns it to. A **shared vertex
or edge is owned by the lowest rank** in its *sharing set* — the set of ranks that
hold it (owned or ghost) after the halo is built. This rule is deterministic and
needs no vote/tie-break. The sharing set is materialized by the ghost-build neighbor
exchange (neighbor-bounded, **not** all-to-all).

## 1-deep halo

After `haloExchange()`, every owned vertex has all incident edges/faces and the
opposite vertices of its 1-ring locally; every owned face has its 3 vertices + 3
edges locally. Halo pack/unpack is **GPU-resident** (device buffers handed to
GPU-aware MPI) using persistent, registration-bounded buffer pools ported from
Canopy. The halo is described by a **`HaloExchangePlan`** (per-peer index maps +
buffer pools); any operation that changes the local entity count or ghost set
invalidates the plan, which is rebuilt before the next sync.

## Slice/handle validity

Every count-changing operation (`distribute()`, `migrate()`, `refine()`, the
serial builder) reallocates and reassigns the mesh's AoSoAs, CSR adjacency, and
edge/face key-Views. `Mesh` tracks this with a monotonic **`generation()`**
counter, bumped by `resizeVertices/Edges/Faces`, `setOwnedCounts`, and the
key-View/CSR replacement methods (`setEdgeKeys`/`setFaceKeys`,
`rebuildVertexFaces`/`rebuildVertexEdges`). `vertexSlice<M>()`/`edgeSlice<M>()`/
`faceSlice<M>()` (and the `...Handle()` CSR/key-View accessors) return a
`GenerationHandle` stamped with the generation at creation time: copying a
stale handle (e.g. capturing it into a `KOKKOS_LAMBDA` after a topology-
changing call) aborts with a diagnostic instead of silently reading dangling
storage. Element access (`operator()`) forwards unchanged, so there is no
per-element device-side cost; the check itself compiles out when
`Tessera_ENABLE_DEBUG_CHECKS` is off. `haloExchange()` never bumps `generation()`
(it is topology-preserving), so handles survive it. `vertexSlices<M...>()`/
`edgeSlices<M...>()`/`faceSlices<M...>()` re-derive a whole tuple of slices in
one call, for the "re-slice at the top of every solver stage" discipline.

One caveat: `Tessera_Migrate.hpp`'s raw `migrate(comm, aosoa, dest, bufs)`
primitive is mesh-agnostic (it operates on a bare AoSoA, not a `Mesh&`) and
therefore cannot bump `generation()` itself. The mesh-level
`Tessera::migrate(mesh, halo, dest)` wrapper (see Load balancing, below) already
bumps it via its own `resize()`/`setOwnedCounts()` calls. If you ever call the
raw primitive directly against `mesh.vertices()`/`edges()`/`faces()` instead of
going through that wrapper, call `mesh.bumpGeneration()` yourself afterward.

## Adaptive refinement

### Refinement modes

Which conformity contract refinement obeys is a **compile-time `Mesh` template
parameter**, `RefinementMode Mode` (`src/Tessera_RefinementMode.hpp`), appended
*last* in the parameter list so every existing seven-argument `Mesh<...>` spelling
is unchanged:

```cpp
enum class RefinementMode { HangingNode2to1, Conforming };

using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                   MemSpace, ExecSpace, RefinementMode::Conforming>;
//                                      ^ optional; default HangingNode2to1
```

| Mode | Contract |
|---|---|
| `HangingNode2to1` *(current default)* | 2:1-bounded hanging nodes — the behavior described in the rest of this section. A partial mask leaves T-junctions; owned-only Euler `V−E+F = 2` holds only for a uniform refine. |
| `Conforming` | The same 2:1-balanced **red layer**, plus a *transient* red–green–blue **closure** pass that retriangulates every kept face carrying hanging nodes, so the visible mesh has no T-junctions for an arbitrary adaptive mask. The closure creates no new vertices (it reconnects midpoints the neighbouring red splits already made) and is recomputed from scratch each `refine()` call, which bounds the triangle similarity classes. |

Compile time rather than a runtime flag, because the closure bookkeeping face
members then exist **only** in `Conforming` mode — a hanging-node mesh pays zero
extra memory. Those two members, `ClosureParent` (`GlobalId`, the red parent's
gid; `invalid_gid` on a red face) and `ClosureParentVerts` (`GlobalId[3]`, the
parent's corners in winding order), are appended **after** the user field pack, so
`FaceField::UserBegin` stays `5` and `userFaceField<M>()` is bit-identical in both
modes; their slice indices come from `closureParentField<FaceUserFields>()` /
`closureParentVertsField<...>()`, or the `MeshT::closure_parent_field` /
`closure_parent_verts_field` constants. Anything iterating face *user* fields must
size the loop with `numFaceUserFields<FaceUserFields>()` rather than
`face_member_types::size - FaceField::UserBegin`, which over-counts by two in
`Conforming` mode.

`refine()` and `refineLocal()` dispatch on `MeshT::refinement_mode` with
`if constexpr`. **The `Conforming` branch is currently a stub that aborts**; the
closure kernel, its distributed wiring, migration, and I/O are staged in
[tasks/conforming-refinement.md](../tasks/conforming-refinement.md), which holds
the full design. In `Conforming` mode a face's `Level` will remain the **red**
level — a closure child carries its parent's level — so `Level` no longer maps 1:1
to triangle size.

### The red layer (both modes)

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

## Quality-based refinement marking

`Tessera::markByQuality(mesh, criterion)` inspects mesh geometry and returns the
owned-face `std::vector<char>` mask `refine()` already consumes, so AMR can be
driven from mesh quality instead of a hand-authored mask:
`markByQuality(mesh, crit) -> refine(mesh, halo, mask, policy)`. A criterion is
duck-typed like `RefinePolicy` — any struct exposing
`template<class MeshT> std::vector<char> mark(const MeshT& mesh) const` — and owns
its own evaluation, including any communication it needs; both metrics recompute
on demand from the existing `Position`/`Verts` fields (no new stored fields, no
halo/migrate/refine propagation).

`EdgeLengthCriterion<Scalar>{maxLen}` marks an owned face if **any** of its 3 edges
exceeds the absolute target length `maxLen` (the vortex-sheet/interface-tracking
convention: insert points once a segment exceeds ε). It is pure per-owned-face
geometry with **no MPI** — every owned face's 3 vertices are local (owned or ghost,
the 1-ring closure invariant) and a shared vertex's `Position` is bit-identical
across ranks, so the marked set is rank-count independent for free. Evaluated with
a device Kokkos kernel: a host-built face→vertex-local-index view feeds a
`parallel_for` that reads the device `Position` slice directly (the only
host→device transfer is the small `3·nOwnedF` index array). A scalar convenience
overload `markByQuality(mesh, maxEdgeLength)` builds an `EdgeLengthCriterion`
directly; a `class = std::enable_if_t<!std::is_arithmetic<Criterion>::value>` SFINAE
guard on the generic `markByQuality(mesh, crit)` overload keeps a bare scalar
threshold from being an ambiguous call against both overloads.

`CurvatureCriterion<Scalar>{maxAngle}` (radians) marks **both** faces incident to
any edge whose dihedral bend exceeds `maxAngle` (an absolute bend threshold). The
dihedral needs both incident faces' unit normals, and the neighbour across a
partition boundary is **not** guaranteed present in the vertex-based 1-ring halo
(at a 3-way corner the edge's two vertices can both be ghosts owned by a lower
rank, so the neighbour is incident to no owned vertex). So unlike the edge-length
criterion it **communicates**, routing the gather through **edge coordinators**
(`edgeCoordRank(EdgeKey) = hash % nranks` + `allToAllV`) — the same idiom the
distributed refinement uses, **not** the halo. Each rank device-computes its owned
faces' outward unit normals (`n = normalize((p1−p0) × (p2−p0))`, relying on the
consistent CCW-seen-from-outside winding with the `e[k]=(v[k],v[k+1])` convention,
so `n0·n1 = cos(dihedral bend)`), advertises `(EdgeKey, normal, faceGid, owner)` to
each edge's coordinator; the coordinator receives exactly the two incident faces of
every edge on the closed surface, flags the edge sharp iff `n0·n1 < cos(maxAngle)`,
and routes a mark-request back to both incident face owners. Because the verdict is
computed at a single deterministic coordinator per edge from both true incident
normals — never from partition-local halo state — the marked set is rank-count
**and** boundary-straddle independent. `Dim==2` (a planar surface has no dihedral)
returns an all-zero mask. One coordinator round-trip (advertise → mark-request),
plus the local device normal kernel; no new stored state and no change to
`refine()`.

A face flagged by more than one criterion is refined once; combining criteria (e.g.
edge-length OR curvature) is a caller-side element-wise OR of their masks — no
combinator is shipped.

**Gotcha for anyone computing mesh geometry right after `refine()`:** `refine()`
clears the halo and leaves only owned-first entities (the 1-deep halo is rebuilt in
Step 7's `migrate()`), so an owned edge's endpoint can be a vertex this rank neither
owns nor holds any copy of — in particular a new midpoint's owner and an edge
incident to it can differ (midpoint owner = min incident *refining-face* owner;
edge owner = min incident *child-face* owner), and `refine()` only ever ships a
midpoint's **gid** to its co-sharers, never its position. A purely local (or
before/after-snapshotted) vertex map is therefore not sufficient in general; code
that needs positions immediately post-refine must gather any missing ones from
their true owner via a gid coordinator (`gid % size`), the same idiom
`MeshInvariants.hpp`'s `check21Balance`/`checkMidpointAgreement` use for cross-rank
edge decisions, applied to position data instead of level/gid (see
`tests/test_markquality_edge.cpp`'s `maxOwnedEdgeLength()` for a worked example).

## Load balancing — optional, external-first

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

## Parallel I/O

Mesh connectivity and any per-vertex/edge/face fields are written with **manual
parallel HDF5** (MPI-IO collective) plus an **XDMF** sidecar that Paraview reads
directly. Each rank writes its owned entities into a contiguous hyperslab; dense
global indices are assigned by an `MPI_Exscan` over owned-only counts (ghosts are not
written). A **full round-trip reader** reconstructs a mesh from the file; tests do
write→read→compare of topology and field checksums, and assert the on-disk global
checksum is independent of rank count.

**API** (`Tessera_HDF5Writer.hpp` / `Tessera_HDF5Reader.hpp`, both header-only,
templated on the mesh type):

```cpp
// Collective on mesh.comm(). Writes <stem>.h5 (parallel HDF5) + <stem>.xmf
// (XDMF sidecar, rank 0 only). Ghost dense indices are fetched from each
// ghost's Owner field -- no halo plan needed.
template <class MeshT>
void writeMesh( const MeshT& mesh, const std::string& stem );

// Collective. Fills an empty mesh (constructed on the same comm) + its halo
// by block-reading <stem>.h5 and re-running migrate() (a self-destination
// move) to establish ownership + the 1-deep halo.
template <class MeshT>
void readMesh( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
              const std::string& stem );
```

**On-disk layout** (single file `<stem>.h5`, groups `/vertices`, `/edges`,
`/faces`, each row `i` = one entity's owned-block hyperslab):

| Group | Datasets | Notes |
|---|---|---|
| `/vertices` | `gid` (uint64), `position` (Scalar×Dim), `u0..uN` (per user field) | |
| `/edges` | `gid`, `verts` (uint64×2, **dense** vertex indices), `level` (int16), `u0..uN` | |
| `/faces` | `gid`, `verts` (uint64×3, dense), `edges` (uint64×3, dense), `level`, `u0..uN` | XDMF triangle connectivity references `/faces/verts` |

Every entity carries both its persistent 64-bit `gid` (the cross-run/checksum
identity, stored verbatim) and is referenced elsewhere by a **dense** index in
`[0,N)` per kind (assigned via `MPI_Exscan` over owned counts) -- XDMF/Paraview
connectivity needs 0-based contiguous indices, so `verts`/`edges` datasets
store dense references while `gid` carries the persistent one. Partition-
dependent state (owner rank, ghost layer, CSR, key tables) is **not** written;
the reader reconstructs it by handing a covering of the file to the tested
`migrate()`. Root attributes (`format_version`, `dim`, `scalar_bytes`,
`Nv`/`Ne`/`Nf`, per-user-field extents) let the reader hard-fail on a
template/schema mismatch instead of silently misreading. Building I/O requires
a **parallel** (`+mpi`) HDF5 -- see **Dependencies and Build Notes** in [README.md](../README.md).
