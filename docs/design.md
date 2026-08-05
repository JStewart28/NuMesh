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
                   MemSpace, ExecSpace, RefinementMode::HangingNode2to1>;
//                                      ^ optional; default Conforming
```

| Mode | Contract |
|---|---|
| `Conforming` *(default)* | The same 2:1-balanced **red layer**, plus a *transient* red–green–blue **closure** pass that retriangulates every kept face carrying hanging nodes, so the visible mesh has no T-junctions for an arbitrary adaptive mask. The closure creates no new vertices (it reconnects midpoints the neighbouring red splits already made) and is recomputed from scratch each `refine()` call, which bounds the triangle similarity classes. |
| `HangingNode2to1` | 2:1-bounded hanging nodes — the behavior described in the rest of this section. A partial mask leaves T-junctions; owned-only Euler `V−E+F = 2` holds only for a uniform refine. Cheaper: no closure faces, no closure bookkeeping, no un-close pass. |

**Why `Conforming` is the default.** A hanging node makes every surface operator
assembled over its neighbourhood silently wrong: the vertex's incident-face set,
its edge 1-ring, and its `vertexFaces()` row are all self-consistent, but they
describe a half-disc rather than a disc, so `applyStencil` and
`reduceVertexFromFaces` never see the kept face the node geometrically touches
(and `CurvatureCriterion`'s edge coordinator skips exactly those edges). Silence
is what makes it the wrong default: a consumer gets a plausible answer rather
than an error. `HangingNode2to1` remains fully supported and is the right choice
for cell-centred work that never assembles a vertex neighbourhood — it is opt-in
rather than absent.

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

`Conforming` is implemented end to end: `refineLocal()` and the distributed
`refine()` (`src/Tessera_RefineClosure.hpp` holds the closure kernel; the modes
share one body in each driver and branch with `if constexpr`), `migrate()` /
`loadBalance()` (see *Redistributing a conforming mesh*), HDF5 write/read (see
*Parallel I/O*), and `markByQuality` — whose two criteria are mode-agnostic,
since a criterion produces a mask over *visible* faces and `refine()` translates
it. In `Conforming` mode a face's `Level` remains the **red** level — a closure
child carries its parent's level — so `Level` no longer maps 1:1 to triangle
size. **None of the conforming path has been executed yet**: it is written,
registered, and compiles clean on both backends, and it is verified in a single
pass at the end of the plan. See
[tasks/conforming-refinement.md](../tasks/conforming-refinement.md), which holds
the full design and the remaining tasks.

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
partition, **and the refined edge gids themselves**.

Edge gids are assigned by that third coordinator phase rather than from a local
scan. A coordinator sees each `EdgeKey` exactly once, so it numbers its own keys
densely from an `MPI_Exscan` over its key count and replies with that gid to every
rank that advertised the key. Refined edge gids are therefore dense in
`[0, globalEdges)` and **identical on both sides of a partition boundary** — the
same property `distribute()`, the mesh builder, `migrate()`'s gid-keyed edge maps
and the HDF5 writer/reader all already assume of an edge gid, and the one vertex
and face gids already had. (Assigning them instead from a local exscan over each
rank's *local* edge count, as the code did before Task 8 D3, left gaps in the owned
gid space wherever a non-last rank held a boundary duplicate, and gave the two
sides of a boundary edge different gids for the same edge.)

Phase 2 advertises the edges of **every** owned face — refining *and* kept — each
tagged with a `refining` flag. An edge is split iff some incident face refines;
ownership is still derived from the refining participants alone, so nothing about
which midpoints exist or what gid each gets depends on the kept advertisements.
What they buy is completeness of the reply: the coordinator answers *every*
participant, and the midpoint owner ships the gid to *every* co-sharer, so a rank
holding only the **kept** side of an edge a neighbour bisected still learns that
edge's midpoint gid. `RefineResult::midpoints` is therefore a full **split-edge
map** — every edge of any owned face that was bisected, with its globally agreed
midpoint gid — which is both what `checkMidpointAgreement` verifies and the input
the conforming closure needs to retriangulate a kept face. The extra traffic is
bounded by three messages per *kept* owned face on the first Phase-2 round only
(plus, in `Conforming` mode, one more per already-bisected edge, which is
advertised as its two halves — see the closure section); no new communication
rounds were added.

Phases 1-3 leave each rank holding only its owned entities — every cross-rank
decision went through a coordinator, so no ghost was needed — and `refine()` then
finishes by calling **`rebuildHalo()`** (`Tessera_HaloRebuild.hpp`), the general
non-replicated 1-deep halo rebuild it shares with `migrate()`. So on return the
passed halo's three plans are valid: `haloExchange()` is meaningful (and a re-sync,
since round C/D fetch ghost values from their owners), and `refine()` may be called
again immediately with nothing in between. `rebuildHalo()` also canonicalises the
local layout — owned first then ghost, each kind ascending by gid — so local index
order is a pure function of the owned gid set, which a caller can observe through a
gid-derived face mask.

### The closure layer (`Conforming` mode only)

`Conforming` refinement is not a second refinement engine: it is the *same* red
engine with three purely local, communication-free steps wrapped around it, all
inside `detail::refineImpl()` under `if constexpr`.

1. **Un-close.** The mesh's *visible* faces are collapsed back to the persistent
   red layer, which is the input phases 1–3 have always expected. Each closure
   child stores its retired red parent's gid and corner gids outright
   (`ClosureParent` / `ClosureParentVerts`), so this is per-face with no sibling
   lookup and never discards a vertex.
2. **Mask translation.** The caller's mask — and `markByQuality`'s output — is
   indexed by *visible* owned faces; a red parent is marked iff **any** of its
   closure children was.
2b. **Split-edge recovery.** Being bisected is a property of the red layer that
   outlives the round that caused it, whereas Phase 2 only ever learns *this*
   round's bisections (an edge with no refining incidence is dropped at the
   coordinator). The persistent map is recovered from the same closure bookkeeping,
   locally and with no extra field: for a closed parent, an edge is split iff no
   child carries it, and its midpoint is the unique child corner outside the
   parent's corners whose two half-edges each belong to exactly one child (a
   half-edge shared by two children is a fan *diagonal*, which is what makes the
   "exactly one" qualifier necessary rather than cosmetic). Two uses beyond the
   closure itself: a coarse face refining across a hanging node **reuses** the
   existing midpoint instead of minting a coincident second vertex, and phases 1
   and 2 key the coordinator on the two **half-edges** of a bisected edge so the
   coarse face meets its true neighbours. That last part is what bounds the level
   jump across a hanging node at all — both coordinator rules require an edge to
   have two incident faces, and a hanging node otherwise leaves one on each side.
   `HangingNode2to1` has no such record and so keeps neither property (README →
   *Known Issues*).
3. **Close.** After the red 1→4 split, every **kept** red face is retriangulated
   according to how many of its edges (|S| ∈ {0,1,2,3}) the split-edge map says
   were bisected: pass-through, green (2 children), blue (3), or red-closure (4).
   Children inherit the parent's `Level` and face user fields, and the |S| = 3
   pattern deliberately does *not* promote the face into the red layer. A red child
   of a face refined in this round has |S| = 0 unless one of its two inherited
   boundary half-edges is bisected this round, which is possible exactly when its
   parent's edge carried a **reused** midpoint; `closeFaces()` asserts that
   narrower form. The blue quad's diagonal is tie-broken on the **lower midpoint
   gid**, which is globally agreed within a run, so the closure does not depend on
   *which rank owns a face* — it is partition-independent at a fixed rank count.
   It is **not** independent of the *rank count*: midpoint gids come from an
   `MPI_Exscan`, so the same red mesh can close with a different blue diagonal
   under a different partition size (measured: np1–4 agree, np5 flips 4 of 20 blue
   parents). Everything else about the closure *is* rank-count invariant — the red
   layer, the `|S|` histogram, the closure-vertex set and V/E/F — and
   `conforming_determinism` asserts that the visible layer differs *only* through
   blue diagonals. A geometric rule (equivalently: connect the midpoint of the
   **longer split edge**) would remove the dependence but cannot be evaluated
   locally, because the closure runs on the un-closed red layer whose corners may
   name vertices the rank does not hold; it would need the split edge's length
   carried in Phase 2's coordinator reply. See Decision 11 in
   `tasks/conforming-refinement.md`.

The closure creates **no vertices**, so no `MPI_Exscan`, no interpolation, and no
`RefinePolicy` involvement. The one widened count is the *face*-gid allocation:
the single existing exscan covers `4·nRefining + countClosureChildren(...)` per
rank, allocated above the global max **visible** face gid so a retired parent gid
(reused by the next round's un-close) can never collide.

**Shape quality is bounded in the round count, and this is measured.** Because the
closure is discarded and rebuilt each `refine()`, every visible triangle is a red
triangle or one of three fixed retriangulations of one, so the triangle similarity
classes are finitely many and the worst shape cannot drift with depth. Over 16
adaptive rounds on a shrinking geodesic cap (`tests/test_conforming_quality.cpp`,
in the gate at ranks 1–5 on both backends), the red layer — the closure's *input* —
holds at radius ratio `Q = 1.0278` and min angle 54.397° in **every** round; the
green family holds at `Q = 1.5672` from round 1; the closure fraction peaks at
0.1864 in round 6 and then *declines* to 0.0623, as an O(perimeter) set inside an
O(area) mesh must. Only the blue family moves, and it moves in discrete steps —
1.7759 (round 6) → 2.2344 (8) → 2.5254 (11) — as the growing cap reaches new
(red class × split pair × diagonal) combinations, then **saturates**: rounds 11–16
are identical while the mesh grows from 4348 to 24608 faces. The worst measured
shape over 16 rounds is min angle 25.987°, `Q = 2.5254`, and amplification
`Q(child)/Q(parent) = 2.4906`. A *persistent* closure would instead bisect
already-bisected green triangles round after round, with no lower bound on the
angle; that difference is the whole reason the closure is transient.

Consequences worth knowing:

- Live face gids are **sparse**, and a closure child may name a vertex gid the
  rank does not hold at the point the closure runs — its midpoint corner is owned
  by the refining neighbour. That is why the halo rebuild's round G (recover
  referenced-but-non-held tuples) is not optional; by the time `refine()` returns,
  every vertex an owned face references is held locally with its position.
- The 2:1 balance is a property of the **red** layer; check it after un-closing.
- Face **user** fields on closure faces are the parent's, copied. If a solver
  writes per-closure-face state, un-close keeps the lowest-gid child's values and
  discards the rest — the same contract as "edge user fields are reset by
  `refine()`".
- Any site that materializes a face AoSoA from scratch rather than copying whole
  tuples must call `initClosureFaceMembers<MeshT>()`: a Cabana `AoSoA` is
  zero-initialized, and a zero `ClosureParent` reads as the perfectly valid face
  gid 0. `buildTriangleMesh()`, `distribute()`, and the HDF5 reader do;
  `migrate()` copies tuples and needs nothing.
- Closure siblings must stay **co-resident** — two ranks each holding a child of
  one parent would each un-close it into a duplicate red face. `migrate()`
  repairs a `dest` that splits a group (round S, local); a caller that hands the
  mesh a partition derived from outside the mesh's own layout — the HDF5 reader
  is the one such caller — must first run the collective
  `repairClosureCohesion(mesh, dest)`, since round S can only see groups that
  are already co-resident.

### Quality-based marking on a conforming mesh

`markByQuality` needs no mode awareness: a criterion returns a mask over
**visible** owned faces and step 2 above translates it to the red layer. One
behavior does improve. `CurvatureCriterion`'s edge coordinator computes a
dihedral only for edges with exactly two incident faces and silently skips the
rest; on a hanging-node mesh those skipped edges are exactly the T-junctions, so
a fold running through a refinement front is under-marked. On a conforming mesh
there are none, so the coordinator's assumption is true rather than merely
usually true.

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

**Note for anyone computing mesh geometry right after `refine()`:** this used to be
a gotcha and is no longer one. `refine()` finishes with `rebuildHalo()`, whose round
G guarantees that every vertex an owned face references is held locally **with its
position** — so a purely local vertex map is sufficient. Before that, `refine()`
cleared the halo and left only owned entities, and an owned edge's endpoint could be
a vertex the rank neither owned nor held any copy of, because a new midpoint's owner
and an edge incident to it can differ (midpoint owner = min incident *refining-face*
owner; edge owner = min incident *child-face* owner) and `refine()` only ever ships a
midpoint's **gid** to its co-sharers, never its position. Code that must gather a
position it does not hold — `ownedFaceCentroids()` on a mesh whose owned set was
changed by something else — does so from the true owner via a gid coordinator
(`gid % size`), the same idiom
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
  through per-gid vertex/edge coordinators).
- `Tessera::rebuildHalo(mesh, halo)` (`Tessera_HaloRebuild.hpp`) is the **general
  (non-replicated) ghost builder** on its own, with no move: rounds G (recover
  referenced-but-non-held vertex/edge tuples), B (ownership + ghost discovery via
  coordinators), C (ghost fetch from face owners) and D (owned-first assembly, CSRs,
  key tables, the three plans). `migrate()` is rounds S and A plus this; `refine()`
  calls it directly. The entire interface between the move half and the halo half is
  the three gid-keyed maps `faceById` / `vById` / `eById`. Callers normally never
  need it — it is public for the case where a mesh's owned set was changed by
  something other than `refine()`/`migrate()`.
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

### Redistributing a conforming mesh

In `RefinementMode::Conforming` the faces a partitioner sees are the **visible**
(closed) faces, and two things follow.

- **Closure siblings must stay co-resident.** A closure child names its retired
  red parent outright, so un-closing is local *per child* — but if two ranks each
  hold a child of the same parent, each restores that parent and the red layer
  gains a duplicated face. `migrate()` therefore runs a local **sibling-cohesion
  fixup** (round S) before the move: every child follows the **lowest-gid**
  sibling's destination. A violating `dest` is **repaired, not rejected** —
  `computeLoadBalance()` partitions by face *centroid* and siblings have
  different centroids, so Zoltan2 scatters them routinely; rejecting would make
  `loadBalance()` unusable on a conforming mesh, and pushing the repair onto
  callers would duplicate it at each one. The tie-break reads only globally
  agreed gids, so the repair does not reintroduce partition dependence. The
  number of `dest` entries overridden is returned in `MigrateStats`
  (`migrate()` and `loadBalance()` now return one; existing call sites that
  ignore the value are unaffected).
- **A red parent is one unit of work.** `ownedFaceWeights()` gives a closure
  child weight `1/nsiblings` and a passed-through red face weight `1.0`, so the
  total weight is the **red**-face count and a sibling group weighs 1.0 however
  it is split. Without this the closure — an O(level-jump-boundary) set that is
  rebuilt on every refine — would masquerade as real load and pull parts toward
  refinement fronts. The per-owned-face-in-local-index-order contract is
  unchanged; only the values differ, and only in `Conforming` mode.

The halo rebuild itself needs nothing new: on a conforming mesh every edge has
exactly two incident faces, so the 1-ring closure round D performs is cleaner,
not harder.

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
| `/faces` | `gid`, `verts` (uint64×3, dense), `edges` (uint64×3, dense), `level`, `u0..uN`, and in `Conforming` mode `closure_parent` (uint64), `closure_parent_verts` (uint64×3) | XDMF triangle connectivity references `/faces/verts` |

Every entity carries both its persistent 64-bit `gid` (the cross-run/checksum
identity, stored verbatim) and is referenced elsewhere by a **dense** index in
`[0,N)` per kind (assigned via `MPI_Exscan` over owned counts) -- XDMF/Paraview
connectivity needs 0-based contiguous indices, so `verts`/`edges` datasets
store dense references while `gid` carries the persistent one. Partition-
dependent state (owner rank, ghost layer, CSR, key tables) is **not** written;
the reader reconstructs it by handing a covering of the file to the tested
`migrate()`. Root attributes (`format_version`, `refinement_mode`, `dim`,
`scalar_bytes`, `Nv`/`Ne`/`Nf`, per-user-field extents) let the reader hard-fail
on a template/schema mismatch instead of silently misreading. Building I/O requires
a **parallel** (`+mpi`) HDF5 -- see **Dependencies and Build Notes** in [README.md](../README.md).

### Conforming meshes on disk (format version 2)

The visible faces of a `Conforming` mesh are the *transient* closure layer; the
persistent thing is the red layer un-close derives from the two closure
bookkeeping members. A file that dropped them would read back looking perfectly
healthy — same faces, same gids, Euler 2 — and then un-close to garbage on the
next `refine()`. So they are written as their own `/faces/closure_parent` and
`/faces/closure_parent_verts` datasets. Three details:

- **They are not user fields.** The two members sit *after* the face user pack,
  so the pack is no longer the face tuple's suffix and every face user-field
  loop in the writer and reader is bounded by `numFaceUserFields<>()`. Emitting
  them as `u<n>`/`u<n+1>` would round-trip by accident but would also inflate
  `n_user_f_fields` and put them in the XDMF attribute list.
- **They hold persistent gids** (a retired red face gid, three vertex gids), so
  unlike `verts`/`edges` they need no dense translation and no ghost fetch.
- **The reader must re-cohere the closure siblings.** Its fresh dense-index
  block partition cuts wherever the block boundaries fall and so routinely
  splits a sibling group across ranks, which `migrate()`'s local round S cannot
  detect; `readMesh()` calls the collective `repairClosureCohesion()` to build a
  `dest` that puts each group back on one rank before migrating.

`format_version` is **2**, and the root attribute `refinement_mode`
(0 = `HangingNode2to1`, 1 = `Conforming`) is written in both modes. That is how
the two file shapes are told apart: reading a hanging-node file into a
conforming mesh, or the reverse, aborts with a named attribute mismatch rather
than producing a mesh with no closure bookkeeping. A version-1 file (no
`refinement_mode` attribute) is not readable by this reader.
