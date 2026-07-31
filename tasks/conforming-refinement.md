# Conforming Refinement

> **Handoff contract.** This file is the source of truth for the conforming-refinement
> work: the design, the task breakdown, and what is done. **At the start of any
> session touching this task, read this file first.**
>
> **How to work this file.** It is *not* a log. When a task completes, **rewrite** the
> affected sections so the file always describes the *current* state of the code —
> update the status table, delete or rewrite any design text the implementation
> superseded, and fold what was learned into the design sections. Only the
> short *Decision record* and *Progress log* at the bottom are append-only.
> **Commit and push at the completion of each task.**
>
> **Do not run tests in Tasks 1–7.** End each of those tasks by committing and
> pushing — nothing else. Do not run `ctest`, do not submit
> `run_regression_minset.flux`, do not run a one-off test binary. **Task 8** is the
> single verification task that runs the full suite and fixes whatever it finds.
> Consequently the per-task **Acceptance** sections in Tasks 1–7 are a
> *specification of what must hold* — write the tests, do not execute them; Task 8
> verifies them. Building (`make -j`) and `--target format-check` are not tests and
> are still expected, so each task's code at least compiles clean when it lands.

## Status

| # | Task | Status |
|---|------|--------|
| 1 | Refinement-mode plumbing (`RefinementMode`, conditional face fields, dispatch) | **Done** (compiles clean as of Task 2) |
| 2 | Serial closure kernel: `closeFaces` / `unclose` + patterns | **Done** |
| 3 | Distributed split-edge discovery (extend Phase 2 to kept faces) | **Done** |
| 4 | Wire closure into distributed `refine()` | Not started |
| 5 | `migrate()` / `loadBalance()` / halo rebuild on a closed mesh | Not started |
| 6 | I/O round-trip, `markByQuality`, example + docs | Not started |
| 7 | Dedicated conforming test suite; flip the default to `Conforming` | Not started |
| 8 | **Run the full suite and fix everything it finds** (the only task that runs tests) | Not started |

Tasks 1–3 have landed. The closure kernel and its inverse exist as pure local
functions in `src/Tessera_RefineClosure.hpp`, and `refineLocal()`'s `Conforming`
branch runs un-close → red split → close on one rank. The distributed `refine()`
now publishes a **complete split-edge map** in `RefineResult::midpoints` (Task 3),
which is the input the closure needs — but its `Conforming` branch is still the
abort stub. Task 4 is next.

---

## Problem

Tessera's adaptive refinement is **non-conforming**: a partial refine mask leaves
T-junctions (hanging nodes), bounded to a 2:1 level jump across any edge
(`README.md` → Known Issues; `src/Tessera_RefineParallel.hpp`). A hanging node means
a half-edge `(a, m)` has only **one** incident face, so:

- owned-only Euler `V − E + F = 2` holds only for a *uniform* refine;
- `CurvatureCriterion`'s edge coordinator silently skips such edges
  (`inc.size() != 2 → continue` in the propagation fixpoint and the analogous
  guard in `Tessera_MarkQuality.hpp`);
- a downstream solver that assembles a surface operator (`applyStencil`,
  `reduceVertexFromFaces`) sees an inconsistent 1-ring at the T-junction vertex.

The consuming codebase requires a conforming triangulation. This task adds one.

---

## Decision 1 — option, not replacement

**Conforming refinement is added as an additional mesh option; the 2:1 hanging-node
path is retained.** Rationale:

1. **It is strictly additive, not a fork.** The design below *builds on* the existing
   2:1-balanced red refinement: the 2:1 balance stays, and conforming refinement is a
   **closure post-pass** over exactly the mesh the current `refine()` already
   produces. Deleting the hanging-node mode would not simplify the conforming path —
   it is a prerequisite of it.
2. **The 2:1 mode is cheaper** — no closure faces, no closure bookkeeping fields, no
   un-close pass — and is sufficient for consumers doing cell-centred work.
3. **The existing regression tests pin the 2:1 mode.** Keeping the mode keeps that
   coverage as a live regression instead of deleting it.

### How the option is spelled

`RefinementMode` is a **compile-time `Mesh` template parameter**, appended last so
every existing `Mesh<...>` spelling stays source-compatible:

```cpp
enum class RefinementMode
{
    HangingNode2to1,   //!< 2:1-bounded hanging nodes (today's behavior)
    Conforming         //!< red-green-blue transient closure; no hanging nodes
};

template <class Scalar, int Dim = 3, class VertexUserFields = VertexFields<>,
          class EdgeUserFields = EdgeFields<>, class FaceUserFields = FaceFields<>,
          class MemorySpace = ..., class ExecutionSpace = ...,
          RefinementMode Mode = RefinementMode::Conforming>
class Mesh;
```

Compile-time was chosen over a runtime flag so the closure bookkeeping fields exist
in the face AoSoA **only** in `Conforming` mode — a hanging-node mesh pays zero
memory. `refine()` dispatches on `MeshT::refinement_mode` with `if constexpr`.

**The default is `Conforming`** (per Decision 2 in the record below). Because the
existing tests and examples assert hanging-node behavior, the default flip is
sequenced **last among the implementation tasks** (Task 7): Tasks 1–6 ship with the
default still `HangingNode2to1`, and Task 7 pins the legacy tests to
`HangingNode2to1` explicitly, then flips the default. Since no task before Task 8
runs tests, the flip lands *before* conforming has ever been executed — Task 8
verifies both modes together, and reverting the flip is Task 8's documented escape
hatch if conforming cannot be made green (Decision 5).

---

## Design — red–green–blue transient closure

### Why closure, and why transient

Two alternatives were rejected:

- **Red-only conforming refinement** (propagate the 1→4 red split until no edge has a
  level jump) degenerates to *uniform* refinement: face A refined forces its three
  edge-neighbours to refine, which cascades over the whole connected component. Not
  usable for AMR.
- **Newest-vertex / longest-edge bisection** is conforming by construction with
  provable shape bounds, but replaces the red engine wholesale: new `Level`
  semantics, a new cross-rank compatible-edge propagation loop, and it breaks the
  icosphere-subdivision correspondence the uniform-refine Euler test relies on.
  Recorded as a possible Milestone-2 alternative; not this task.

So: keep red refinement, add a **green/blue closure** layer. The closure must be
**transient** — recomputed from scratch each refine call — because repeatedly
bisecting an already-bisected closure triangle degrades its shape without bound.
Transient closure gives a bounded number of triangle similarity classes: every
visible triangle is a red-refined triangle or one of three fixed retriangulations of
one.

### Two layers

| Layer | Persistence | Contents |
|---|---|---|
| **Red layer** | persistent, authoritative | Faces produced by red 1→4 splits only, 2:1 balanced. Exactly what `refine()` produces today. Each face carries its red `Level`. |
| **Closure layer** | transient, derived | Green/blue/red retriangulation of red faces that have hanging nodes, so the visible mesh is conforming. |

The **visible mesh** — what `mesh.faces()`, the slices, I/O, and the geometry/stencil
API see — is the closure layer. The red layer is recovered by the un-close pass.

### The key simplification: closure creates no new vertices

A hanging node on a kept face's edge is a midpoint vertex that the *neighbour's* red
split already created. Closure only reconnects existing vertices into new triangles.
Therefore:

- **no new vertex gids**, no `MPI_Exscan` for vertices, no position/field
  interpolation, and **no `RefinePolicy` involvement** in the closure;
- un-closing removes only faces and edges — **never a vertex** — so no vertex state
  is ever discarded by the transient closure;
- closure is **purely local** to one red face, hence to one rank (a face is owned by
  exactly one rank). No communication in the closure step itself.

### Closure patterns

For a kept red face with corners `(a,b,c)`, edges `ab, bc, ca`, let `S` be the subset
of its edges that are bisected in the red layer, with midpoints `m_ab, m_bc, m_ca`.
The 2:1 balance bounds the jump to one level, so each split edge contributes exactly
one midpoint and `|S| ∈ {0,1,2,3}`:

| `|S|` | Pattern | Children | Child count |
|---|---|---|---|
| 0 | none | face emitted unchanged | 1 |
| 1 | **green** (say `ab` split) | `(a, m_ab, c)`, `(m_ab, b, c)` | 2 |
| 2 | **blue** (say `ab`, `bc` split) | `(a, m_ab, c)`, `(m_ab, b, m_bc)`, `(m_ab, m_bc, c)` | 3 |
| 3 | **red-closure** | `(a,m_ab,m_ca)`, `(b,m_bc,m_ab)`, `(c,m_ca,m_bc)`, `(m_ab,m_bc,m_ca)` | 4 |

Rules:

- **Every closure child inherits the parent's `Level` and the parent's face user
  fields.** The closure never changes the red layer — in particular the `|S| = 3`
  red-closure does **not** increment `Level` and does **not** promote the face into
  the red layer. This deliberately avoids a promotion cascade.
- **Winding is preserved**: children are emitted in the parent's CCW order so
  `faceNormalRaw` / `CurvatureCriterion` keep the consistent
  outward-seen-from-outside orientation.
- The **blue** pattern has two valid diagonals. Choose deterministically — connect
  the midpoint with the **lower gid** to the opposite corner — so the output is
  reproducible and rank-count independent (needed for the topology checksum used by
  the migrate/IO tests).
- Face `e[3]` follows the existing convention `e[k] = edge(v[k], v[(k+1)%3])` and is
  re-derived with the rest of the edge table.

**As implemented (Task 2).** The `|S| = 1` and `|S| = 2` patterns are written once
for a *rotated* corner triple `(A,B,C)`, where the rotation is chosen by which edge
is split (green: rotate the split edge to position 0) or unsplit (blue: rotate the
unsplit edge to position 2, `rot = (u+1) % 3`). Two consequences:

- a cyclic rotation of a CCW triple is CCW, so winding preservation is structural
  rather than checked per pattern;
- the working triple `(A,B,C)` is a function of *the triangle and which edge is
  (un)split*, **not** of which corner happens to be stored at `v[0]`. The emitted
  child-triangle set is therefore invariant under a cyclic relabelling of the
  parent's corners, which is the concrete form of partition independence. Pinned by
  the `refine_closure` rotation-invariance check.

**Blue tie-break, precisely.** With `(A,B,C)` as above, `q0 = mid(A,B)` and
`q1 = mid(B,C)`; the corner triangle `(q0, B, q1)` is always cut off and the
remaining quad `(A, q0, q1, C)` is split along the diagonal from the lower-gid
midpoint to *its opposite corner* — `q0 ↔ C` when `q0 < q1`, else `q1 ↔ A`:

| condition | children |
|---|---|
| `q0 < q1` | `(A,q0,C)`, `(q0,B,q1)`, `(q0,q1,C)` |
| `q1 < q0` | `(A,q0,q1)`, `(q0,B,q1)`, `(A,q1,C)` |

This is partition-independent because *both* inputs are global: the two midpoint
gids are bit-identical on every rank sharing the bisected edge (`refine()`'s Phase-2
guarantee, verified by `checkMidpointAgreement`), and `(A,B,C)` is rotation-derived
as above. Nothing partition-local — local index, owner rank, map iteration order,
`v[0]` — enters the comparison. `ClosureStats` counts both branches so a test can
assert the tie-break actually fires in both directions rather than being vacuous.

### Data model — closure bookkeeping

Un-closing must reconstruct the red parent from its children. Each closure child
stores the parent outright, so **any single child determines its parent** with no
sibling lookup:

| Field | Type | Meaning |
|---|---|---|
| `ClosureParent` | `GlobalId` | Parent red face's gid; `invalid_gid` for a red face (no closure). |
| `ClosureParentVerts` | `GlobalId[3]` | Parent's three corner vertex gids, in the parent's winding order. |

**As implemented (Task 1).** The two members live in `ClosureFaceMembers` in
`Tessera_Fields.hpp` and are appended to the face member list **after** the user
pack — *not* added to `CoreFaceMembers` — only when `Mode == Conforming`:

```cpp
using ClosureFaceMembers = Cabana::MemberTypes<GlobalId, GlobalId[3]>;

template <class UserFaceFields, RefinementMode Mode>          // primary
struct FaceMemberTypesImpl { using type = MemberTypesCat_t<CoreFaceMembers, UserFaceFields>; };
template <class UserFaceFields>                                // partial spec on the value
struct FaceMemberTypesImpl<UserFaceFields, RefinementMode::Conforming>
{ using type = MemberTypesCat_t<MemberTypesCat_t<CoreFaceMembers, UserFaceFields>,
                                ClosureFaceMembers>; };
```

so the layout is `[core | user | (closure)]`. Rationale: `FaceField::UserBegin` is
the hard-coded constant `5` and `userFaceField<M>()` is `UserBegin + M`; inserting
closure fields among the core members would shift every user field index and
silently break existing consumers. With the append, `UserBegin` stays `5` and the
closure indices are computed from the user pack size —
`closureParentField<FaceUserFields>()` / `closureParentVertsField<...>()`, also
re-exported as `MeshT::closure_parent_field` / `closure_parent_verts_field`.

> **Consequence — the user pack is no longer the tuple's suffix on faces.** Any code
> sizing a face user-field loop as `face_member_types::size - FaceField::UserBegin`
> over-counts by two in `Conforming` mode and would treat the closure members as
> user data. Task 1 added `numFaceUserFields<UserFaceFields>()` (== the pack size)
> plus `detail::copyUserFieldsN<UserBegin, N>` and switched both face-field copy
> sites in `Tessera_Refine.hpp` / `Tessera_RefineParallel.hpp` to it. The
> tuple-suffix-derived `detail::copyUserFields<UserBegin>` remains, correct for
> vertices and edges. **The HDF5 writer/reader still derives its face field set
> from the tuple suffix — Task 6 must audit it against `numFaceUserFields<>()`.**

> **Consequence — the closure members must be explicitly initialized.** A Cabana
> `AoSoA`'s backing View is *zero*-initialized, so an untouched `ClosureParent`
> reads as face gid `0` — a perfectly valid gid — and `unclose()` would restore a
> bogus parent for every face. Task 2 added
> `initClosureFaceMembers<MeshT>( hostFaceAoSoA, begin, end )` (a no-op in
> `HangingNode2to1` mode) and calls it from `buildTriangleMesh()` in
> `Tessera_MeshBuilder.hpp`. `distribute()` / `migrate()` / `haloExchange()` are
> generic over the whole face tuple and carry the members through unchanged, so they
> need nothing. **The HDF5 reader materializes faces too — Task 6 must call it
> there.** `unclose()` guards the failure mode anyway: it aborts if two restored red
> faces share a gid, which is exactly what an uninitialized field produces.

**Memory.** Only the *visible* face array carries the two fields, in `Conforming`
mode only: 4 extra `GlobalId` per face on a ~78 B face core. A compressed encoding
(parent gid + a pattern/child-index byte pair, reconstructing parent corners from
siblings) is a possible follow-up — record it in README *Future Optimizations*, not
in this task.

### Distributed algorithm — what changes in `refine()`

`refine()` in `Tessera_RefineParallel.hpp` gains three steps. Phases 1–3 are the
existing code, unchanged in substance.

```
Conforming refine( mesh, halo, mask ):

  0. UN-CLOSE (new, local, no comm)
       For each distinct ClosureParent gid held locally, emit one red parent face:
         gid   = ClosureParent
         v[3]  = ClosureParentVerts
         level = the child's Level (closure children carry the parent's level)
         user fields = from the lowest-gid child (all children inherited an
                       identical copy from the parent)
       Red faces (ClosureParent == invalid_gid) pass through unchanged.
       Result: the persistent red layer, exactly the shape phases 1-3 expect.

  0b. MASK TRANSLATION (new, local)
       A red parent is marked iff ANY of its closure children was marked in the
       caller's visible-face mask. The caller's mask is indexed by VISIBLE owned
       faces; phases 1-3 want it indexed by RED owned faces.

  1. 2:1 mark-propagation fixpoint          [unchanged]
  2. midpoint-gid assignment                [EXTENDED, see below]
  3. red face list: kept + 4 children each  [unchanged]

  3b. CLOSE (new, local, no comm)
       For each KEPT red face, look up its 3 edges in the split-edge map from
       phase 2; apply the |S| pattern above; emit the closure children with
       ClosureParent / ClosureParentVerts set. Red children of a REFINED face
       are never closed (their edges are all newly created, so |S| = 0 for them
       by construction — assert this).

  3c. face gid allocation                   [EXTENDED: must cover closure children]
  3d..3i. edges / keys / CSR / ownership    [unchanged, driven by the final face list]
```

**Phase 2 extension — kept faces must learn their split edges. (Landed, Task 3.)**
Before Task 3, Phase 2 advertised only *refining* faces' edges to the edge
coordinator, and only refining participants received the midpoint gid. A kept face
on rank R needs to know which of its edges were bisected by a refining neighbour on
rank R′. The existing rounds were extended — **no new communication rounds**:

- **2a:** `detail::OwnerMsg` gained an `unsigned char refining` field and is now
  emitted for the edges of *all* owned faces, refining and kept.
- **2b:** the coordinator aggregates per edge into `{participants, refOwner,
  anyRefining}`. An edge is **split iff `anyRefining`** — one with no refining
  incidence is dropped right there, so the extra advertisements create no
  downstream state. For a split edge the midpoint owner is the minimum *refining*
  participant (rule unchanged), and `(key, midOwner)` is replied to **all**
  participants, with `(key, cosharer)` sent to the owner for each other
  participant.
- **2c:** unchanged code — the midpoint owner therefore ships the gid to **all**
  co-sharers, refining or kept.

Every rank then holds `EdgeKey → midpoint gid` for every split edge it touches, which
is exactly the split-edge map step 3b needs. It is published as
**`RefineResult::midpoints`**, whose contract was widened from "edges of my refining
faces" to "edges of *any* of my owned faces that were bisected" — a superset, so
`checkMidpointAgreement` (its other consumer) only gets stronger. `RefineResult`
also gained `phase2Adverts` / `phase2AdvertsRefining` counters so a test can report
the added message volume without instrumenting the library at the call site.

**Why the midpoint assignment is bit-identical to before.** Ownership reads only
refining participants, and the set of split edges is unchanged (an edge with no
refining incidence is dropped), so `myMid` — the owned-midpoint list each rank
exscans over — is the same list in the same order as before. Vertex gids, counts,
and positions are therefore untouched; only the *distribution* of already-decided
gids got wider.

**A closure child may reference a non-local vertex.** The midpoint gid arrives, the
midpoint *position* does not. This is not new: `refine()` already leaves an
owned-only mesh whose faces reference vertex gids the rank does not hold (see the
post-refine gotcha in `docs/design.md`). The halo rebuild inside `migrate()` brings
them in. Do **not** add a position gather to the closure.

**Face gid allocation.** Closure children are new faces and need new gids. Fold them
into the existing single `MPI_Exscan`: count
`4 * nRefining + countClosureChildren(newRed, midpointOf)` per rank (that helper
exists precisely so the count is available *before* `closeFaces()` runs) and allocate
one contiguous block above the global max face gid. A retired
parent gid is *reused* by un-close on the next round; since it is below the global
max it can never collide with a later allocation. Live face gids stay sparse — which
is already true today and already handled.

### Interaction with the rest of the library

| Area | Impact |
|---|---|
| `Level` semantics | Face `Level` remains the **red** level. A closure child carries its parent's level, so in `Conforming` mode `Level` no longer maps 1:1 to triangle size. Edge level stays `min` of incident face levels. Document in `docs/design.md`. |
| `migrate()` / `loadBalance()` | The visible (closed) mesh migrates. Un-close is local *per child*, so siblings need not stay co-resident for correctness — but two ranks each holding a child of the same parent would each restore the parent, duplicating it. **Fix: constrain the `dest` array so all closure siblings follow the lowest-gid sibling's destination.** Siblings are co-resident *before* migrate, so this fixup is purely local. `loadBalance()` should weight a parent's children as one unit. Task 5. |
| Halo rebuild | Unchanged in mechanism. On a conforming mesh every edge has exactly 2 incident faces, so the 1-ring closure invariant is cleaner, not harder. |
| I/O | The two closure fields must round-trip through HDF5/XDMF, else a read-back mesh cannot be un-closed. Task 6. |
| `markByQuality` | Returns a mask over **visible** owned faces; step 0b translates it. `CurvatureCriterion`'s "exactly two incident faces" coordinator assumption becomes *true* in conforming mode rather than silently skipped — a correctness improvement. |
| `MeshGeometry` / `VertexStencil` | No change; they read the visible mesh and are already generation-guarded. Conforming topology makes the k=1 stencil consistent at former T-junctions. |
| `RefinePolicy` | Untouched. The closure interpolates nothing. |
| Profiling | Add level-2 regions `refine_unclose`, `refine_close`; level-3 `refine_closure_patterns`. |

### Invariants and acceptance

New shared checks in `tests/MeshInvariants.hpp`:

- `checkConforming(mesh)` — every edge in the global mesh has **exactly two**
  incident faces, verified through the edge coordinator (`edgeCoordRank` +
  `allToAllV`), *not* the halo, so the verdict is rank-count independent.
- `checkOwnedEuler(mesh) == 2` — owned-only `V − E + F = 2` for an **arbitrary**
  (adaptive, not just uniform) mask. This is the headline acceptance criterion: it is
  precisely what fails today.
- `checkNoInteriorVertex(mesh)` — no vertex lies in the interior of any edge, i.e.
  for every edge `(u,w)` no vertex `m` exists with edges `(u,m)` and `(m,w)`.
- `checkClosureInverse(mesh)` — `unclose ∘ close` reproduces the red face set
  (gids, corner gids, levels, user fields) bit-for-bit.
- `check21Balance(mesh)` — existing check, now applied to the **red layer** (after
  un-close) rather than the visible mesh.

The existing `checkMidpointAgreement` and `checkOwnershipPartition` must keep
passing unchanged.

**Gate.** New conforming tests join the ship gate at label `regression` × backends
{SERIAL, HIP} × ranks {1,2,3,4,5} — the gate definition itself does not change, so
no edits to `CLAUDE.md`, `run_regression_minset.flux`, or CI are needed. Existing
hanging-node tests stay in the gate.

### Known limits to document, not solve

- Face **user** fields on closure faces are the parent's, copied. If a solver writes
  per-closure-face state, un-close takes the lowest-gid child's values and the rest
  are discarded. Document as the contract (it parallels the existing "edge user
  fields are reset by `refine()`" rule).
- Edge user fields continue to be reset by `refine()`.
- Coarsening / edge collapse remains out of scope.

---

## Tasks

Each task is one Claude Code session. Each session must **rewrite** the affected
sections of this file (design text and status table) to match what landed — not
append to it.

**Tasks 1–7 end at a commit + push, with no test run** (see the handoff contract
above). Their **Acceptance** sections specify the tests to *write* and the properties
those tests must pin; **Task 8** is where the suite is executed and failures are
fixed. Do compile and keep `format-check` clean as you go — a task that lands
non-compiling code makes Task 8 unable to distinguish a build break from a real
defect.

Work on the `conforming-refinement` branch (it exists and currently points at
`redesign`).

---

### Task 1 — Refinement-mode plumbing — **DONE**

**Goal.** The option exists and is inert. No behavior change.

**What landed.**

- `src/Tessera_RefinementMode.hpp` (new, in `<Tessera.hpp>`) —
  `enum class RefinementMode { HangingNode2to1, Conforming }` with the contract of
  each mode documented on the enumerators.
- `Mesh` gained an 8th and last template parameter
  `RefinementMode Mode = RefinementMode::HangingNode2to1`, exposed as
  `static constexpr RefinementMode refinement_mode`. **No call site changed** —
  every existing spelling in `src/`, `tests/`, and `examples/` passes exactly seven
  arguments. *(Default flips to `Conforming` in Task 7.)*
  `Mesh` also now re-exports `vertex_user_fields` / `edge_user_fields` /
  `face_user_fields` (needed to size a user-field loop, see below) and the
  `closure_parent_field` / `closure_parent_verts_field` slice indices.
- `Tessera_Fields.hpp` — `ClosureFaceMembers`, the `FaceMemberTypesImpl` partial
  specialization on the mode value, `numFaceUserFields<>()`,
  `closureParentField<>()`, `closureParentVertsField<>()`. See *Data model —
  closure bookkeeping* above for the spelling and the user-pack-is-no-longer-the-
  suffix consequence.
- `refineLocal()` and `refine()` are now thin dispatchers on
  `MeshT::refinement_mode`; the existing bodies moved verbatim into
  `detail::refineLocalHangingNode()` / `detail::refineHangingNode()`. The
  `Conforming` branch is a runtime `Kokkos::abort("... not implemented yet ...")`
  naming the task that will implement it — deliberately not a `static_assert`, so
  a `Conforming` mesh type stays instantiable (Task 2+ tests need to construct one
  before the kernels exist).
- `docs/design.md` gained an *Adaptive refinement → Refinement modes* subsection;
  `README.md`'s API sketch documents the 8th parameter and its *Known Issues*
  non-conforming entry now records conforming as in progress with the abort stub.
- `tests/test_refinement_mode.cpp` + registration (`unit`, SERIAL + HIP, np1).

**Notes for later tasks.**

- **Cabana friction: none.** `Cabana::MemberTypes` concatenation via the existing
  `MemberTypesCat_t` composed cleanly, and a partial specialization on the
  `RefinementMode` *value* works without a tag-type detour. `AoSoA`,
  `MemberTypeAtIndex`, and `Cabana::slice<I>` are all indifferent to the two extra
  members.
- The `Conforming` face tuple is 9 members for an empty user pack (7 + 2), and the
  closure members are always the **last two**, at `tupleSize-2` / `tupleSize-1`.
- The `Kokkos::abort` in a host-side `if constexpr` branch compiles but is not a
  `[[noreturn]]` the compiler can see through, so `refine()`'s `Conforming` branch
  also returns a default `RefineResult{}` to keep the return path well-formed.

**Acceptance.**
`unit` test `refinement_mode` (SERIAL + HIP, np1) — written, not run (see the
handoff contract). It pins: a `Conforming`-typed mesh instantiates, constructs,
resizes, and round-trips written closure-member values host↔device;
`FaceField::UserBegin == 5` and `userFaceField<0..1>()` indices *and types* are
identical in both modes; the `Conforming` face tuple is exactly two members larger
and the closure indices are its last two, with types `GlobalId` / `GlobalId[3]`;
vertex and edge member lists are mode-independent;
`numFaceUserFields<>()` counts the pack while the tuple suffix over-counts by two.
Most of these are `static_assert`s (a `UserBegin` shift must be a build failure,
not a silent wrong read) mirrored by run-time checks so a failure names itself.
The test deliberately does **not** assert what the 7-argument default mode *is*, so
it survives Task 7's flip.

**Report back.** *(delivered — see Notes above; the short version: no Cabana
friction, no call site broken, and the one real hazard the layout choice creates is
the face user pack no longer being the tuple suffix, which the I/O path still
assumes — flagged for Task 6.)*

---

### Task 2 — Serial closure kernel — **DONE**

**Goal.** The closure and its inverse exist as pure, testable local functions.

**What landed.**

`src/Tessera_RefineClosure.hpp` (new, in `<Tessera.hpp>`, depends only on
`Tessera_Fields` / `Tessera_RefinementMode` / `Tessera_Types` so both
`Tessera_MeshBuilder.hpp` and `Tessera_Refine.hpp` can include it without a cycle):

| Symbol | Role |
|---|---|
| `RedFace { v[3], gid, level }` | one face of the persistent red layer |
| `VisibleFace { v[3], gid, level, parent, parentVerts[3] }` | one face of the visible layer; `parent == invalid_gid` ⇒ a passed-through red face |
| `ClosureStats` | `|S|` histogram, visible/closure-child counts, per-diagonal blue counts |
| `closureChildCount(nSplit)` | `nSplit + 1` |
| `faceSplitEdges(v, midpointOf, mid)` | fills `mid[k]` per edge, returns `|S|` |
| `countClosureChildren(red, midpointOf)` | new-gid count, for Task 4's `MPI_Exscan` *before* closing |
| `closeFaces(red, midpointOf, firstChildGid, freshChild = {})` → `CloseResult` | the patterns; children take consecutive gids from `firstChildGid` |
| `unclose(visible)` → `UncloseResult` | the inverse |
| `translateMask(visibleMask, uncloseResult)` | visible-face mask → red-face mask |
| `readVisibleFaces<MeshT>(hostFaceAoSoA, n)` | AoSoA → `VisibleFace` vector |
| `writeClosureFace<MeshT>(hf, f, vf)` / `initClosureFaceMembers<MeshT>(hf, b, e)` | the write side; both no-ops in `HangingNode2to1` mode |

`CloseResult` carries `sourceRed[i]` (which red face visible face *i* takes its face
user fields from) and `UncloseResult` carries `sourceVisible[r]` (the lowest-gid
child) plus `redOfVisible[i]` — the latter is what makes `translateMask()` a plain
scatter with no gid map, and Task 4 will reuse it.

`detail::refineLocalConforming()` in `Tessera_Refine.hpp` runs
un-close → mask translation → red 1→4 split → close, then re-derives edges, keys, and
the vertex 1-ring CSR from the **visible** face list. `refineLocal()`'s dispatcher
calls it; the abort stub is gone. Two contract changes, both documented at the
function:

- **Face gids are no longer local indices** in `Conforming` mode. A closed red face's
  gid is retired into its children's `ClosureParent`, so it must not be reissued:
  red gids persist across the call and closure children are allocated above the
  current max. (Vertex and edge gids still equal their index.) Live face gids are
  therefore sparse — already true of the distributed `refine()`. As a knock-on, the
  edge AoSoA's incident-face field now stores real face **gids** rather than indices
  that happened to equal them.
- **`refineLocal()` enforces no 2:1 balance** (neither mode does — that is
  `refine()`'s Phase 1). *One* call from a balanced mesh bisects each edge of a kept
  face at most once, so the patterns apply; a *sequence* of adaptive `refineLocal()`
  calls can drive a >2:1 jump, at which point an edge carries more than one midpoint
  and no fixed pattern applies. `closeFaces()` detects exactly that — it checks
  whether either half-edge `(v[k], m)` / `(m, v[k+1])` of a split edge is itself in
  the split-edge map — and aborts, rather than silently emitting a mesh that still
  has hanging nodes.

**Where the "red child of a refined face has `|S| = 0`" assert lives.** Inside
`closeFaces()`, driven by the optional `freshChild` flag array that
`refineLocalConforming()` passes (Task 4's `refine()` will pass the same). It sits
where `|S|` is already computed, so it costs nothing, and it is a property of the
*pair* (red list, split-edge map) — checking it in the caller would duplicate the
`|S|` computation. **Why it holds by construction:** the 1→4 split replaces parent
`(a,b,c)` with children whose nine edges are the six half-edges `(a,m_ab)`,
`(m_ab,b)`, … and the three interior edges `(m_ab,m_bc)`, … Every one of those edges
has a *midpoint vertex of this round* as an endpoint, and the split-edge map is keyed
by edges of the **pre-split** red layer, whose endpoints are all pre-existing
vertices. So no child edge can be a key in the map. It can only fire if the map is
polluted with an edge that did not exist before the split — which is the real bug it
is there to catch. Whether it ever fires is a Task 8 observation.

**Extra hazard found and fixed: closure-member initialization.** See the consequence
box under *Data model — closure bookkeeping* above. This was not in the design and is
the one thing Task 2 had to add outside the closure header.

**Acceptance.** New `unit` test `refine_closure` (SERIAL + HIP, np1) — written and
compiled, **not run** (see the handoff contract). It pins:

- each of the four `|S|` cases against hand-written child triangles, in emission
  order, for a hand-built parent; CCW winding via a planar embedding of the parent
  (signed area > 0); consecutive child gids from `firstChildGid`; every child
  carrying the parent's level and `ClosureParent`/`ClosureParentVerts`; the `|S| = 3`
  red-closure *not* incrementing the level; `unclose()` restoring the parent exactly
  from any single child;
- the green pattern for each of the three edges in turn (the rotation);
- the blue tie-break in **both** directions, by relabelling the two midpoint gids so
  the diagonal must flip, with `ClosureStats`' per-diagonal counters asserted;
- **rotation invariance**: cyclically relabelling the parent's corners leaves the
  emitted triangle *set* (sorted corner-gid triples) unchanged, for all four `|S|`;
- `translateMask()`: any single marked child marks the parent, no marked child marks
  nothing, and a random visible mask maps to exactly the owning red set;
- `unclose ∘ close == identity` on a hand-built red layer from a random (seeded LCG)
  ~35 % mask over `buildIcosphere(3)`, compared gid-keyed on gid/corners/level;
  plus `countClosureChildren()` agreeing with the number actually emitted, and
  `sourceVisible[r]` being the lowest-gid child;
- `refineLocal()` with a **partial** mask on `buildIcosphere(2)` in `Conforming`
  mode: owned Euler `V−E+F == 2`, every edge with exactly two incident faces, no
  vertex in the interior of an edge, all `faceNormalRaw` outward, and the **same
  vertex count** as `HangingNode2to1` mode; face user fields inherited (all children
  of one parent agree, every value is one of the seeded base-face values).

The "no interior vertex" check is **geometric, not topological**: the topological
reading ("some `m` has edges `(u,m)` and `(m,w)`") is true of every ordinary
triangle. It intersects the neighbour sets of `u` and `w` and tests each candidate for
collinearity with *and* strict betweenness on the segment.

**Non-vacuity guard.** The three conformity checks are also run on the
`HangingNode2to1` result *with the same mask*, and the test **fails** if they pass
there — otherwise a mask too weak to create a hanging node would make the conforming
case prove nothing.

**Report back.** *(delivered — see the blue-tie-break and assert subsections above.
Short version: the tie-break compares the two globally-agreed midpoint gids over a
rotation-derived corner triple, so it reads no partition-local quantity; the
`|S| = 0` assert lives in `closeFaces()` behind the `freshChild` flag and holds
because every edge of a fresh child has a this-round midpoint as an endpoint while
the split-edge map is keyed by pre-split edges. The one design gap was closure-member
initialization.)*

---

### Task 3 — Distributed split-edge discovery — **DONE**

**Goal.** Every rank learns the midpoint gid of every split edge it touches,
including on kept faces. No closure yet.

**What landed.** All of it in `src/Tessera_RefineParallel.hpp` — see *Phase 2
extension* above for the spelling. `detail::OwnerMsg` gained `refining`; 2a
advertises every owned face; 2b aggregates `{participants, refOwner, anyRefining}`
and drops non-split edges; 2c is unchanged and now reaches kept co-sharers.
`RefineResult::midpoints` is documented as the split-edge map, and
`phase2Adverts` / `phase2AdvertsRefining` were added for the Task-8 volume
measurement. No call site changed; `refine_parallel` and both `markquality` tests
were not edited.

**The `inc.size() != 2` guard is unaffected.** That guard lives in **Phase 1**, the
2:1 mark-propagation fixpoint, which *already* advertised every owned face's edges
(one `PropMsg` per (owned face, edge)). Task 3 did not touch Phase 1, so the
coordinator still receives exactly two `PropMsg`s per edge of a closed surface — one
from each incident face, since a face is owned by exactly one rank and every face
advertises. **Phase 2 has no such guard**: it aggregates by *participant rank*, not
by incidence, into a `std::set<Rank>` whose size is 1 (both incident faces on one
rank) or 2. A count-of-two assumption would have been wrong there both before and
after this change, and none is made.

**Estimated message-volume increase (Task 8 measures the actual).** Phase 2a goes
from `3·nRefining` to `3·nOwnedF` messages per rank — a factor of `1/ρ` where
`ρ = nRefining/nOwnedF` is the post-fixpoint refine fraction. Downstream rounds grow
far less: 2b's replies and 2c's gid sends are per *(split edge, participant)* and a
split edge has at most 2 participants either way, so their volume rises only by the
edges whose kept side sits on a different rank — an O(boundary) term, not O(F). The
test's fixture (`gid % 7`, subdiv-2 icosphere, no fixpoint propagation at level 0)
has `ρ ≈ 1/7`, so 2a should be roughly **7×** on round 1 and closer to `1/ρ` for the
later rounds' larger `ρ`. Since 2a is one `allToAllV` of a 24-byte struct over
`3·nOwnedF` entries, this is a constant-factor bump on the cheapest of Phase 2's
rounds, not a complexity change. The test prints total-vs-refining-only per round.

**Acceptance.** New `regression` test `refine_splitedges` (SERIAL + HIP, ranks 1–5)
— written and compiled, **not run** (handoff contract). It pins:

- **round-1 exactness against a partition-free reference.** On the level-0 subdiv-2
  icosphere the 2:1 fixpoint provably cannot fire (all levels equal ⇒ no
  final-level gap reaches 2), so the refining set *is* the caller's mask and the
  split set is exactly the edges of `{f : gid % 7 == 0}` — computed from a
  replicated, un-partitioned mesh with no MPI. Each rank's key set must equal that
  reference restricted to the edges of its own pre-refine owned faces.
- **the midpoint gid block**: the globally distinct gids are exactly
  `[V0, V0 + |S|)`, checked by count *and* sum *and* XOR against the closed forms,
  so a duplicate and a gap both fail. This is the check that would catch the
  extension accidentally creating or renumbering a midpoint.
- **non-vacuity**: at ranks ≥ 2 the global count of split edges a rank holds that
  lie on *none* of its refining owned faces must be **positive** — those are
  exactly the kept-side discoveries the pre-Task-3 code missed. Zero ⇒ the test
  proved nothing ⇒ it fails. (It is legitimately 0 at np1, so the check is
  rank-gated.)
- **multi-round soundness + completeness** over three further adaptive rounds,
  where levels do differ and the fixpoint really propagates so no cheap serial
  reference exists. `checkSplitEdgeCoverage()` derives the ground truth from the
  edge coordinator instead: an edge is split iff *some* rank reports it; then every
  rank touching it must report it (completeness) and no rank may report an edge
  none of its faces touches (soundness). Plus `checkMidpointAgreement` and
  `check21Balance` each round.

**Report back.** *(delivered — see the `inc.size() != 2` and message-volume
subsections above. Short version: the two-incidence guard is Phase 1's and Phase 1
was already advertising every owned face, so nothing there changed; Phase 2 never
assumed two incidences. Estimated 2a growth is `1/ρ` ≈ 7× on the test fixture's
round 1, with the reply/gid rounds growing only by an O(partition-boundary) term.)*

---

### Task 4 — Wire closure into distributed `refine()`

**Goal.** `Conforming` distributed refinement works. This is the milestone.

- Insert step 0 (un-close), 0b (mask translation), 3b (close), and the extended
  face-gid allocation into `refine()`, all under `if constexpr Conforming`.
- Add `checkConforming`, `checkOwnedEuler`, `checkNoInteriorVertex`,
  `checkClosureInverse` to `tests/MeshInvariants.hpp`; apply `check21Balance` to the
  red layer.
- Add the `refine_unclose` / `refine_close` profiling regions.

**Acceptance.** New `regression` test `refine_conforming` (SERIAL + HIP, ranks 1–5):
over **three successive adaptive refine rounds** — `checkConforming`,
owned Euler `== 2` (the criterion that fails today), no interior vertex,
`checkOwnershipPartition`, `checkMidpointAgreement`, `check21Balance` on the red
layer, and `checkClosureInverse`. Plus the degenerate empty-mask and full-mask
(uniform → closure is a no-op, `|S| = 0` everywhere) cases.

Have the test **print** the closure-face fraction and the `|S|`-pattern histogram per
round, so Task 8's run yields those numbers without a re-run.

**Report back.** Any place the red-layer/visible-layer distinction leaked into a
caller-visible API; anything in the un-close / mask-translation / close insertion that
the design section got wrong (and rewrite that section). Closure-face fractions and
`|S|` distributions are Task 8 measurements.

---

### Task 5 — migrate / loadBalance / halo on a closed mesh

**Goal.** A conforming mesh survives redistribution and re-haloing.

- Sibling-cohesion fixup on the `dest` array in `migrate()`: all closure children of
  one parent follow the lowest-gid sibling's destination (local — siblings are
  co-resident pre-migrate). Reject or repair an external `dest` that violates it,
  loudly.
- `loadBalance()` / `computeLoadBalance()`: weight a red parent's closure children as
  one unit so Zoltan2 does not see the closure as spurious load.
- Confirm the halo rebuild inside `migrate()` closes the 1-ring on a conforming mesh.

**Acceptance.** Extend `refine_conforming` (or add `regression` test
`conforming_migrate`, SERIAL + HIP, ranks 1–5): refine → migrate → `haloExchange` →
all conforming invariants still hold; the rank-count-independent topology checksum
matches across ranks 1–5; siblings are co-resident post-migrate;
`owned1RingLocal` passes; `loadBalance` reduces imbalance while preserving every
conforming invariant.

Have the test **print** the count of `dest` entries the sibling fixup moved, and the
pre/post imbalance figures, so Task 8's run yields them directly.

**Report back.** Whether a sibling-cohesion-violating external `dest` is *rejected* or
*repaired*, and why you chose that; whether the Zoltan2 weighting changed the
`computeLoadBalance` signature or the `ownedFaceWeights` contract. Fixup counts and
balance-quality numbers are Task 8 measurements.

---

### Task 6 — I/O, marking, example, docs

**Goal.** The feature is usable and documented end to end.

- Persist `ClosureParent` / `ClosureParentVerts` in the HDF5 writer/reader. Audit the
  writer's face field set against `numFaceUserFields<>()` (it still derives it from
  the tuple suffix, which over-counts by two in `Conforming` mode), and call
  `initClosureFaceMembers<MeshT>()` on the reader's freshly-materialized face AoSoA —
  a zero-filled `ClosureParent` reads as face gid 0.
- Confirm `markByQuality` (both criteria) drives `Conforming` refine correctly through
  the mask translation; confirm `CurvatureCriterion`'s coordinator now sees two
  incident faces at former T-junction edges.
- `examples/02_mesh_pipeline`: add `--refine-mode {hanging,conforming}` (two mesh
  type instantiations). Mirror the new argument into the README example table.
- Rewrite the `docs/design.md` *Adaptive refinement* section for both modes; remove
  the **non-conforming** entry from README *Known Issues*; add the compressed
  closure-encoding idea to README *Future Optimizations*.

**Acceptance.** `io` regression test extended: write → read → un-close a conforming
mesh and recover the red layer bit-for-bit. Both `markquality` tests gain a
conforming variant. README/`docs/design.md` in sync; `format-check` clean.

**Report back.** Whether the reader needed a format-version bump, and how a
hanging-node-written file and a conforming-written file are told apart on read; the
*computed* on-disk size delta per face (`+4 × sizeof(GlobalId)`) — Task 8 reports the
actual file sizes.

---

### Task 7 — Dedicated conforming test suite; flip the default

**Goal.** Conforming refinement has first-class test coverage of its own — not just
the per-feature tests Tasks 2–6 added along the way — and `Conforming` becomes the
default.

This is the largest task. Do it in the order below and **commit at each of the three
checkpoints** so a session that runs long leaves the repo in a reviewable state.

#### Checkpoint A — mode-parity registration

Pinning the legacy tests to `HangingNode2to1` (needed for the flip) would otherwise
*remove* conforming coverage from those paths. So pin **and** register a conforming
counterpart for every mode-sensitive test:

| Existing test | Pin to hanging-node | Conforming counterpart |
|---|---|---|
| `refine` (unit, np1) | yes | `refine_closure` (Task 2) — already covers `refineLocal` conforming |
| `refine_parallel` (regression) | yes | `refine_conforming` (Task 4) |
| `migrate_mesh`, `loadbalance` (regression) | yes | `conforming_migrate` (Task 5) |
| `io` (regression) | yes | `io` conforming case (Task 6) |
| `markquality_edge`, `markquality_curv` (regression) | yes | conforming variants (Task 6) |
| `distribute`, `connectivity`, `keys`, `data_model`, `halo`, `migrate`, `geometry`, `global_reduce` | n/a — no `refine()` call, mode-insensitive | — |

Mode-sensitive tests must be registered in **both** modes, not switched over.

#### Checkpoint B — new conforming-specific tests

These test properties that only exist in conforming mode and that no earlier task
covers. All are new files.

1. **`conforming_operators`** — `regression`, SERIAL + HIP, ranks 1–5.
   **The payoff test: the reason the downstream solver needs this feature.**
   Pipeline: `buildIcosphere` → `distribute` → adaptive conforming `refine` →
   `migrate` → `haloExchange`, then
   - identify the **closure vertices** (former hanging nodes: vertices that are the
     midpoint of some closure parent's edge) and assert the set is non-empty at every
     rank count — otherwise the test is vacuous and must fail loudly;
   - `buildVertexStencil(mesh, 1)`: each closure vertex's `k=1` CSR row equals the
     true edge-derived 1-ring, and every locally-held face incident to it contains it.
     This is exactly what is inconsistent on a hanging-node mesh;
   - `applyStencil` with the analytic field `f(p) = p_x` and uniform weights:
     `out(i)` matches a reference computed independently from haloed positions, **to
     the same tolerance at closure vertices as at interior vertices** (assert the two
     max-error figures separately so a closure-specific regression cannot hide in an
     aggregate);
   - `reduceVertexFromFaces` with the one-third-area op: global owned-vertex sum
     equals the global owned-face `faceArea` sum (partition-independent identity),
     and each closure vertex's accumulated area is the true one-third sum over its
     full incident-face set.

2. **`conforming_determinism`** — `regression`, SERIAL + HIP, ranks 1–5.
   Closure output must not depend on decomposition or on how many times it is applied:
   - **Rank-count independence:** a rank-count-independent topology checksum over the
     visible mesh (sorted owned face corner-gid triples, plus V/E/F counts) is
     identical at ranks 1, 2, 3, 4, 5 for the same geometrically-defined mask. This is
     the test that pins the **blue-diagonal lower-gid tie-break** — a
     partition-dependent tie-break passes everywhere else and fails only here.
   - **Closure idempotence:** conforming `refine` with an **empty** mask on an
     already-closed mesh reproduces the visible topology bit-for-bit (un-close →
     no red split → re-close is the identity).
   - **Cross-mode equivalence under a uniform mask:** with every face marked there are
     no kept faces, so `|S| = 0` everywhere and no closure children are emitted —
     the two modes must produce **identical topology** (gids, corner gids, levels;
     compare topology, not tuples, since the conforming face has two extra fields).
     A mismatch here means the closure fired when it should have been inert.

3. **`conforming_quality`** — shape quality over depth, the transient-closure
   guarantee. Adaptive refine for **≥8 rounds**, tracking global min triangle angle
   and max aspect ratio per round; assert both stay within a fixed bound independent
   of round count, and record the measured bounds in the test output. Also assert the
   closure-face fraction stays bounded (closure is an O(level-jump-boundary) set, not
   O(F)).
   Lands as `regression` at ranks 1–5 **if stable**; `unit` otherwise — and if it is
   `unit`, record why in README *Known Issues* and report it. Never promote a failing
   test to the gate.

4. **`staleslice_guard` extension** — the conforming refine path un-closes and
   re-closes, changing local counts, so a slice or `MeshGeometry`/`VertexStencil`
   handle held across it must abort. Add the conforming case to the existing
   SERIAL-only np1 test rather than a new file.

#### Checkpoint C — flip the default and sync docs

- Flip the `Mesh` template default to `RefinementMode::Conforming`.
- Final sweep of `README.md`, `docs/design.md`, and `CLAUDE.md` for the new default.
- Rewrite this file's *Status* table and design sections to describe the shipped
  state.

**Acceptance.** Both modes are registered and covered across the suite (the parity
table above is complete), the three new tests are written and registered at their
intended tiers/ranks, the `Mesh` default is `Conforming`, docs are in sync, and
`format-check` is clean.

`conforming_quality`'s angle / aspect-ratio bounds cannot be chosen from measurements
yet — no test has run. **Register it as `unit` with the bounds you believe are
correct and a comment marking them provisional.** Task 8 measures the real values and
decides whether it is stable enough to promote to `regression`. Do not guess a bound
and register it in the gate.

Have all three tests **print** their measurements (closure-vertex counts per rank
count, closure-vertex vs interior-vertex `applyStencil` max error, per-round min angle
/ max aspect ratio / closure-face fraction) so Task 8's run yields the numbers
directly.

**Report back.** Which legacy tests needed explicit pinning and whether any resisted
it; how `conforming_operators` identifies the closure-vertex set, and how it fails
loudly if that set is empty; the provisional quality bounds and the reasoning behind
them; anything the default flip broke at compile time.

---

### Task 8 — Run the full suite and fix everything it finds

**Goal.** The whole test suite is green. **This is the only task in this plan that
runs tests** — Tasks 1–7 wrote code and tests without executing them, so this is the
first execution of any of it.

Budget a long session. If the failure list is large, treat each fix as its own
commit + push and keep the *Open failures* list below current, so the task can be
resumed by a later session.

**Procedure.**

```bash
hostname                                          # expect tuolumne* → systems/tuolumne/claude.md
spack env activate ~/spack_envs/tuolumne_trilinos/
cd build-tuolumne && bash ../run_cmake_toulumne.sh && make -j $(nproc)
cmake --build . --target format-check

flux batch ../scripts/tuolumne/run_regression_minset.flux   # gate: regression x {SERIAL,HIP} x ranks 1-5
ctest -L unit --output-on-failure                            # unit tier
```

Fix in **dependency order**, not failure-count order — a Task 1 field-index bug will
manifest as a dozen unrelated failures downstream:
build errors → `refinement_mode`/`refine_closure` (units) → `refine_splitedges` →
`refine_conforming` → `conforming_migrate` → `io` → `markquality` → the Task 7
conforming suite → the legacy hanging-node tests.

**Triage first at these known risk points** (each is a place the design admits a
plausible defect):

1. **`FaceField::UserBegin` shift.** If `userFaceField<0>()` differs between modes,
   every user-field read on a conforming mesh is silently wrong. Symptom: garbage
   face user data, not a crash. Check this before anything else.
2. **Blue-diagonal tie-break.** A partition-dependent tie-break passes every test
   except `conforming_determinism`'s rank-count checksum. If that checksum differs
   across ranks 1–5, the tie-break is reading a partition-local quantity.
3. **Face gid collisions across rounds.** Retired parent gids are reused by un-close
   while children are allocated above the global max. Surfaces as duplicate face gids
   in `checkOwnershipPartition` *only after* a `migrate()` brings two ranks together —
   so a multi-round test, not a single-round one.
4. **Non-local vertices referenced by closure children.** By design a closure child
   may name a vertex gid the rank does not hold until the halo is rebuilt. Anything
   that assumes post-refine locality (see the post-refine gotcha in
   `docs/design.md`) will abort or read garbage.
5. **Sibling co-residency after `migrate()`/Zoltan2.** If the `dest` fixup missed a
   case, two ranks each restore the same parent → duplicate face, caught by
   `checkOwnershipPartition`.
6. **HIP-specific.** The closure derivation is host-side, but the face tuple grew by
   two members: check the `deep_copy` of the wider face AoSoA and any place a face
   tuple size or member index was assumed. A SERIAL pass with a HIP failure points
   here.
7. **I/O field-set mismatch.** The writer emitting the two closure fields and the
   reader not expecting them (or vice versa) fails as a round-trip checksum
   mismatch, not a read error.
8. **Generation-guard over-firing.** Un-close bumps `generation()` an extra time; a
   handle the tests expect to survive may now abort. Distinguish a real stale-handle
   bug from an over-eager bump.

**Rules while fixing.**

- **A failing test is never silently excluded from the gate.** If something cannot be
  made green in this task, relabel it `unit`, record it in README *Known Issues* with
  how it reproduces and whether it predates this work, list it under *Open failures*
  below, and report it. Never delete a test or weaken an assertion to get green.
- **`conforming_quality` bound calibration.** Task 7 registered it as `unit` with
  provisional bounds. Replace them with values justified by the measured per-round
  data, then promote it to `regression` at ranks 1–5 *only if* it is stable across
  repeated runs. If the measured quality is genuinely unbounded with round count, that
  is a **design** finding, not a tolerance to loosen — stop, record it, and report it.
- **Escape hatch on the default flip.** Task 7 flipped the `Mesh` default to
  `Conforming` before anything had been executed. If conforming cannot be made green
  in this task, revert the default to `HangingNode2to1` (keeping conforming opt-in and
  fully built), record why in README *Known Issues* and under *Open failures*, and
  report it. Reverting the default is preferable to leaving a broken default.
- If a fix changes a **design** decision, rewrite the affected design section of this
  file — do not leave the design describing something the code no longer does.
- Recompute and record the **new expected test totals** (they were 70/70 regression
  and 28/28 unit before this work) in the *Status* section here and in the CLAUDE.md
  task-log row.

**Acceptance.** `flux batch scripts/tuolumne/run_regression_minset.flux` fully green
at the gate definition (label `regression` × {SERIAL, HIP} × ranks 1–5), `ctest -L
unit` fully green, `format-check` clean, and the *Open failures* list empty.

**Open failures.** *(none recorded yet — Task 8 has not run)*

**Report back.** The full failure list as first observed and the root cause of each;
which of the eight risk points above actually fired; the new regression/unit totals;
any test relabelled to `unit` and why; whether the `Conforming` default survived; any
design section this task had to rewrite.

**Plus the measurements Tasks 2–7 deferred to here** — the tests print all of these,
so collect them from the run output rather than re-running:

| Deferred from | Measurement |
|---|---|
| Task 2 | Whether the "red child of a refined face has `|S| = 0`" assert in `closeFaces()` ever fired; and the `refine_closure` printout (per-`\|S\|` histogram, closure-face count, both blue-diagonal counts, conforming vs hanging-node Euler / bad-incidence / T-junction figures) |
| Task 3 | Actual message-volume increase on the extended Phase-2 rounds vs the estimated `1/ρ`; and the `refine_splitedges` printout (split-edge count, kept-side-discovered count per rank count, phase-2a total vs refining-only per round) |
| Task 4 | Closure-face fraction vs mask fraction, and the `|S|`-pattern histogram, per rank count |
| Task 5 | How many `dest` entries the sibling-cohesion fixup moved; pre/post `loadBalance` imbalance |
| Task 6 | Actual on-disk size delta for a conforming vs hanging-node file |
| Task 7 | Closure-vertex count per rank count (non-vacuity); closure-vertex vs interior-vertex `applyStencil` max error; per-round min angle / max aspect ratio / closure-face fraction over ≥8 rounds |

---

## Decision record

*(append-only)*

- **2026-07-31 — Decision 1: option, not replacement.** Conforming refinement is an
  additional mesh option; the 2:1 hanging-node path is retained. It is additive —
  conforming refinement is a closure pass over the 2:1-balanced red mesh, so the
  hanging-node path is a prerequisite, not a competitor.
- **2026-07-31 — Decision 2: compile-time template parameter.** `RefinementMode` is a
  `Mesh` template parameter (appended last, source-compatible), not a runtime flag,
  so the closure bookkeeping fields exist only in `Conforming` mode.
- **2026-07-31 — Decision 3: `Conforming` becomes the default**, hanging-node
  opt-in. Sequenced last (Task 7) so the flip lands only after conforming is
  gate-green at ranks 1–5 over multiple adaptive rounds.
- **2026-07-31 — Decision 5: supersedes the sequencing clause of Decision 3.** Since
  Tasks 1–7 run no tests, Task 7's default flip lands *before* conforming has ever
  been executed — not after it is gate-green, as Decision 3 stated. Task 8 verifies
  both modes together, and reverting the default is Task 8's escape hatch. The choice
  of `Conforming` as the default (Decision 3 proper) is unchanged.
- **2026-07-31 — Decision 4: transient red–green–blue closure**, not
  newest-vertex bisection and not red-only propagation. Red-only propagation
  degenerates to uniform refinement; bisection replaces the red engine wholesale
  (recorded as a Milestone-2 alternative).

## Progress log

*(append-only)*

- 2026-07-31 — Design scoped; Decisions 1–4 recorded. No code yet. Next: Task 1.
- 2026-07-31 — Task 7 expanded: it now owns a dedicated conforming test suite
  (`conforming_operators`, `conforming_determinism`, `conforming_quality`, plus a
  `staleslice_guard` case) alongside the mode-parity registration and the default
  flip, split across three commit checkpoints.
- 2026-07-31 — Added Task 8 (run the full suite, fix everything). Tasks 1–7 are now
  explicitly **no-test** tasks that end at a commit + push; their Acceptance sections
  specify tests to write, not to run. Task 8 is the single verification point.
- 2026-07-31 — **Task 1 landed.** `RefinementMode` enum + `Mesh`'s 8th template
  parameter, conditional `ClosureParent`/`ClosureParentVerts` face members appended
  after the user pack, index/count helpers, `if constexpr` dispatch with an
  aborting `Conforming` stub, `test_refinement_mode`, docs. **Not compiled:** the
  build was skipped at the user's request this session, so Task 2 should expect to
  absorb any Task-1 build break — build before adding to it.
  `clang-format` (v21, `/usr/bin/clang-format` on Tuolumne) is clean on every file
  Task 1 touched, but note that several *untouched* files (e.g.
  `src/Tessera_Geometry.hpp`) already violate v21, so a wholesale `format-check`
  under that version is not clean and predates this work.
  Also, running a bare `cmake .` in `build-tuolumne` without the spack env active
  deleted its `CMakeCache.txt`; that directory needs
  `bash ../run_cmake_toulumne.sh` (with `spack env activate
  ~/spack_envs/tuolumne_trilinos/`) re-run before the next `make`.
- 2026-07-31 — **Task 2 landed.** `src/Tessera_RefineClosure.hpp` (`RedFace`,
  `VisibleFace`, `ClosureStats`, `closeFaces`, `unclose`, `translateMask`,
  `countClosureChildren`, `faceSplitEdges`, `readVisibleFaces`, `writeClosureFace`,
  `initClosureFaceMembers`), `detail::refineLocalConforming()` replacing
  `refineLocal()`'s `Conforming` abort stub, `initClosureFaceMembers()` wired into
  `buildTriangleMesh()`, and `tests/test_refine_closure.cpp` registered `unit`
  SERIAL + HIP np1. **Task 1's build break risk is discharged:** the Task-1 code
  compiled clean with no changes needed. Note for later sessions — the Bash tool
  runs a *non-login* shell, where no `PrgEnv-*` module is loaded and the Cray `CC`
  wrapper fails at configure with "A PrgEnv-* modulefile must be loaded"; run the
  build under `bash -lc` (and export
  `SPACK_USER_CONFIG_PATH=$HOME/.spack/tuolumne`). `clang-format` on this checkout
  resolves to `/opt/rocm-6.4.2/llvm/bin/clang-format` (v19), under which every file
  Task 2 touched is clean; pre-existing violations in untouched files (e.g.
  `src/Tessera_Geometry.hpp`) remain and predate this work.
- 2026-07-31 — **Task 3 landed.** `detail::OwnerMsg` gained a `refining` flag and
  Phase 2a now advertises every owned face's edges; the coordinator aggregates
  `{participants, refOwner, anyRefining}`, drops non-split edges, and replies to all
  participants, so 2c's gid delivery reaches kept co-sharers.
  `RefineResult::midpoints` is now the complete split-edge map (contract widened and
  documented), plus new `phase2Adverts` / `phase2AdvertsRefining` counters.
  `tests/test_refine_splitedges.cpp` registered `regression` SERIAL + HIP ranks 1–5.
  `docs/design.md`'s Phase-2 paragraph rewritten, and its stale "the `Conforming`
  branch is a stub that aborts" line corrected — that is now true of the
  *distributed* `refine()` only, since Task 2 implemented `refineLocal()`.
  Midpoint **assignment** is provably unchanged (ownership still reads refining
  participants only, and the split-edge set is unchanged), so no existing test's
  expected gids move. Build note: `spack env activate` needs
  `. /usr/WS2/stewartj/spack/share/spack/setup-env.sh` sourced first even under
  `bash -lc`. `clang-format` is clean on both touched files under **both** the v21
  (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`) binaries on this machine.
- 2026-07-31 — Audited every task's Acceptance / Report-back for consistency with the
  no-test rule: dropped the "gate green" claims from Tasks 2–7, moved all
  runtime-measurement report-back items into a deferred-measurements table in Task 8,
  required the tests to *print* those measurements, made `conforming_quality` land as
  `unit` with provisional bounds for Task 8 to calibrate, and added the default-flip
  escape hatch (Decision 5).
