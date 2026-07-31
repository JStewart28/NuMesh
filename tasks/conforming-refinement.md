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
| 1 | Refinement-mode plumbing (`RefinementMode`, conditional face fields, dispatch) | Not started |
| 2 | Serial closure kernel: `closeFaces` / `unclose` + patterns | Not started |
| 3 | Distributed split-edge discovery (extend Phase 2 to kept faces) | Not started |
| 4 | Wire closure into distributed `refine()` | Not started |
| 5 | `migrate()` / `loadBalance()` / halo rebuild on a closed mesh | Not started |
| 6 | I/O round-trip, `markByQuality`, example + docs | Not started |
| 7 | Dedicated conforming test suite; flip the default to `Conforming` | Not started |
| 8 | **Run the full suite and fix everything it finds** (the only task that runs tests) | Not started |

Nothing is implemented yet. Task 1 is next.

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

### Data model — closure bookkeeping

Un-closing must reconstruct the red parent from its children. Each closure child
stores the parent outright, so **any single child determines its parent** with no
sibling lookup:

| Field | Type | Meaning |
|---|---|---|
| `ClosureParent` | `GlobalId` | Parent red face's gid; `invalid_gid` for a red face (no closure). |
| `ClosureParentVerts` | `GlobalId[3]` | Parent's three corner vertex gids, in the parent's winding order. |

These are added to `CoreFaceMembers` **only** when `Mode == Conforming`, via a
conditional member-type alias in `Tessera_Fields.hpp`.

> **API wrinkle — put the closure fields *after* the user pack.** `FaceField::UserBegin`
> is the hard-coded constant `5`, and `userFaceField<M>()` is `UserBegin + M`.
> Inserting closure fields among the core members would shift every user field index
> and silently break existing consumers. Instead append them after the user pack, so
> `UserBegin` stays `5` and the closure indices are computed from the user pack size:
> `closureParentField<FaceUserFields>()` etc. Task 1 must add a compile-time test
> that `userFaceField<0>()` is identical in both modes.

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

**Phase 2 extension — kept faces must learn their split edges.** Today Phase 2
advertises only *refining* faces' edges to the edge coordinator, and only refining
participants receive the midpoint gid. A kept face on rank R needs to know which of
its edges were bisected by a refining neighbour on rank R′. Extend the existing
rounds — **no new communication rounds**:

- **2a:** advertise the edges of *all* owned faces, refining and kept, carrying a
  `refining` flag.
- **2b:** the coordinator computes the midpoint owner from the *refining*
  participants only (unchanged rule: lowest incident refining-face owner), but
  replies `(key, midOwner)` to **all** participants of a split edge.
- **2c:** the midpoint owner sends the gid to **all** co-sharers, refining or kept.

Every rank then holds `EdgeKey → midpoint gid` for every split edge it touches, which
is exactly the split-edge map step 3b needs.

**A closure child may reference a non-local vertex.** The midpoint gid arrives, the
midpoint *position* does not. This is not new: `refine()` already leaves an
owned-only mesh whose faces reference vertex gids the rank does not hold (see the
post-refine gotcha in `docs/design.md`). The halo rebuild inside `migrate()` brings
them in. Do **not** add a position gather to the closure.

**Face gid allocation.** Closure children are new faces and need new gids. Fold them
into the existing single `MPI_Exscan`: count `4 * nRefining + Σ closureChildren` per
rank and allocate one contiguous block above the global max face gid. A retired
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

### Task 1 — Refinement-mode plumbing

**Goal.** The option exists and is inert. No behavior change.

- Add `enum class RefinementMode { HangingNode2to1, Conforming }` (new header
  `src/Tessera_RefinementMode.hpp`, folded into `<Tessera.hpp>`).
- Add the `RefinementMode Mode = RefinementMode::HangingNode2to1` template parameter
  to `Mesh`, last in the list; expose
  `static constexpr RefinementMode refinement_mode = Mode;`.
  *(Default flips to `Conforming` in Task 7, once the conforming machinery and its
  tests all exist; Task 8 then verifies both modes.)*
- In `Tessera_Fields.hpp`, add the conditional closure members
  (`ClosureParent`, `ClosureParentVerts`) **after** the user pack, plus
  `closureParentField<UserFaceFields>()` / `closureParentVertsField<...>()` index
  helpers. `FaceField::UserBegin` must not move.
- `refine()` / `refineLocal()` dispatch on `MeshT::refinement_mode` with
  `if constexpr`; the `Conforming` branch is a `static_assert`-free runtime
  `Kokkos::abort("not implemented")` stub for now.
- Update `docs/design.md` (refinement section) and `README.md` (API + Known Issues:
  note conforming is in progress).

**Acceptance.**
New `unit` test `refinement_mode` (SERIAL + HIP, np1): a `Conforming`-typed mesh
compiles; `userFaceField<0>()` is identical in both modes; a `Conforming` face tuple
is larger than a `HangingNode2to1` one; `format-check` clean.

**Report back.** The final conditional-member-type spelling and any Cabana template
friction; whether appending the template parameter broke any call site.

---

### Task 2 — Serial closure kernel

**Goal.** The closure and its inverse exist as pure, testable local functions.

- New header `src/Tessera_RefineClosure.hpp`:
  - `closeFaces(...)` — given a red face list (corner gids, gids, levels) and a
    `EdgeKey → midpoint gid` map, produce the visible face list plus each child's
    `ClosureParent` / `ClosureParentVerts`. Implements the green / blue / red-closure
    patterns above, with the lower-gid blue diagonal and winding preserved.
  - `unclose(...)` — inverse: group by `ClosureParent`, restore one red face per
    parent (gid, corner gids, level, user fields from the lowest-gid child).
  - `translateMask(...)` — visible-face mask → red-face mask (OR over children).
- Add a `RefinementMode::Conforming` path to `refineLocal()` that runs
  un-close → red split → close on one rank.

**Acceptance.** New `unit` test `refine_closure` (SERIAL + HIP, np1):
hand-built parent + each of the four `|S|` cases with expected child triangles;
`unclose ∘ close` identity on a random partial mask over `buildIcosphere(3)`;
after `refineLocal` with a **partial** mask — Euler `V−E+F == 2`, every edge exactly
2 incident faces, no interior vertex, winding consistent (all `faceNormalRaw`
outward), vertex count identical to the hanging-node mode's (closure adds no
vertices).

**Report back.** The blue-diagonal tie-break as implemented, and why it is
partition-independent. Where you placed the assert that a red child of a refined face
can never have `|S| > 0`, and the argument for why that holds by construction —
whether it actually fires is a Task 8 observation.

---

### Task 3 — Distributed split-edge discovery

**Goal.** Every rank learns the midpoint gid of every split edge it touches,
including on kept faces. No closure yet.

- Extend Phase 2 of `refine()` per *Phase 2 extension* above: advertise all owned
  faces' edges with a `refining` flag; coordinator replies to all participants;
  owner ships the gid to all co-sharers. Midpoint **ownership** rule unchanged
  (lowest incident *refining*-face owner) — verify `checkMidpointAgreement` still
  passes.
- Return the split-edge map from `refine()` (extend `RefineResult`, or an internal
  handoff to Task 4 — the map is already recorded in `RefineResult::midpoints`, so
  prefer widening what that field contains and documenting it).

**Acceptance.** New `regression` test `refine_splitedges` (SERIAL + HIP, ranks 1–5):
for an adaptive mask, each rank's `(EdgeKey → midpoint gid)` set equals a
partition-free reference computed independently, at every rank count. The existing
`refine_parallel` and both `markquality` tests must be behaviorally unchanged — do
not edit them.

**Report back.** The analytic argument that the coordinator still receives exactly
two incident faces per edge now that kept faces also advertise (the
`inc.size() != 2` guard depends on it); your *estimated* message-volume increase on
the extended rounds and how you derived it — Task 8 measures the actual.

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

- Persist `ClosureParent` / `ClosureParentVerts` in the HDF5 writer/reader.
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
| Task 2 | Whether the "red child of a refined face has `|S| = 0`" assert ever fired |
| Task 3 | Actual message-volume increase on the extended Phase-2 rounds vs the estimate |
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
- 2026-07-31 — Audited every task's Acceptance / Report-back for consistency with the
  no-test rule: dropped the "gate green" claims from Tasks 2–7, moved all
  runtime-measurement report-back items into a deferred-measurements table in Task 8,
  required the tests to *print* those measurements, made `conforming_quality` land as
  `unit` with provisional bounds for Task 8 to calibrate, and added the default-flip
  escape hatch (Decision 5).
