# Caller-driven edge split

**Status:** NOT STARTED. First of the four topological-edit tasks
([edge-split](edge-split.md) → [mesh-compaction](mesh-compaction.md) →
[edge-flip](edge-flip.md) → [edge-collapse](edge-collapse.md)). **Read the
"Editing families" section below before starting any of the four** — it is stated
once, here, and cross-referenced from the other three.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

Tessera is **split-based only**, and even the split is not addressable by the
caller. `refine()` takes a **face mask** and bisects all three edges of every
marked face; the split-edge map (`RefineResult::midpoints`) is an **output**, not
an input.

There is no way to say "bisect these particular edges". The closest expressible
thing — mark every face incident on a wanted edge — is strictly wrong for the
purpose: it splits all three of that face's edges, not the one wanted, and then
pulls in the 2:1 balance closure on top. The resulting edit is larger than
requested and differently shaped.

**Why it matters.** Metric-driven remeshing selects edges, not faces: an edge
longer than the local target length is split, an edge shorter than a floor is
collapsed, and a flip is applied where it improves valence. The whole family is
edge-addressed. The driving consumer (the Beatnik z-model's dynamic remesher) is
built out of exactly `split_selected_edges` / `collapse_short_edges` /
`flip_edges_for_quality`, and none of the three is expressible today.

## Approach

### Editing families — the decision that scopes all four tasks

Tessera has one topology editor today, `refine()`, and its whole design rests on a
**level model**: `FaceField::Level` and `EdgeField::Level` only ever rise, the 2:1
balance invariant is stated in terms of level differences, and
`RefinementMode::Conforming` maintains a transient closure layer keyed off
`ClosureParent`. That model is coherent because `refine()` only ever performs the
uniform 1→4 red split.

An edge-addressed split does **not** fit that model. Bisecting one edge of a
triangle produces two children whose edges have mixed levels; there is no single
integer that describes them, and a 2:1 level-difference invariant is not the right
statement about the resulting mesh.

**Decision 1 — two disjoint editing families, and a mesh belongs to one.**

| Family | Operations | Invariant maintained | Level semantics |
| --- | --- | --- | --- |
| **Hierarchical** | `refine()` | 2:1 level balance; conforming closure | `Level` is authoritative |
| **Remesh** | `splitEdges()`, `collapseEdges()`, `flipEdges()`, `compact()` | conformity and manifoldness only | `Level` is advisory |

A `splitEdges()` child face **inherits its parent's level** and the mesh is
thereafter not 2:1-level-meaningful. Interleaving `refine()` with any remesh
operation on the same mesh is **unsupported**. Enforce it rather than documenting
it: add a small `EditFamily` tag to the mesh (`None` initially, set on first edit),
and have each entry point throw a message naming both families when the tag
disagrees. A one-line check that turns a subtle wrong answer into an immediate
abort.

The alternative — extending the level model to anisotropic bisection — is a much
larger design (per-edge levels with a compatible balance rule) and is not needed
by any known consumer. Record it as future work; do not attempt it here.

**Decision 2 — an edge-mask split needs no closure and no mark propagation.**
This is the pleasant surprise and it should be stated prominently, because it is
counter-intuitive next to `refine()`. Bisecting a set of edges and splitting
**every** incident face according to how many of its edges were bisected yields a
**conforming** mesh directly: a face with 1, 2 or 3 bisected edges becomes 2, 3 or
4 children respectively, and no hanging node survives. The 2:1 closure machinery
exists because `refine()` bisects all three edges of a marked face and leaves the
neighbour untouched. `splitEdges()` has no such asymmetry. So Phase 1 of
`refineImpl` (the mark-propagation fixpoint) and the closure pass have **no
analogue here** — do not port them.

### API

New header `src/Tessera_EdgeSplit.hpp`.

```cpp
struct SplitResult
{
    //! (bisected edge, midpoint gid) for every split edge this rank TOUCHES —
    //! same contract and same shape as RefineResult::midpoints, so
    //! checkMidpointAgreement consumes it unchanged. Sorted, unique by EdgeKey.
    std::vector<std::pair<EdgeKey, GlobalId>> midpoints;
    long long requested = 0;   //!< marked owned edges, summed globally
    long long split = 0;       //!< edges actually bisected, globally
    long long facesBefore = 0, facesAfter = 0;
};

//! Bisect exactly the marked edges. `edgeMask.size() == mesh.numOwnedEdges()`;
//! the OWNER of an edge decides, and the decision is propagated to every rank
//! holding an incident face. Every incident face is subdivided into 2, 3 or 4
//! children according to how many of its edges are bisected. Conforming on exit
//! with no closure layer and no 2:1 balance pass.
//!
//! Ends by calling rebuildHalo(), so the halo is valid on return — the same
//! postcondition refine() has had since `a55a8de`. Collective.
template <class MeshT, class Policy = DefaultRefinePolicy>
SplitResult splitEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                        const std::vector<char>& edgeMask,
                        const Policy& policy = Policy{} );
```

Host `std::vector<char>` sized `numOwnedEdges()`, matching `refine()`'s mask
convention — a device-computed indicator round-trips to the host. Keep the
convention rather than inventing a second one.

### Implementation

Reuse `refine()`'s Phase 2 wholesale; that is most of the work already done.

1. **Cross-rank agreement on the mask.** An edge is owned by one rank but incident
   on faces owned by up to two. Advertise, per owned face, its three
   `(EdgeKey, faceGid, ownerRank)` to `detail::edgeCoordRank`. The coordinator
   learns the split verdict from the edge's **owner** and replies to every
   co-sharer. Identical routing to `refineImpl` Phase 2a, with `refining` replaced
   by the owner's mask bit.
2. **Midpoint gid assignment.** Unchanged from `refineImpl` Phase 2: the midpoint
   owner is the lowest incident face owner; owners count their midpoints,
   `MPI_Exscan` a contiguous global block onto the pre-split global vertex count,
   assign, and **send the gid to all co-sharers** so both sides agree without
   relying on identical local ordering. This is what makes `midpoints` a complete
   split-edge map and what makes `checkMidpointAgreement` applicable.
3. **Midpoint position and field transfer.** `RefinePolicy` unchanged —
   `DefaultRefinePolicy`'s linear average is the correct rule and applies to every
   user field automatically. **Do not project midpoints onto a sphere**; Tessera
   is not a sphere library and `refine()` does not either.
4. **Local topology reconstruction.** New code, and the only genuinely new logic.
   Per local face, dispatch on the bit pattern of its three bisected edges:
   - **1 bisected edge** → 2 children. Split along the median from the midpoint to
     the opposite corner. One new interior edge.
   - **2 bisected edges** → 3 children. Two diagonals are possible; **pick the
     shorter**, tie-broken by the smaller `EdgeKey`. Both operands are locally
     available (the face's own corners and the two midpoints), so the tie-break is
     evaluable locally and is rank-count invariant.
     **Use `Tessera::edgeLen2Canonical()` (`Tessera_RefineClosure.hpp:212`) for the
     comparison, not a hand-rolled squared length.** The conforming blue closure
     chooses its diagonal geometrically through that exact helper (Decision 15,
     `4cee602`), and the helper exists precisely so two ranks comparing the same
     edge get bit-identical doubles regardless of endpoint order. Reusing it means
     this task inherits that determinism instead of re-deriving it, and it keeps
     the two diagonal rules in the library consistent with each other. Follow the
     same exact-tie fallback the closure uses, and report the tie count as
     `ClosureStats` does.
   - **3 bisected edges** → 4 children, the red split. Identical to `refine()`'s
     existing 1→4 case; reuse it.
   Child face gids: `MPI_Exscan` a contiguous block per rank onto the pre-split
   global face count, as `refine()` does. New interior edge gids likewise, via the
   edge coordinator so the two sides of a shared new edge agree (a new interior
   edge is never shared — it lies strictly inside one parent face — so it can be
   assigned locally; the *halves* of a bisected edge **are** shared and must go
   through the coordinator).
5. **Child levels.** Parent's level, per Decision 1. Set the `Conforming`
   closure members to "visible, no parent" via `initClosureFaceMembers` so a
   zero-filled `ClosureParent` cannot read as face gid 0 — the same trap
   `distribute()` already documents.
6. **Halo.** Call `rebuildHalo()` (`Tessera_HaloRebuild.hpp`) before returning,
   preserving `halo.depth` if [halo-depth.md](halo-depth.md) has landed. Never
   return with a cleared halo — `refine()` stopped doing that in `a55a8de` and
   `splitEdges` must not reintroduce the trap.
7. **Edge user fields.** Edges are re-derived from the new face connectivity here,
   exactly as in `refine()`, so per-edge user data cannot be carried through.
   README *Known Issues* already records this for `refine()`/`refineLocal()`
   ("Edge user fields are reset by `refine()`/`refineLocal()`"); **extend that
   entry to name `splitEdges` rather than adding a second one.** Vertex and face
   user fields are preserved and interpolated as usual.
8. **Empty mask** is a no-op fast path: no communication beyond the collective
   that establishes the global request count, and V/E/F unchanged.

### Non-goals

- Anisotropic level bookkeeping (Decision 1's alternative).
- Choosing *which* edges to split. That is the consumer's metric.
- Collapse, flip, compaction — the three sibling tasks.

### Tests

New `tests/test_split_edges.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

Every count assertion below is a closed-form identity, so the test checks
arithmetic rather than checking Tessera against itself. All are asserted via the
global owned-count reductions and hold at every rank count.

1. **One edge.** Mark the single edge with the globally smallest `EdgeKey` (a
   rank-count-invariant choice). Deltas: `V+1`, `E+3`, `F+2`. Euler
   `1 - 3 + 2 = 0`, so `V-E+F == 2` still. `checkConforming`,
   `checkNoInteriorVertex`, `checkOwnedEuler == 2`.
2. **Three edges of one face.** Deltas: `V+3`, `E+9`, `F+6` (that face → 4, its
   three neighbours → 2 each). Euler delta 0.
3. **All edges — equivalence with `refine()`.** Mark every owned edge. Result must
   have `V' = V+E = 642`, `E' = 2E+3F = 1920`, `F' = 4F = 1280` — **the same
   counts as a uniform `refine()`** on a fresh mesh. Stronger: the two meshes'
   **vertex position multisets** (sorted lexicographically, so gid numbering is
   irrelevant) must be **bitwise identical**, and so must the multiset of face
   corner-position triples. This is the sharpest available correctness check on
   the 3-edge path, because it pins the new code against machinery already
   verified 150/150.
4. **Two-edge pattern determinism.** Mark edges by a rank-count-invariant rule
   (`EdgeKey.id[0]` even) so faces with all of 1, 2 and 3 bisected edges occur.
   Assert the resulting V/E/F and the face corner-position triple multiset are
   **identical at ranks 1–5**. This is the check that the two-diagonal tie-break
   really is local and rank-count invariant.
5. **Midpoint agreement.** `checkMidpointAgreement( result.midpoints )` on every
   case above. Plus: an edge appears in the map iff some rank touching it reports
   it, and no rank reports an edge it does not touch — reuse the globally-decided
   ground-truth pattern from `test_refine_splitedges.cpp`.
6. **Midpoint positions.** Each midpoint equals the exact average of its endpoints
   (bitwise for the default policy, since `(a+b)/2` is one operation), and is
   **not** on the unit sphere — asserting the negative pins Decision 3.
7. **User-field transfer.** A vertex user field seeded to a linear function of
   position is reproduced at every midpoint to `1e-15` relative, for both a
   `double` and a `double[3]` field.
8. **Repeated rounds.** Five successive `splitEdges` calls with a
   length-threshold mask (split any edge above the current mean length), no
   intervening `migrate()`. Euler `== 2` and `checkConforming` after each round;
   minimum triangle radius ratio reported per round and asserted above a floor
   measured in the first implementation run rather than guessed — record the
   measured value here.
9. **Empty mask.** No-op: V/E/F and `topologyChecksum` unchanged.
10. **Halo valid on return.** Immediately after `splitEdges`, `haloExchange()`
    leaves every ghost vertex position equal to its owner's, and a **second**
    `splitEdges` with no intervening `migrate()` succeeds. Confirms step 6.
11. **Family guard.** `refine()` on a mesh already edited by `splitEdges` throws,
    naming both families; and `splitEdges` on a `refine()`d mesh throws. Pins
    Decision 1.

## Exit criterion

- `test_split_edges` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled.
- Check 3 passes **bitwise** — the all-edges split and a uniform `refine()`
  produce the same geometry.
- Check 4 passes: the edit is rank-count invariant, including the two-edge
  diagonal choice.
- README gains `splitEdges` in the API section and an **Editing families**
  subsection carrying Decision 1's table; the family guard's error message is
  quoted there.
- `docs/design.md` gains an *Edge-addressed splitting* subsection recording
  Decision 2 (no closure, no mark propagation, and why) and the three bit-pattern
  cases with the two-edge tie-break rule.

## Where this sits

**No hard prerequisite.** `rebuildHalo()` already exists in tree
(`Tessera_HaloRebuild.hpp`, `25980f2`), so step 6 is available today —
[halo-depth.md](halo-depth.md) is only needed if a caller wants a halo deeper than
1, which `splitEdges` itself does not. Prerequisite in spirit for the other three
remesh operations, since Decision 1 and the coordinator idiom are established here.
See the ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
