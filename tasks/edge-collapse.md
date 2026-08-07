# Edge collapse

**Status:** NOT STARTED. Largest of the eleven gap tasks; do it **last** of the
four topological edits. **Read the "Editing families" section of
[edge-split.md](edge-split.md) first.**

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

Edge collapse does not exist at any level, and unlike flip it is not merely
unimplemented — **the data model has no coarsening path at all.**
`FaceField::Level` and `EdgeField::Level` only ever rise, `refine()` is split-only,
`RefinementMode::Conforming` tracks a `ClosureParent` for refinement children with
no inverse, and nothing anywhere removes an entity. So neither the link condition
nor a cross-rank owner-decides protocol has anywhere to attach.

**Why it matters.** Collapse is the only operation that removes degrees of freedom.
Without it, a remeshing scheme is monotone: a metric-driven remesher can refine
where the metric demands resolution but can never coarsen where it does not, so the
mesh grows without bound and a long run dies on memory or on timestep. It is also
half of the standard quality-repair pair — collapse removes short edges and
slivers that no amount of flipping fixes. The driving consumer (the Beatnik
z-model's dynamic remesher) is built out of `collapse_short_edges` alongside split
and flip, and its evolving vortex sheet both stretches and compresses, so
coarsening is not optional for it.

## Approach

New header `src/Tessera_EdgeCollapse.hpp`. Four hard problems, each with a decision
to make and record.

### What a collapse is, precisely

For an interior manifold edge `(a,b)` with incident faces `(a,b,c)` and `(b,a,d)`:
merge `a` and `b` into one vertex at the policy-chosen position, delete the two
incident faces, delete edge `(a,b)`, and merge the two pairs of edges that become
coincident. Net: **V−1, E−3, F−2**, so Euler is preserved. Every other face that
referenced `b` now references `a`.

### Problem 1 — where coarsening attaches to the data model

**Decision 1 — collapse is a remesh-family operation and requires a mesh with no
transient closure and no hanging nodes.** It is defined on a pristine mesh and on a
mesh edited by `splitEdges`/`flipEdges`; it is **not** defined on a mesh produced by
`refine()`, in either refinement mode. Under `Conforming` there is a transient
closure layer whose children reference retired red parents, and collapsing across
it would have to un-close first; under `HangingNode2to1` there are T-junctions,
and the link condition is not even well-posed at one. The existing `EditFamily`
guard from [edge-split.md](edge-split.md) Decision 1 enforces this with a clear
message. `Level` on the merged vertex's incident entities is set to the `min` of
the merged values and is **advisory** thereafter, per that same decision.

This is the decision that makes the task tractable at all. The alternative — a real
inverse to `refine()`, un-refining a red split back to its parent — is a different
and much larger feature (it needs the parent's identity retained, sibling
co-residency, and an inverse to the closure); it is **not** what a metric remesher
wants, because it can only coarsen along the refinement tree. Record it as future
work and do not attempt it here.

### Problem 2 — the link condition needs a two-ring

Collapsing `(a,b)` is topologically valid iff
`link(a) ∩ link(b) == {c, d}` — the two vertices opposite the edge and nothing
else. Violating it welds two distant parts of the surface together and produces a
non-manifold mesh that no later check will untangle.

Evaluating it needs the **full one-rings of both endpoints**, i.e. a two-ring
around the edge. **Requires [halo-depth.md](halo-depth.md) at `depth >= 2`.**
`collapseEdges` must therefore **check `mesh.haloDepth() >= 2` and throw** if not —
this is precisely the class of silent-seam bug the depth task exists to close, so
do not let it recur here.

### Problem 3 — conflicts, and why not shortest-first

Two collapses sharing a vertex conflict: applying one changes whether the other is
valid, and applying both can produce garbage. Serial reference codes process
**shortest-first**, re-evaluating validity after each acceptance. That is inherently
sequential and its result depends on the processing order.

**Decision 2 — a deterministic maximal independent set, one round per call.** Each
candidate edge gets a priority `(squared length ascending, EdgeKey ascending)` —
shortest first, exactly matching the serial *preference* — and an edge is accepted
only if it has the strictly best priority among all candidate edges incident on
either of its endpoints' one-rings. That set is pairwise non-conflicting, is
computable in one coordinator round plus one exchange, and is **independent of rank
count** because the priority is a total order on geometric quantities and keys.

The accepted set will **not** equal the serial shortest-first set. It is a subset of
it, reached in fewer passes. A caller wanting more progress calls again and watches
`accepted`, exactly as with [edge-flip.md](edge-flip.md) Decision 3. Consumers must
compare **statistics** — face count, quality distribution, edge-length histogram —
not the edit set. State this in the header, in the README, and in `docs/design.md`;
it is the single most likely source of a "why doesn't this match the serial code"
question.

### Problem 4 — cross-rank owner-decides

The one-rings of `a` and `b` may span several ranks even at depth 2, and the merged
vertex's identity must be agreed before any rank rewrites connectivity.

Protocol, built on the existing edge coordinator (`detail::edgeCoordRank`,
`allToAllV`), the same machinery `refine()`'s 2:1 balance uses:

1. **Advertise** per owned face, per edge: `(EdgeKey, faceGid, ownerRank,
   oppositeVertexGid, oppositeVertexPosition)` — identical to
   [edge-flip.md](edge-flip.md) step 1, so factor that pack/route into a shared
   `detail` helper rather than duplicating it.
2. **The edge owner assembles the candidate**: `a`, `b`, `c`, `d`, their positions,
   and the two one-rings (available locally at depth 2). It evaluates the link
   condition, the geometric tests, and computes the priority.
3. **Independent-set round.** Each *vertex* coordinator collects the candidates
   incident on its vertex, and replies to each with whether it holds the best
   priority. An edge is accepted iff both endpoints' coordinators say yes. One
   `allToAllV` round trip.
4. **Apply.** The surviving vertex is `min(gid(a), gid(b))` — deterministic and
   rank-independent — moved to `policy.position(a, b)`. Every rank holding a face
   or edge that references the dying gid rewrites it to the surviving gid; that set
   is exactly the ranks in the dying vertex's one-ring, reachable through the
   vertex coordinator. The two incident faces, the collapsed edge, and one of each
   coincident edge pair are **tombstoned** (per
   [mesh-compaction.md](mesh-compaction.md)).
5. **`compact()`**, then `rebuildHalo()` at the incoming depth. `collapseEdges`
   returns a compact, fully-haloed mesh — a caller must never see tombstones.

### API

```cpp
struct CollapseResult
{
    long long requested = 0, accepted = 0;
    long long rejectedBoundary = 0;
    long long rejectedLinkCondition = 0;
    long long rejectedNormalFlip = 0;
    long long rejectedQuality = 0;
    long long rejectedConflict = 0;
    long long verticesRemoved = 0, edgesRemoved = 0, facesRemoved = 0;
};

struct DefaultCollapsePolicy
{
    //! 0.5 = midpoint. 0.0 keeps a, 1.0 keeps b (a = lower gid).
    double t = 0.5;
    //! Reject if any surviving incident face's normal rotates more than this
    //! (radians) — the standard fold guard.
    double maxNormalRotation = 0.5;
    //! Reject if any surviving incident face's radius ratio falls below this.
    double minQuality = 0.05;
};

//! Collapse the marked edges. `edgeMask.size() == mesh.numOwnedEdges()`; the
//! OWNER decides. At most a maximal independent set is applied per call. Requires
//! mesh.haloDepth() >= 2 (link condition) and throws otherwise. Compacts and
//! rebuilds the halo before returning. Collective.
template <class MeshT, class Policy = DefaultCollapsePolicy>
CollapseResult collapseEdges( MeshT& mesh,
                              MeshHalo<typename MeshT::memory_space>& halo,
                              const std::vector<char>& edgeMask,
                              const Policy& policy = Policy{} );
```

User field values at the merged vertex come from the same `RefinePolicy`-style
blend used for midpoints, at parameter `t`. Reuse the existing policy plumbing so
every user field is handled automatically.

### Non-goals

- Un-refining a `refine()`d mesh (Decision 1's alternative).
- Vertex removal / valence-3 cleanup, half-edge collapse to a fixed endpoint beyond
  `policy.t`, or quadric error metrics. All are consumer policy.
- Looping to exhaustion (Decision 2).
- Boundary collapses. A boundary edge is rejected; a boundary-preserving collapse is
  a separate feature.

### Tests

New `tests/test_collapse_edges.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

1. **One edge, exact deltas.** Subdivision-2 icosphere at depth 2, mark the
   globally smallest `EdgeKey`. `V−1`, `E−3`, `F−2`; `globalOwnedEuler == 2`;
   `checkConforming`, `checkNoInteriorVertex`, `owned1RingLocal` all pass. The
   surviving vertex gid is `min(a,b)` and sits at the midpoint.
2. **Link condition is enforced.** Mark all three edges of one face. Collapsing two
   of them is topologically invalid; assert `accepted <= 1`, Euler `== 2`, and
   conformity. Then a deliberate positive: on a hand-built soup containing two
   triangles that share an edge *and* a third vertex (the classic link-condition
   violation), assert `rejectedLinkCondition == 1` and `accepted == 0`.
3. **Depth guard.** `collapseEdges` on a depth-1 mesh **throws**, naming the
   required and actual depth. Pins Problem 2.
4. **Independent set.** Mark every owned edge. Assert no two accepted edges share a
   vertex (gather the accepted set and check against the pre-collapse
   vertex→edge map), and that `verticesRemoved == accepted`,
   `edgesRemoved == 3 * accepted`, `facesRemoved == 2 * accepted`.
5. **Rank-count invariance — the strongest determinism claim in the task.** Check 4
   at ranks 1–5: identical `accepted`, identical resulting V/E/F, and an identical
   multiset of face corner-**position** triples. If this fails, the priority order
   is not total or the surviving-vertex rule leaked a local index.
6. **Geometric rejection.** `minQuality = 0.99` rejects everything: `accepted == 0`,
   `topologyChecksum` bitwise unchanged, and the mesh is not compacted into a
   different ordering (a no-op must be a genuine no-op).
7. **Normal-flip rejection.** Build a soup with a near-degenerate spike whose
   collapse folds a neighbour; assert `rejectedNormalFlip >= 1` and that no
   surviving face has a normal reversed relative to its pre-collapse orientation
   (dot product with the pre-collapse normal positive for every surviving face).
8. **Decimation to a floor.** Loop `collapseEdges` with an all-edges mask until
   `accepted == 0` or 20 rounds. After every round: Euler `== 2`, conformity,
   `owned1RingLocal`, and no duplicate edge keys. Report face count and minimum
   radius ratio per round. Assert the face count is strictly decreasing while
   `accepted > 0`, and record the terminal face count and quality floor **measured
   in the first implementation run** here rather than guessing them.
9. **Round trip with split.** `splitEdges(all edges)` (→ 1280 faces) then collapse
   loops until no progress. Assert Euler `== 2` throughout and that the final face
   count is within a factor of 2 of the original 320 — a coarse but real check that
   coarsening actually undoes refinement rather than jamming.
10. **User-field blend.** A vertex user field seeded to a linear function of
    position: after a collapse at `t = 0.5`, the merged vertex's value equals the
    linear function evaluated at its new position, to `1e-14` relative. Both
    `double` and `double[3]`.
11. **Halo and compaction on return.** No tombstones remain (`Gid != invalid_gid`
    for every local entity); `haloExchange()` leaves ghost positions equal to
    owners'; `checkOwnershipPartition` passes; `halo.depth == 2` still; a second
    `collapseEdges` with no intervening `migrate()` succeeds.
12. **Empty mask** is a no-op; all counters zero; `topologyChecksum` unchanged.
13. **Family guard.** `collapseEdges` on a `refine()`d mesh throws per
    [edge-split.md](edge-split.md) Decision 1, in both refinement modes.
14. **Composed remesh loop — the integration test.** Twenty rounds of
    { `splitEdges` above a length threshold → `flipEdges` for valence →
    `collapseEdges` below a length floor }. After every operation: Euler `== 2`,
    conformity, no duplicate keys, no tombstones. Report face count, edge-length
    histogram and minimum radius ratio per round, and assert the face count stays
    within a band and the quality floor holds. This is the test that proves the
    four remesh operations compose, which is the actual deliverable of the family.

## Exit criterion

- `test_collapse_edges` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled.
- Checks 2, 3 and 7 pass — the three ways a collapse silently corrupts a mesh are
  each detected and reported through a named counter.
- Check 5 passes: the accepted set is rank-count invariant.
- Check 14 passes: split, flip and collapse compose over 20 rounds without
  degrading Euler, conformity or the quality floor.
- README API section documents `collapseEdges`, `DefaultCollapsePolicy`, the
  `haloDepth() >= 2` precondition, and **Decision 2 with its consequence** (the
  independent set differs from a serial shortest-first pass; compare statistics,
  not edit sets).
- README *Known Issues* records the accepted limits: collapse is undefined on a
  `refine()`d mesh, boundary edges are always rejected, and one call applies at most
  an independent set.
- `docs/design.md` gains an *Edge collapse* subsection with all four problems and
  their decisions, and states Decision 1's alternative (a true inverse to `refine()`)
  as explicitly out of scope.

## Where this sits

**Requires [halo-depth.md](halo-depth.md) — this is the one task with a hard
dependency on it**, for `depth >= 2` (the link condition, Problem 2);
`rebuildHalo()` itself already exists in tree (`Tessera_HaloRebuild.hpp`,
`25980f2`). Also requires [mesh-compaction.md](mesh-compaction.md) (`compact()`)
and [edge-split.md](edge-split.md) (Editing families). Shares its advertisement
pack/route helper with [edge-flip.md](edge-flip.md) — land that first so the helper
exists. See the ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
