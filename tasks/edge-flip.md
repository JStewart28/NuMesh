# Edge flip

**Status:** NOT STARTED. **Read the "Editing families" section of
[edge-split.md](edge-split.md) first** — a flip is a remesh-family operation.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

Edge flipping does not exist at any level. It is not merely a missing convenience:
a flip needs **both faces incident on the edge**, and the 1-deep *vertex* halo does
not guarantee they are co-resident (an owned face all three of whose corners are
ghosts can have edge-neighbours held on neither this rank nor any single other
rank's local set). So a correct distributed flip needs the same edge-coordinator
machinery `refine()`'s 2:1 balance uses, not a local rewrite.

**Why it matters.** Flipping is the cheapest of the three remeshing operations and
the one that does the most for element quality per unit cost: it changes
connectivity without moving a vertex or changing V, E or F. Valence equalization
(drive interior valences toward 6) and Delaunay-style quality repair are both
flip-only passes. The driving consumer (the Beatnik z-model's mesh-quality
maintenance) uses flips to keep a tightening roll-up from producing slivers, and
without them a long run dies on element quality rather than on physics.

## Approach

New header `src/Tessera_EdgeFlip.hpp`.

### What a flip is, precisely

For an interior manifold edge `(a,b)` with incident faces `(a,b,c)` and `(b,a,d)`:
delete edge `(a,b)`, create edge `(c,d)`, and replace the two faces with `(a,c,d)`
and `(b,d,c)`. **V, E and F are all unchanged.** Only connectivity changes, plus:

- the flipped edge's `EdgeField::Verts` changes, so its **`EdgeKey` changes** — and
  therefore its `detail::edgeCoordRank` routing changes, and the `edgeKeys()` side
  table must be rebuilt. Easy to miss; it is the bookkeeping trap of this task.
- the two faces' `Verts` and `Edges` change, so `faceKeys()` must be rebuilt.
- the vertex→edge and vertex→face CSRs change for all four of `a`, `b`, `c`, `d`.

**Decision 1 — gids are preserved.** The flipped edge keeps its gid (with a new
key) and the two faces keep theirs (with new corners). Nothing is created or
destroyed, so there is nothing to assign, and preserving gids means no peer's
reference is invalidated. Consequence: a gid no longer determines a key, so any
code assuming `gid ↔ key` is a bijection breaks. Audit for that assumption as part
of the task.

### API

```cpp
struct FlipResult
{
    long long requested = 0;             //!< marked owned edges, globally
    long long accepted = 0;
    long long rejectedBoundary = 0;      //!< fewer than two incident faces
    long long rejectedDuplicateEdge = 0; //!< (c,d) already exists
    long long rejectedGeometric = 0;     //!< policy test failed
    long long rejectedConflict = 0;      //!< lost the independent-set round
};

struct DefaultFlipPolicy
{
    //! Reject if either new face's normal points more than this far from the
    //! area-weighted average of the two old normals (radians). Guards against
    //! flipping a nearly-flat pair into a fold.
    double maxNormalDeviation = 0.35;
    //! Reject if either new face's radius ratio (inradius/circumradius, scaled to
    //! 1 for equilateral) falls below this.
    double minQuality = 0.05;
};

//! Flip the marked edges. `edgeMask.size() == mesh.numOwnedEdges()`; the OWNER of
//! an edge decides. At most an independent set is applied per call — two flips
//! sharing a FACE conflict and only one survives, so a caller wanting more
//! progress calls again. Collective. Ends with rebuildHalo().
template <class MeshT, class Policy = DefaultFlipPolicy>
FlipResult flipEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                      const std::vector<char>& edgeMask,
                      const Policy& policy = Policy{} );
```

### Decision 2 — valence selection is the caller's, not Tessera's

The dominant use of flipping is valence equalization, which needs the **full
valence** of `a`, `b`, `c` and `d`. A vertex's valence is a local quantity at its
owner (the vertex→edge CSR owned rows are complete at halo depth 1), so the caller
can compute it and encode it in the mask. Keeping it out of `flipEdges` means the
operation itself needs **only depth 1** — the topological and geometric tests below
are all evaluable from the two incident faces, which the coordinator supplies.

Pulling valence inside would force `flipEdges` to need depth 2 and would hard-code
one selection criterion into a general operation. Document the intended pattern
(caller computes valences from `mesh.vertexEdges()`, builds the mask, calls
`flipEdges`, repeats) so consumers do not each reinvent it.

### Implementation — owner decides, both face owners apply

1. **Advertise.** Each rank sends, per **owned** face, `(EdgeKey, faceGid,
   ownerRank, oppositeVertexGid, oppositeVertexPosition)` for each of its three
   edges to `detail::edgeCoordRank`. The opposite vertex is the face corner not on
   the edge — `c` for one face, `d` for the other. Carrying the **position** is
   what lets the coordinator run the geometric test without needing a second round,
   and it is cheap (one `double[3]` per advertisement).
   **There is in-tree precedent: `refine()`'s Phase-2 coordinator reply already
   carries the split edge's squared length** (`SplitLenMsg`, with its `double len2`,
   `Tessera_RefineParallel.hpp:182`), added as Decision 15 in `4cee602` so the blue
   closure diagonal could be chosen geometrically. Follow that message shape, and
   compute every length comparison through `Tessera::edgeLen2Canonical()`
   (`Tessera_RefineClosure.hpp:212`) so two ranks comparing the same edge get
   bit-identical doubles regardless of endpoint order — that helper is what makes
   the priority in step 3 genuinely rank-count invariant.
2. **Decide, at the coordinator.** For each edge with exactly two advertisements
   and the owner's mask bit set, the coordinator has `a`, `b`, `c`, `d` and all
   four positions, so it evaluates:
   - **Boundary**: fewer than two incident faces → reject.
   - **Duplicate edge**: does `(c,d)` already exist? The coordinator for
     `makeEdgeKey(c,d)` is a *different* rank in general, so this needs one extra
     round — the deciding coordinator queries the `(c,d)` coordinator and only
     accepts on a negative answer. Do not skip it: flipping into an existing edge
     produces a non-manifold mesh, and it happens routinely on a coarse mesh (any
     valence-3 vertex).
   - **Geometric**: the policy's normal-deviation and quality tests, from the four
     positions.
3. **Conflict resolution — a deterministic independent set.** Two flips sharing a
   **face** conflict (that face would be rewritten twice); two sharing only a
   vertex do not. Build the conflict graph from
   [face-adjacency.md](face-adjacency.md)'s `nbrGid`/`nbrOwner` — actually the
   relation needed is "two candidate edges belong to the same face", which each
   face owner can evaluate locally over its own three edges' verdicts, so no graph
   traversal is required: **a face with two or more accepted candidate edges keeps
   only the highest-priority one.** Priority is `(squared length descending,
   EdgeKey ascending)` — computable at the coordinator from the positions it
   already has, totally ordered, and independent of rank count. Face owners
   exchange their local decisions back through the coordinator so both sides of
   each edge agree before anything is written.
4. **Apply.** Each face owner rewrites its face's `Verts`/`Edges`; the edge owner
   rewrites the edge's `Verts`/`Faces`. Rebuild `edgeKeys()`/`faceKeys()` and both
   CSRs; `setOwnedCounts` with unchanged counts to bump the generation.
5. **Halo.** `rebuildHalo()` before returning, preserving `halo.depth`.
6. **Empty mask** and **all-rejected** are no-ops with the counters reporting why.

**Decision 3 — one independent set per call, not a loop to exhaustion.** Iterating
inside `flipEdges` would hide an unbounded number of collectives behind one call
and make the cost unpredictable. The caller loops and watches `accepted`, which is
also what lets it re-derive valences between rounds. State the idiom in the header.

This deliberately does **not** reproduce a serial "rebuild the edge map after each
accepted flip" pass, which is inherently sequential and order-dependent. The
independent-set result differs from the serial one; both are valid. A consumer must
compare **quality statistics**, not the flip set.

### Non-goals

- Boundary edge flips (undefined — a boundary edge has one face).
- Choosing which edges to flip (Decision 2).
- Looping to exhaustion (Decision 3).

### Tests

New `tests/test_flip_edges.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

1. **Counts are invariant.** For every case below: `globalOwnedVertices`,
   `globalOwnedEdges`, `globalOwnedFaces` unchanged, `globalOwnedEuler == 2`,
   `checkConforming` and `checkNoInteriorVertex` pass. The cheapest possible
   detector of a botched rewrite.
2. **Involution — the sharpest check.** Mark one edge (globally smallest
   `EdgeKey`), flip, then mark the *same edge gid* and flip again. The multiset of
   face corner-**gid** triples must return **exactly** to the original, and so must
   the edge key set. A flip is its own inverse; nothing else in the test pins the
   rewrite this tightly.
3. **Duplicate-edge rejection.** On a subdivision-0 icosahedron every vertex has
   valence 5, so construct the case deliberately: a hand-built soup with a
   valence-3 vertex, where flipping the opposite edge would duplicate an existing
   edge. Assert `rejectedDuplicateEdge == 1`, `accepted == 0`, mesh unchanged, and
   that the edge key set contains no duplicates afterwards.
4. **Independent set.** Mark **every** owned edge on a subdivision-2 icosphere.
   Assert no face was rewritten twice — i.e. for each face gid, at most one of its
   three edges is in the accepted set. Derive the accepted set by gathering it and
   checking against the pre-flip face→edge map.
5. **Rank-count invariance of the accepted set.** Check 4 at ranks 1–5: identical
   `accepted`, and the identical multiset of face corner-**position** triples
   afterwards. This is the payoff of the deterministic priority in step 3; if it
   fails, the priority is not total or not rank-independent.
6. **Geometric rejection.** A policy with `minQuality = 0.99` rejects everything:
   `accepted == 0`, `rejectedGeometric == requested - rejectedBoundary`, mesh
   bitwise unchanged (`topologyChecksum`).
7. **Valence use case.** Compute valences from `mesh.vertexEdges()`, mask the edges
   whose flip would reduce total valence deviation from 6, and run three rounds.
   The subdivision-2 icosphere starts optimal (12 vertices of valence 5, 150 of
   valence 6), so assert the valence histogram does **not degrade** — no vertex
   below 4 or above 8, and the count of valence-6 vertices does not decrease. Also
   report minimum radius ratio before and after. Statistics, not edit sets, per
   Decision 3.
8. **Key and gid bookkeeping.** After any accepted flip: the edge **gid** set is
   unchanged; the edge **key** set has changed by exactly the accepted flips;
   `edgeKeys()` and `faceKeys()` contain no duplicates and match the AoSoA
   connectivity entry for entry.
9. **Halo valid on return.** `haloExchange()` immediately after leaves ghost
   positions equal to owners'; a second `flipEdges` with no intervening `migrate()`
   succeeds; `checkOwnershipPartition` passes.
10. **Empty mask** is a no-op (`topologyChecksum` unchanged), all counters zero.
11. **Family guard.** `flipEdges` on a `refine()`d mesh throws per
    [edge-split.md](edge-split.md) Decision 1; `flipEdges` after `splitEdges` is
    allowed and passes checks 1 and 2.
12. **Composed with split.** `splitEdges` (length-threshold mask) then three
    rounds of valence flips, five times over. Euler `== 2` and `checkConforming`
    after every operation; report the minimum radius ratio per round and assert it
    stays above a floor **measured in the first implementation run and recorded
    here** rather than guessed.

## Exit criterion

- `test_flip_edges` green at **SERIAL and HIP, ranks 1–5**, and the full gate still
  green with nothing relabelled.
- Check 2 (involution) passes exactly, and check 5 (rank-count invariance) passes —
  those two together are what make the operation trustworthy.
- Check 3 passes: flipping into an existing edge is detected, including across the
  extra coordinator round.
- README API section documents `flipEdges`, `DefaultFlipPolicy`, the three
  decisions (gid preservation and the broken `gid ↔ key` bijection; valence
  selection is the caller's; one independent set per call), and the caller-loop
  idiom.
- `docs/design.md` gains an *Edge flip* subsection with the owner-decides protocol,
  the duplicate-edge round, and the independent-set priority rule — plus the
  explicit note that the result differs from a serial shortest-first pass and that
  consumers compare statistics.

## Where this sits

**Requires [edge-split.md](edge-split.md)** (Editing families, Decision 1).
`rebuildHalo()` already exists in tree (`Tessera_HaloRebuild.hpp`, `25980f2`), so
[halo-depth.md](halo-depth.md) is **not** a prerequisite — Decision 2 keeps this
operation depth-1-safe on purpose. [face-adjacency.md](face-adjacency.md) is useful
context for step 3 but the final design avoids needing the face graph. See the
ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
