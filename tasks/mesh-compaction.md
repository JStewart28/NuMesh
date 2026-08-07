# Mesh compaction (tombstone removal and renumbering)

**Status:** NOT STARTED. Prerequisite of [edge-collapse.md](edge-collapse.md).
**Read the "Editing families" section of [edge-split.md](edge-split.md) first.**

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

Tessera has no way to remove entities. Nothing orphans a vertex today —
`refine()` is split-only, `Level` never falls, and every entity that exists stays
— so the gap has been invisible. The moment a coarsening operation exists
([edge-collapse.md](edge-collapse.md)), it produces dead entities and there is
nowhere for them to go: the AoSoAs have no delete, the owned-first ordering has no
way to close a hole, the vertex→face and vertex→edge CSRs would index removed
slots, the `edgeKeys`/`faceKeys` side tables would carry stale keys, and the halo
plans would name ghost slots that no longer exist.

Consequential rather than independent — but it must exist **before** collapse, not
alongside it, or collapse ends up growing its own private half-compaction.

**A second, less obvious hazard.** Several code paths index by **global id** into
a dense host array — `detail::buildKindPlan` takes
`const std::vector<LocalIndex>& gid2local` sized by the global entity count, and
`distribute()` builds `v2l`/`e2l`/`f2l` the same way. Gids are assigned by
`MPI_Exscan` onto a monotonically rising global count, so a long run of
split-and-collapse rounds grows the gid space **without bound** even while the
mesh stays the same size. The dense arrays then grow without bound too. `migrate()`
already avoids this (it uses `std::map<GlobalId, ...>`), so the hazard is confined,
but it is real and this task is where it gets addressed.

## Approach

New header `src/Tessera_Compact.hpp`.

### Tombstones

Uniform rule for all three entity kinds: **`Gid == invalid_gid` marks a dead
entity.** Uniform beats per-kind cleverness here — `VertexField::Flags` exists and
edges/faces have no equivalent, so using `Flags` for vertices only would mean two
mechanisms.

```cpp
//! Mark a local entity dead. Tombstoning does NOT repair connectivity: a caller
//! that kills a face must also kill the entities that become orphaned, and must
//! not leave a live entity referencing a dead gid. compact() verifies this and
//! throws on a dangling reference rather than producing a corrupt mesh.
template <class MeshT> void tombstoneVertex( MeshT& mesh, LocalIndex v );
template <class MeshT> void tombstoneEdge  ( MeshT& mesh, LocalIndex e );
template <class MeshT> void tombstoneFace  ( MeshT& mesh, LocalIndex f );
```

### `compact()`

```cpp
struct CompactStats
{
    long long verticesRemoved = 0, edgesRemoved = 0, facesRemoved = 0;
    long long gidSpaceBefore = 0, gidSpaceAfter = 0; //!< only if gids renumbered
};

//! Remove every tombstoned entity, restore owned-first ordering, rebuild the two
//! CSRs, the key side tables, and the three halo plans. GIDS ARE PRESERVED —
//! a surviving entity keeps the gid it had, so every cross-rank reference held by
//! another rank stays valid and no communication is needed to agree on renaming.
//! Collective (the halo rebuild is). Bumps generation.
template <class MeshT>
CompactStats compact( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo );

//! compact(), then renumber gids contiguously from 0 per kind via MPI_Exscan over
//! owned counts, so the gid space stops growing across many edit rounds. Strictly
//! more expensive (every rank must learn the new gid of every entity it ghosts,
//! and every connectivity field must be rewritten), so it is a SEPARATE call the
//! caller invokes periodically rather than something compact() does every time.
template <class MeshT>
CompactStats compactAndRenumberGids( MeshT& mesh,
                                     MeshHalo<typename MeshT::memory_space>& halo );
```

**Decision — gid preservation is the default, renumbering is opt-in and periodic.**
Preserving gids makes `compact()` a purely local reordering plus one halo rebuild;
renumbering makes it a global relabelling that invalidates every gid a peer holds.
A remesher calling compact every timestep wants the cheap one; the gid-space
growth is a slow leak that a call every few hundred steps fixes. Both are needed;
conflating them would make the common case pay for the rare one.

### Implementation

1. **Verify before mutating.** Sweep live entities for references to dead gids: a
   live face's `Verts`/`Edges`, a live edge's `Verts`/`Faces`, and each CSR row.
   Throw naming the offending live entity and the dead gid. A tombstone set that
   does not close is the single most likely caller bug and it must not produce a
   corrupt mesh silently.
2. **Build the local permutation.** Live entities only, **owned first then ghost,
   each ascending by gid** — the identical ordering rule `distribute()` and
   `migrate()` use, so compaction cannot introduce a third ordering convention.
3. **Compact each AoSoA** by whole-tuple copy into a fresh AoSoA of the live size,
   then `resizeX` + `deep_copy`, exactly as `migrate()`'s Round D does. Whole-tuple
   copy carries every user field automatically — do not enumerate fields.
4. **Rebuild the derived structures**: `rebuildVertexFaces`, `rebuildVertexEdges`,
   `setEdgeKeys`, `setFaceKeys`. Connectivity fields hold gids and gids are
   preserved, so **no connectivity rewrite is needed** in `compact()` — call that
   out, it is the payoff of the preservation decision.
5. **`setOwnedCounts`** with the live owned counts, which bumps the generation and
   invalidates every outstanding handle.
6. **Rebuild the halo** via `rebuildHalo()` (`Tessera_HaloRebuild.hpp`),
   preserving `halo.depth` if [halo-depth.md](halo-depth.md) has landed. Note that
   `rebuildHalo()`'s **round G** already pulls back any vertex or edge an owned face
   references but does not hold, so a compaction that removes a ghost still
   referenced by a surviving owned face is repaired rather than fatal — but do not
   rely on that: step 1's dangling-reference check must still fire on a
   non-closing tombstone set, because round G repairs *missing locality*, not a
   *deleted entity*. The ghost set genuinely changes — a ghost whose owner
   deleted it must disappear — so this is not optional and cannot be a plan patch.
7. **`compactAndRenumberGids`** adds, after step 5: `MPI_Exscan` over owned counts
   per kind to get each rank's new gid block; assign new gids to owned entities in
   the current (gid-ascending) order so the mapping is deterministic; exchange
   `old → new` for every ghosted entity through the existing halo plan direction
   (owner → ghost, so the plan built in step 6 already carries it); rewrite every
   connectivity field and both key side tables through the map; rebuild the halo
   **again** because the plans were keyed on old gids. Two halo rebuilds is the
   honest cost; do not try to fuse them in the first implementation.
8. **All-dead-on-one-rank** must work: that rank ends with zero owned entities, its
   peers drop it from their plans, and the collectives still complete. This is a
   real load-balance state, not a pathological one.

### Non-goals

- Deciding *what* to tombstone. That is collapse's job, or the caller's.
- Shrinking AoSoA capacity. Cabana resize semantics are Cabana's; compaction
  changes the logical size and lets capacity be whatever Cabana keeps.
- Repairing a non-closed tombstone set. It throws (step 1).

### Tests

New `tests/test_compact.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

The test needs a way to produce a valid tombstone set without collapse existing
yet. Use a **closed sub-mesh removal**: pick a set of faces by a rank-count-
invariant rule, then tombstone exactly those faces plus the edges and vertices
that no surviving face references. That set closes by construction, and the
resulting mesh is a surface **with boundary** — which is a legitimate exercise of
the machinery even though it is not what collapse will produce.

1. **No-op.** Nothing tombstoned. V/E/F, `topologyChecksum`, and every gid
   unchanged; `owned1RingLocal` still consistent; `CompactStats` all zero.
2. **Idempotence.** `compact()` twice: the second call is a no-op by check 1's
   criteria (`topologyChecksum` bitwise equal).
3. **Closed sub-mesh removal.** Remove the faces of one base-icosahedron patch
   (a fixed gid range at subdivision 2). Assert: local and owned counts drop by
   exactly the tombstoned counts on each rank; `CompactStats` matches;
   **every surviving entity's gid is unchanged** (compare against a pre-edit
   gid→position snapshot); owned-first ordering holds with gids ascending in each
   block; `owned1RingLocal` passes; and the surviving faces' corner-position
   triples are exactly the pre-edit set minus the removed ones.
4. **Halo correctness after removal.** `haloExchange()` leaves every ghost vertex
   position equal to its owner's; `checkOwnershipPartition` passes; the removed
   entities appear in no plan.
5. **Euler on the result.** For a disc-like removal, `V - E + F == 1` per boundary
   component (Euler characteristic of a sphere minus one disc). Compute the
   expected value from the removed patch's boundary rather than hardcoding — this
   is a real check that connectivity survived, and it is why removing a *closed*
   sub-mesh is a better test fixture than removing scattered faces.
6. **Dangling reference throws.** Tombstone one vertex and nothing else. Step 1
   must throw naming a live face and the dead vertex gid.
7. **Everything dead on the last rank.** At ranks ≥ 2, tombstone every entity on
   rank `size-1`. That rank ends with zero owned entities; the call completes on
   every rank; `globalOwnedEuler` reflects the reduced mesh; `haloExchange`
   succeeds; the empty rank appears as no peer's send or recv target.
8. **Generation guard.** A `VertexStencil` (or a slice) taken before `compact()`
   aborts when used after, matching `test_staleslice_guard.cpp`.
9. **`compactAndRenumberGids`.** After check 3's removal: gids are `[0, N)`
   contiguous per kind globally (verify by gathering owned gid sets);
   `gidSpaceAfter < gidSpaceBefore`; every connectivity field resolves; the
   surviving faces' corner-**position** triple multiset is **unchanged** from
   before renumbering (positions are the gid-independent identity of the mesh);
   `haloExchange` correct; `owned1RingLocal` passes.
10. **Renumbering is deterministic.** `compactAndRenumberGids` on the same input at
    ranks 1–5 gives the same *global* gid→position map. Assert as a sorted
    (gid, position) list comparison.
11. **Repeated rounds do not leak gid space.** Ten rounds of
    "tombstone a small patch, `compact()`" and assert `gidSpaceBefore` is
    monotone; then one `compactAndRenumberGids` and assert it returns to the live
    count. Pins the hazard this task exists to close.

## Exit criterion

- `test_compact` green at **SERIAL and HIP, ranks 1–5**, and the full gate still
  green with nothing relabelled.
- Check 3 confirms **gid preservation** and check 9 confirms **renumbering**; both
  are required, since the two calls have opposite contracts.
- Check 7 passes: a rank with zero owned entities is a supported state.
- Check 11 demonstrates the gid-space leak and its fix, with the measured numbers
  recorded in the progress log.
- README API section documents `tombstoneX`, `compact`, `compactAndRenumberGids`,
  the `Gid == invalid_gid` tombstone convention, and the gid-preservation decision
  with its rationale (cheap per-step compaction vs. periodic renumbering).
- `docs/design.md` gains a *Compaction* subsection, including the gid-space growth
  hazard and which code paths index densely by gid.

## Where this sits

**No hard prerequisite** — `rebuildHalo()` already exists in tree
(`Tessera_HaloRebuild.hpp`, `25980f2`), so step 6 is available today, and
[halo-depth.md](halo-depth.md) is only needed if a caller wants depth > 1.
**Required by
[edge-collapse.md](edge-collapse.md)**, which calls `compact()` internally. See the
ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
