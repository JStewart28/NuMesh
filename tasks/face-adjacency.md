# Face→face adjacency through shared edges

**Status:** NOT STARTED.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

There is no way to ask "which faces share an edge with this face". The mesh
maintains vertex→face and vertex→edge CSRs, and the edge AoSoA carries
`EdgeField::Faces`, but neither answers the question on a distributed mesh.

**It cannot be derived locally, for two independent reasons:**

1. `EdgeField::Faces` holds **global ids** of the (at most two) incident faces.
   Those gids may name faces this rank does not hold, and `migrate()` carries them
   verbatim without repair. A gid is not a usable neighbour handle.
2. A vertex-incidence walk only finds an edge-neighbour that happens to be
   co-resident. The local face set is *owned faces plus faces incident on an owned
   vertex*, so an owned face all three of whose corners are ghosts can have
   edge-neighbours that are not locally held at all. The 1-deep **vertex** halo
   does not guarantee edge-neighbour co-residency.

Tessera's own `refine()` says exactly this, which is why its 2:1 balance routes
every cross-rank mark decision through an **edge coordinator** rather than through
the halo. That machinery is already built and already correct; it is just not
exposed as an adjacency query.

**Why it matters.** Two standard operations need it and neither is expressible
today. Growing a refinement mark by neighbour rings — mark a face, then its
edge-neighbours, then theirs — is how an indicator-driven AMR scheme avoids
refining isolated slivers. And any conflict-resolution pass over faces (choosing
an independent set of edge flips, for instance) needs the face graph. The driving
consumer is the Beatnik z-model's AMR mark expansion and its remeshing
conflict resolution; both are blocked on this.

## Approach

New header `src/Tessera_FaceAdjacency.hpp`.

### The honest return type

A CSR of local face indices is what a kernel wants, but reason 2 above means an
owned face's edge-neighbour may not be locally held. Returning `invalid_local` and
saying nothing would recreate the silent-incompleteness failure mode. Returning
gids only would be complete but unusable on device.

So return **both**, generation-guarded exactly like `VertexStencil`:

```cpp
template <class MemorySpace>
struct FaceAdjacency
{
    using memory_space = MemorySpace;

    //! Row per LOCAL face; entries are local face indices, or invalid_local when
    //! that neighbour is not held on this rank. Rows for OWNED faces are
    //! complete in the sense that every true edge-neighbour appears — as a local
    //! index if resident, as invalid_local if not. Rows for GHOST faces are
    //! best-effort and may be short; do not iterate them.
    GenerationHandle<CsrAdjacency<MemorySpace>> csr;

    //! Parallel to csr.neighbors: the neighbour's global id and owning rank.
    //! ALWAYS valid, resident or not. This is what a mark-propagation consumer
    //! uses — it sends to nbrOwner and names the face by nbrGid, and never needs
    //! the neighbour locally.
    Kokkos::View<GlobalId*, MemorySpace> nbrGid;
    Kokkos::View<Rank*, MemorySpace> nbrOwner;

    //! Count of owned-row entries with invalid_local, summed over this rank.
    //! Zero means every owned face's neighbours are co-resident, so a geometric
    //! consumer may use csr directly.
    long long numNonResident = 0;
};

//! Build the edge-adjacency of every local face. Collective. Rows are sorted
//! ascending by neighbour GLOBAL ID (not local index) so the row order is
//! rank-count invariant.
template <class MeshT>
FaceAdjacency<typename MeshT::memory_space> buildFaceAdjacency( MeshT& mesh );
```

**The precondition split is the design decision to record.** A *topological*
consumer (mark growth, conflict resolution, anything that communicates with the
neighbour's owner) needs only `nbrGid`/`nbrOwner` and works at halo depth 1. A
*geometric* consumer (one that reads the neighbour's vertex positions) needs
`numNonResident == 0`, which a deeper halo makes likely but does not guarantee
in general. Document both, and have the geometric consumers check
`numNonResident` rather than assume.

### Implementation — reuse the edge coordinator

No new communication machinery. Reuse `detail::edgeCoordRank` and `allToAllV`,
mirroring `refine()` Phase 2:

1. **Advertise.** For each **owned** face, send `(EdgeKey, faceGid, ownerRank)` to
   `edgeCoordRank(key, comm_size)` for each of its three edges. Route by the hash,
   disambiguate by the full `EdgeKey` at the coordinator, exactly as `refine()`
   does.
2. **Coordinate.** Each coordinator groups its advertisements by `EdgeKey` and
   sees the (at most two) incident faces of each edge. For each advertisement it
   replies to that advertiser with the **other** face's `(faceGid, ownerRank)`.
   An edge with one advertisement is a boundary edge and yields no reply; an edge
   with more than two is non-manifold — fail loudly with the offending key rather
   than silently truncating, since Tessera's data model assumes manifold.
3. **Assemble.** Each rank builds its owned rows from the replies. Ghost rows are
   filled best-effort from the locally derivable information (a ghost face's
   `FaceField::Edges` plus the local edge→face gids) and are marked as such by
   the row-completeness rule in the doc comment; ghost faces are not advertised,
   so no extra traffic.
4. **Resolve to local indices.** Build `gid → local face index` from the face
   AoSoA's `Gid` slice on the host (a `std::unordered_map`, as
   `buildVertexStencil` already does), fill `csr.neighbors` with the local index
   or `invalid_local`, and accumulate `numNonResident`.
5. **Sort each row ascending by `nbrGid`** and deep-copy the three arrays to the
   device. Sorting by gid rather than by local index is what makes the row order
   reproducible across rank counts, which is what makes check 4 below possible.

Build on the host with ordered containers, then deep-copy — the same locus and
idiom as `buildVertexStencil` and the halo builder. Stamp the CSR with
`mesh.generation()` / `mesh.generationPtr()` so a stale handle aborts.

### Conforming mode

On a `RefinementMode::Conforming` mesh, build adjacency over the **visible** faces
(what `ownedVisibleFaces()` returns), not over the retired red parents. A retired
parent is not part of the current triangulation and giving it a row would let a
consumer refine or flip something that is not there. State this, and skip retired
parents when advertising in step 1.

### Non-goals

- Growing a refinement mask by neighbour rings. That is the consumer's loop over
  this CSR; it needs a mark-exchange collective the consumer already has.
  Mention it as the intended use, do not implement it.
- Vertex→vertex or face→vertex adjacency. Both already exist or are derivable.
- Maintaining the adjacency incrementally across a topology edit. It is rebuilt,
  like every other derived structure, and the generation guard enforces that.

### Tests

New `tests/test_face_adjacency.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

Ground truth is exact and independent: the coarse icosphere soup is generated
identically on every rank, so the test derives the true face→face gid map from the
replicated soup with a plain `std::map<EdgeKey, vector<int>>` and compares.

1. **Neighbour gid sets match the reference.** Subdivision-2 icosphere,
   `distribute`. For every **owned** face, the `nbrGid` set equals the reference
   set for that face gid. Zero missing, zero extra, at every rank count. This is
   the definitive check; everything else is a corollary or a different regime.
2. **Degree and the closed-surface identity.** Every face has exactly 3
   edge-neighbours (closed manifold, no boundary). Sum of owned row lengths
   `== 3 * globalOwnedFaces == 2 * globalOwnedEdges` (960 == 960 at subdiv 2).
3. **Symmetry.** For every owned face `f` and neighbour gid `g`, `f`'s gid appears
   in `g`'s row. Checked against the reference, so it holds even when `g` is not
   locally held.
4. **Rank-count invariance.** The full owned-row map, keyed by face gid and with
   rows as gid lists, is identical at ranks 1–5. Rows are gid-sorted, so this is
   an exact list comparison, not a set comparison.
5. **`numNonResident` is reported, not hidden.** Print it per rank. Assert it is
   `0` at rank 1. At ranks ≥ 2 it may be nonzero; assert only that every
   `invalid_local` entry corresponds to a `nbrGid` that is genuinely absent from
   the local face gid set — i.e. the flag never lies in either direction.
6. **After `refine()`, Conforming mode.** Uniform refine, then rebuild. Every
   **visible** face has 3 neighbours; no retired parent has a row; `checkConforming`
   still passes; the reference for this case is derived from
   `ownedVisibleFaces()` gathered to one rank.
7. **After `refine()`, `HangingNode2to1` mode.** A T-junction face has more than 3
   edge-neighbours by construction, so do **not** assert degree 3 here. Assert
   symmetry (check 3) and that the union of rows over all faces is edge-consistent
   with `globalOwnedEdges`. This split is the mode contract, not a defect.
8. **Non-manifold input fails loudly.** Construct a small soup with three faces on
   one edge and assert `buildFaceAdjacency` throws naming the `EdgeKey`. Use a
   hand-built `TriangleSoup`, not a mutated icosphere.
9. **Generation guard.** Build the adjacency, `refine()`, then touch the stale
   handle — must abort, matching the `test_staleslice_guard.cpp` idiom.
10. **Single rank.** `numNonResident == 0` and the CSR equals the reference
    converted to local indices, exactly.

## Exit criterion

- `test_face_adjacency` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled.
- Check 1 passes against ground truth derived from the replicated soup, not from
  Tessera.
- Check 5 holds: `numNonResident` is exact in both directions — it is the whole
  reason the return type has two halves.
- README API section documents `buildFaceAdjacency`, the two-halves return type,
  and the **topological vs. geometric consumer** precondition split.
- `docs/design.md` gains a *Face adjacency* subsection recording why it cannot be
  derived locally (both reasons) and that it reuses `refine()`'s edge coordinator
  rather than a new mechanism.

## Where this sits

No hard prerequisite. [halo-depth.md](halo-depth.md) at depth ≥ 2 makes
`numNonResident == 0` far more likely and is worth landing first if geometric
consumers are the near-term need. Required by
[edge-flip.md](edge-flip.md) and [edge-collapse.md](edge-collapse.md) for their
conflict-resolution passes. See the ordering diagram in
[halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
