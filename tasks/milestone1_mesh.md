# Milestone 1 — Distributed Unstructured Triangle Mesh

> **Handoff contract.** This library is built across multiple sessions with
> different models. This file is the source of truth for *what to build next, where
> it starts/ends, how it's accepted, who builds it (Sonnet/Opus), and what to report
> back* so the next session resumes cleanly. **At the start of any session touching
> this task, read this file first, then append to the Progress log as work lands.**
> The full design rationale lives in `README.md` (design reference) and the approved
> plan `plans/compiled-hopping-torvalds.md`.

## Problem

Tessera must provide the distributed, halo-able, refinable triangle mesh that the
Canopy FMM solver and the Beatnik/rocketrig rising-bubble problem build on. The
global Birkhoff–Rott/FMM velocity solve is **out of scope** (it lives in Canopy);
Tessera owns the local mesh machinery: entity storage, connectivity, 1-deep halo,
split-based AMR, optional migration/load-balancing, and parallel I/O.

## Approach

C++17, header-first, Kokkos + Cabana AoSoA, hand-built MPI. Mirror three proven
Canopy patterns (`~/spack_envs/tuolumne_beatnik/canopy/src/Canopy_{TreePartitioner,
MpiCoalescedExchange,RegisteredBufferPool}.hpp`) — do not modify Canopy. Key design
locks (see README for detail):

- `Mesh<Scalar, int Dim = 3, VertexFields, EdgeFields, FaceFields, MemorySpace,
  ExecSpace>` — **scalar precision AND embedding dimension are template parameters**
  (never hard-code `double`/`3`). `position` is `Scalar[Dim]`; `Dim=3` for the
  bubble, `Dim=2` expressible later.
- **Compile-time Cabana field pack** per entity kind for arbitrary user data
  (e.g. `VertexFields<Vort2>`); every field haloes/migrates with its entity.
- **Structured 128-bit gids** (ordered endpoint-gid pair), never a hash — guarantees
  cross-rank determinism with no comm and no collision risk at 1e8+ entities.
- **Lowest-rank ownership** of shared V/E; sharing set materialized by ghost build.
- **Conforming, 2:1-balanced** split refinement (hanging nodes bounded to ≤1 level).
  Midpoint placement via a **pluggable interpolation policy** (default linear
  midpoint for `position` + linear average for fields, per-field override hook).
- **Migration is the public contract** (`migrate(dest_rank_per_owned_face)`);
  internal Zoltan2 `loadBalance()` is a thin wrapper. LB is **optional** — Canopy
  can drive `migrate()` externally to avoid double migration.
- **Manual parallel HDF5 + XDMF** I/O with a full round-trip reader; dense global
  numbering via `MPI_Exscan` over owned-only counts.

### Gate / conventions (binding — see `CLAUDE.md`)

- Regression gate = label `regression` × backends {SERIAL, HIP} × ranks {1,2,3,4,5},
  single-sourced across `tests/CMakeLists.txt`, `run_regression_minset.flux`, CI,
  and `CLAUDE.md`. Unit/diagnostic tests use label `unit`.
- Every new source file carries the BSD-3-Clause SPDX header.
- README kept in sync with public API / example args. `--target format-check` clean.
- A failing test is never silently dropped from the gate (label `unit`, record in
  README Known Issues, report it).

## Step contract

Per-step model assignment: **Sonnet** = mechanical / lower-complexity (executed
against an Opus-written spec where the column says so); **Opus** = high design
complexity (distributed determinism, comm substrate, parallel AMR balance, Zoltan2
migration path).

| # | What to implement | Starts / ends | Acceptance (tests) | Model | Report back |
|---|---|---|---|---|---|
| 0 | Add `hdf5 +mpi` to the spack env; commit `docs/tuolumne/spack.yaml` snapshot; reconfirm full gate builds. | Start: edit spack env. End: gate builds clean with HDF5 discoverable by CMake. | Existing gate (Steps ≤7 once present) still green; `find_package(HDF5)` succeeds. | Sonnet | HDF5 version added; whether the Trilinos-adjacent rebuild perturbed anything; exact `spack.yaml` diff. Do **right before Step 8** to keep the env-mutation window small. |
| 1 | Design docs: this file + README design/data-model/API section. No code. | Start: TEMPLATE.md + README placeholder. End: README is a usable design reference; this contract complete. | N/A (docs). | Opus | Any design ambiguity found while writing; confirm README API matches intended template signature. **(DONE — see Progress log.)** |
| 2 | Core data model: `Mesh<Scalar, int Dim = 3, ...>` template skeleton; core topology AoSoAs + templated user-state AoSoA (field pack, `position` mandatory `Scalar[Dim]`); 128-bit structured-key utils (edge key, midpoint/child derivation); dense-local-index + canonical-key side table; CSR vertex-adjacency container. Convert INTERFACE target to real lib if needed. | Start: empty `src/`. End: headers under `src/mesh/`; mesh constructs empty; keys + CSR usable. | Unit (SERIAL+HIP), for **both `double` and `float`** (and a `Dim=2` compile check): structured-key determinism (ordering-invariant, same key regardless of endpoint order); AoSoA build; field-pack slice read/write. | Sonnet (keys Opus-spec'd) | Final gid/key struct + footprint decision (128-bit in AoSoA vs side table); field-pack declaration syntax; any Cabana template friction on HIP. |
| 3 | Serial mesh builder: coarse icosphere triangle soup → derive edges + both-direction connectivity (face→v/e, edge→f, vertex→e/f CSR). Single rank. | Start: Step 2 types. End: `buildIcosphere(subdiv)` yields a valid connected mesh. | Unit: Euler `V−E+F=2`; every edge has exactly 2 incident faces; every face has 3 edges + 3 verts; CSR 1-ring round-trips. | Sonnet | Icosphere generation choice (recursive subdivision vs lookup); base entity counts per subdiv level for test fixtures. |
| 4a | Comm primitive — migrate. Port `RegisteredBufferPool` verbatim; whole-tuple migrate generic over any AoSoA tuple (Alltoall counts + `MPI_Type_contiguous(sizeof tuple)` Isend/Irecv); self-peer guard. GPU-resident. | Start: Step 2. End: `src/comm/{RegisteredBufferPool,Migrate}.hpp`. | Round-trip migrate at ranks 1–5, SERIAL+HIP: send tuples to a known rank pattern, verify counts + payload; np1 self-peer no-op doesn't fault on HIP. | Opus | Confirm MI300A self-peer + signed-int-count guards reproduce; any Cabana tuple-copy gotchas on device. |
| 4b | Comm primitive — `HaloExchangePlan` (peer lists + gid→local-index maps + pools, `rebuild()`); in-place field-sync generalized to arbitrary value type / whole pack; variable-arity topology/connectivity exchange (for ghost build / sharing-set discovery). **Invariant: any local-count or ghost-set change → `rebuild()` before next sync.** | Start: 4a. End: `src/comm/{HaloExchangePlan,FieldSync}.hpp`. | Ranks 1–5: hand-build a plan, sync a multi-field pack, compare ghosts to owners; topology exchange reconstructs a known neighbor map. | Opus | The plan object's public shape (what Steps 5/6/7 call); how stale-plan invalidation is signaled; aligned pack-order rule used. |
| 5 | Distributed mesh + ownership + 1-deep halo. Replicated deterministic static partition (no Zoltan2); ghost build via 4b topology exchange; sharing-set → **lowest-rank** ownership; first halo field sync. | Start: 3 + 4b. End: `Mesh` distributes, owns, and halos. | Regression ranks 1–5 SERIAL+HIP: ownership is a partition (Σ owned = global; no gid double-/un-owned); every owned vertex's 1-ring local; ghost fields == owner; np1 = zero ghosts, no-op exchange. | Opus | Static-partition scheme used; the sharing-set discovery exchange; which invariants moved into `tests/MeshInvariants.hpp`. |
| 6a | Local 1→4 red refinement (single rank): midpoint vertices via structured keys; 4 child faces; connectivity rebuild. **Pluggable interpolation policy** (`src/mesh/RefinePolicy.hpp`): default linear midpoint for `position` + linear average for user fields, per-field override hook. | Start: 3. End: `refineLocal(face_mask)`. | Unit: Euler holds post-refine; child/parent incidence correct; midpoint shared between sibling faces; midpoint position == endpoint average under default policy; a custom policy overrides one field. | Sonnet | Child-face vertex/edge ordering convention; how `level` is propagated; the policy hook signature. |
| 6b | Parallel 2:1 balance + re-halo. Mark-and-propagate fixpoint (decision pure in gids+levels); sync `level` each iteration; `MPI_Allreduce` changed-count to fixpoint with hard cap + strictly-decreasing assert; rebuild halo plan each iteration. | Start: 6a + 5. End: `refine()` conforming across ranks. | Regression ranks 1–5: shared boundary edge midpoint gid bit-identical both sides (**key test**); 2:1 invariant; Euler over owned-only. | Opus | Iteration-count distribution observed; the cap value; any nondeterminism caught by the strictly-decreasing assert. |
| 7 | Migration API + optional Zoltan2 LB. Public `migrate(dest)` (faces + their V/E + whole field pack move) → recompute ownership → rebuild halo → sync (order explicit). Accessors `ownedFaceCentroids/Gids/Weights`. `loadBalance()` = Zoltan2 MultiJagged (SerialComm, solve-on-0+Bcast, never RCB) → `migrate()`. | Start: 4a + 5. End: `src/balance/Zoltan2Balancer.hpp` + `Mesh::migrate/loadBalance`. | Regression ranks 1–5: external path — hand-computed `dest` (and synthetic Canopy-style assignment) lands entities on requested ranks, invariants preserved; internal path — balance improves; rank-count-independent topology checksum. | Opus | The migrate→ownership→halo ordering that avoids phantom sends; Zoltan2 adapter/param specifics reused from Canopy; external-API surface Canopy will call. |
| 8 | Parallel HDF5 + XDMF writer + round-trip reader. Dense global numbering via `MPI_Exscan` over owned-only counts; XDMF references dense indices; write connectivity + per-V/E/F fields; reader reconstructs a mesh re-passing Step-5 invariants. (Needs Step 0 done.) | Start: 5 + Step 0. End: `src/io/{HDF5Writer,HDF5Reader}.hpp`. | Regression ranks 1–5: write→read→compare topology + field checksums; on-disk global checksum **independent of rank count**; opens in Paraview. | Sonnet (layout Opus-spec'd) | HDF5 dataset layout + XDMF schema used; the dense-numbering exscan; any collective-IO tuning needed on Tuolumne. |
| 9 | End-to-end example (icosphere→refine→rebalance→halo→write); separate unit vs regression runner scripts under `scripts/<system>/`; OPENMP smoke-build (non-gate); CI subset wiring; README/example-arg sync. | Start: 5–8. End: `examples/` + scripts + CI green. | E2E example runs at ranks 1–5; CI subset (`regression`/SERIAL ranks 1–2) green. | Sonnet | Example CLI args (mirror to README); runner script names; CI subset confirmation. |

## Progress log

- 2026-06-30 — **Step 1 started/landed (Opus).** Plan approved
  (`plans/compiled-hopping-torvalds.md`). Confirmed environment: Trilinos 16.2
  (+zoltan2), Cabana, Kokkos 4.7 (Serial/OpenMP/HIP gfx942), googletest present in
  `~/spack_envs/tuolumne_trilinos`; **HDF5 absent** (Trilinos `~hdf5`) → Step 0 will
  add `hdf5 +mpi`. Canopy reference confirmed at `~/spack_envs/tuolumne_beatnik/
  canopy` (the path in `background.md`, `~/spack_envs/beatnik/canopy`, does not
  exist). Wrote this handoff contract and the README design reference. **Next:**
  Step 2 (core data model) — Sonnet, against the gid/key spec in README.
- 2026-07-01 — **Design refinements (Opus, plan-mode).** Two user-requested changes,
  folded into the plan/README/this doc: (1) the mesh is now templated on embedding
  dimension too — `Mesh<Scalar, int Dim = 3, ...>`, `position : Scalar[Dim]` (`Dim=3`
  for the bubble; `Dim=2` expressible later). (2) Refinement midpoint placement is a
  **pluggable interpolation policy** (`src/mesh/RefinePolicy.hpp`): default linear
  midpoint for `position` + linear average for user fields, with a per-field override
  hook for a later curvature-aware/physics-correct rule. No code yet; Step 2 remains
  the next action.
- 2026-07-01 — **Step 2 landed (Opus; Sonnet delegation unavailable this env — the
  `sonnet` alias 403s for this subscription and raw model ids are rejected, so all
  steps run on Opus; the Sonnet/Opus column is advisory here).** Core data model
  implemented header-first under `src/` (flat `Tessera_*.hpp`, included as
  `<Tessera_Mesh.hpp>` per the README):
  - `Tessera_Types.hpp` — `GlobalId`/`LocalIndex`; structured `Key<N>` (EdgeKey=Key<2>,
    FaceKey=Key<3>) with order-invariant `makeEdgeKey`/`makeFaceKey`, device-callable.
    **Key-scheme decision resolved:** the 128-bit key is the cross-rank *matching
    identity* (side table); the 64-bit gid is the persistent identity; midpoint gid
    *assignment* (local exscan interior + owner-broadcast on boundary) is deferred to
    Step 6, which keeps keys bounded at Key<2>/Key<3> across refinement depth.
  - `Tessera_Fields.hpp` — core member layouts + `MemberTypesCat` to append the
    compile-time user field pack; named core slice indices (`VertexField::*` etc.) and
    `userVertexField<M>()`/`userEdgeField`/`userFaceField` for user slices.
  - `Tessera_CsrAdjacency.hpp` — CSR 1-ring container (built Step 3).
  - `Tessera_Mesh.hpp` — `Mesh<Scalar, int Dim=3, VUser, EUser, FUser, Mem, Exec>`
    skeleton: 3 AoSoAs, CSR members, key side tables, slice/resize/accessors.
  - Umbrella `Tessera.hpp`.
  Unit tests `tests/test_keys.cpp` + `tests/test_data_model.cpp` (self-contained, no
  gtest) run host **Serial** and device **Default(HIP)**, for **double and float**,
  incl. a **Dim=2** build; wired via `tessera_add_test` (unit tier, SERIAL+HIP, np1).
  All 4 pass. Build: `run_cmake_toulumne.sh` + make (SERIAL+HIP compile clean);
  `format-check` clean.
  - **Env fix (reach-Flux-tasks):** Cray `CC` links libsci and ROCm dlopens large-TLS
    libs → every binary aborted at load with *"libsci_cray_mp.so.6: cannot allocate
    memory in static TLS block"*. Fixed by `export
    GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608` in
    `scripts/tuolumne/runtime_env.sh` (LD_PRELOAD of libsci does NOT fix it and breaks
    flux itself; the tunable is the right remedy).
  - **Dev runner:** `scripts/tuolumne/run_unit_tests.flux` submits a ctest label to the
    pdebug queue via `flux batch` (login nodes can't `flux run` directly). Requires
    `export TESSERA_REPO=$(pwd)` before submit. Not a release artifact (Step 9
    formalizes runners). **Next:** Step 3 (serial icosphere builder + connectivity).
- 2026-07-01 — **Step 3 landed (Opus).** Serial builder + connectivity:
  - `Tessera_Icosphere.hpp` — `generateIcosphere<Scalar>(subdiv)` → watertight
    triangle soup (base icosahedron + 1→4 subdivision with shared-midpoint cache,
    projected to the unit sphere); deterministic/replicable.
  - `Tessera_MeshBuilder.hpp` — `buildFromTriangleSoup(mesh, soup)` /
    `buildIcosphere(mesh, subdiv)`: derive unique edges (dedup by EdgeKey), fill
    V/E/F AoSoAs (face convention `e[k]=edge(v[k],v[(k+1)%3])`), edge→face +
    face→v/e, vertex→faces and vertex→edges CSR, edge/face key side tables. Host
    derivation (ordered maps) then deep_copy into the mesh's (device) storage.
    Serial: gid == local index (Step 5 introduces the gid→local map).
  - Test `tests/test_connectivity.cpp` (unit, SERIAL+HIP, np1): exact V/E/F counts +
    Euler, every edge has 2 faces, faces have 3 valid distinct v/e, both vertex CSR
    relations round-trip (degrees 3F / 2E, membership), vertices on the unit sphere;
    subdiv 0–3. All 6 unit tests pass.
  - **Build/run env split fix (important, supersedes the Step-2 note):**
    `GLIBC_TUNABLES` **segfaults the Cray linker** if present at build time, but the
    run binaries need it. So it was REMOVED from `runtime_env.sh` (the resolver
    sources that for builds too) and is instead injected per test task via
    `MPIEXEC_PREFLAGS` `--env=GLIBC_TUNABLES=...` in `run_cmake_toulumne.sh` (flux
    run passes it to the task only; flux/compiler/linker env stays clean).
    `docs/tuolumne/claude.md` updated to match. **Reconfigure required** after
    pulling: re-run `run_cmake_toulumne.sh`. **Next:** Step 4a (comm migrate).
- 2026-07-01 — **Step 4a landed (Opus).** Migrate comm primitive:
  - `Tessera_RegisteredBufferPool.hpp` — verbatim port of Canopy's grow-only
    registered device pool (1.5x headroom, stable base address) into
    `Tessera::detail`; bounds CXI NIC registration churn.
  - `Tessera_Migrate.hpp` — `MigrateBuffers<Mem>` (persistent send/recv/idx pools)
    + `migrate(comm, aosoa, dest, bufs)`: generalizes Canopy migrate_particles to a
    caller-supplied per-element destination-rank array (dest(i)=owning rank).
    Alltoall(counts) → device pack into one registered send region → whole-tuple
    `MPI_Type_contiguous(sizeof(tuple))` Isend/Irecv → rebuild AoSoA as kept++recv.
    Self-peer never posted to MPI; np1/no-move fast path does zero MPI (MI300A
    self-send guard). Generic over any AoSoA tuple / Scalar, GPU-resident pack.
  - Test `tests/test_migrate.cpp` (unit, SERIAL+HIP, **np1–5**): scatter
    (dest=gid%size), shift (dest=(rank+1)%size, num_kept==0), identity (fast path,
    returns 0); verifies global conservation, correct landing rank, payload
    integrity, no duplicates. All 16 unit tests pass (incl. migrate np1–5 on HIP).
    **Next:** Step 4b (HaloExchangePlan + field-sync + topology-exchange).
- 2026-07-01 — **Step 4b landed (Opus).** Halo substrate:
  - `Tessera_AllToAllV.hpp` — `allToAllV<T>(comm, send)`: variable-arity neighbour
    topology exchange (Alltoall counts + Alltoallv bytes), host-side; the primitive
    Step 5 uses to advertise boundary gids/keys and build ghosts / discover sharing
    sets. Result groups received items by source rank. `T` trivially copyable.
  - `Tessera_HaloExchange.hpp` — `HaloExchangePlan<Mem>` (per-peer send/recv index
    maps grouped by peer + offsets, persistent registered pools, `clear()`,
    `setFromHost()` which drops self-peer; **alignment contract**: builder orders
    shared entities by gid on both sides so pack/unpack align) and
    `haloExchange(comm, aosoa, plan)` — whole-tuple field sync (syncs the entire
    field pack of every ghost from its owner), in place (no resize), GPU-resident,
    self-peer never posted. Invalidation invariant documented: any local-count/
    ghost-set change → rebuild before next sync.
  - Test `tests/test_halo.cpp` (unit, SERIAL+HIP, np1–5): allToAllV variable-length
    correctness; ring halo (send owned→rank+1, fill ghosts←rank-1) verifying every
    ghost gets the owner's whole tuple, owned untouched, np1 empty-plan no-op. All
    26 unit tests pass.
  - Gotcha fixed: `Kokkos::view_alloc(WithoutInitializing, label)` with a
    `const char*` variable is misread as a wrap-user-memory pointer; wrap in
    `std::string(label)`. **Next:** Step 5 (distributed mesh + ownership + 1-deep
    halo build).
- 2026-07-01 — **Step 5 landed (Opus). First regression-tier test.** Distributed
  mesh + ownership + 1-deep halo:
  - `Mesh`: added owned-count state (`numOwned{Vertices,Edges,Faces}`,
    `setOwnedCounts`); convention = entities stored owned-first, `[0,n_owned)` owned
    then ghosts. Serial builder marks all-owned. `fillCsr` moved to
    `Tessera_CsrAdjacency.hpp` (shared by builder + distribute).
  - `Tessera_Distribute.hpp`: `facePartitionByAxis` (deterministic geometric block
    partition — sort faces by centroid axis, tie-break gid, block into bands;
    replicated → identical on all ranks, no comm; Zoltan2 deferred to Step 7).
    `distribute(mesh, halo, faceOwner)`: lowest-rank ownership (vertex/edge = min
    incident faceOwner), 1-deep local closure (local faces = owned + incident-to-
    owned-vertex; local v/e = union over local faces), compacts to owned-first local
    AoSoAs, rebuilds local CSR, and builds the 3 `HaloExchangePlan`s. Send side of
    each plan discovered via `allToAllV` (advertise ghost gids to owners); recv/send
    aligned by ghoster gid order. `MeshHalo<Mem>` holds the 3 plans;
    `haloExchange(mesh, halo)` syncs V/E/F.
  - `tests/MeshInvariants.hpp` (shared, Steps 5–8): `checkOwnershipPartition`
    (Σowned==global + no gid double-owned via coordinator), `owned1RingLocal`,
    `topologyChecksum` (rank-count-independent BXOR). LOCAL-fail convention (sum ==
    global).
  - `tests/test_distribute.cpp` (**regression**, SERIAL+HIP, np1–5): partition →
    distribute → invariants → halo sync (corrupt ghosts, `haloExchange`, verify
    restored to owner's gid/owner; owned untouched; np1 no-op). All pass: regression
    10/10, unit 26/26.
  - Ran via `flux batch scripts/tuolumne/run_unit_tests.flux regression` (the dev
    runner takes a label arg). **Next:** Step 6a (local 1→4 refinement).
