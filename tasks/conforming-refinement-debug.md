# Conforming Refinement — Task 8 debugging

> **This file is self-contained.** A new session can be told *"read
> `tasks/conforming-refinement-debug.md` and complete the next task"* and needs
> nothing else to start. Background on the feature's design lives in
> [tasks/conforming-refinement.md](conforming-refinement.md) — read it when a
> debug task tells you to, not before.
>
> **How to work this file.** Tasks are ordered by dependency, D1 first. Do the
> lowest-numbered task whose status is not Done. When a task completes: update
> its status, record what the fix actually was under *What landed*, append to the
> *Findings log* at the bottom (append-only), commit, and push. If a fix changes
> a **design** decision, also rewrite the affected section of
> `tasks/conforming-refinement.md` — that file is the design's source of truth
> and must never describe something the code no longer does.
>
> **Rules inherited from Task 8 (non-negotiable).**
> - Never delete a test or weaken an assertion to get green.
> - A test that cannot be made green is relabelled `unit`, recorded in README
>   *Known Issues*, listed under *Open failures* here, and reported.
> - The gate definition (label `regression` × {SERIAL, HIP} × ranks 1–5) does not
>   change. Changing it requires updating `CLAUDE.md`, `tests/CMakeLists.txt`,
>   `scripts/tuolumne/run_regression_minset.flux`, and `.github/workflows/ci.yml`
>   together.

---

## Status

| # | Task | Status |
|---|------|--------|
| D0 | Triage sweep — run the suite, collect first failures | **Done** (2026-08-04) |
| D1 | Fix the `refine_splitedges` np≥2 hang | **Done** (2026-08-04) |
| D2 | Fix risk point 9 — the persistent split-edge map | **Done** (2026-08-04) |
| D3 | Fix the `refine_conforming` np≥2 `unordered_map::at` abort | **Done** (2026-08-04) |
| D4 | Re-sweep: get the remaining nine never-executed tests to a first verdict | **Mostly done by D3's gate run** — see D4 |
| D5 | Downstream conforming tests (`conforming_migrate`, `io`, `markquality_conforming`) | Not started |
| D6 | `conforming_operators` and `conforming_determinism` | Not started |
| D7 | `conforming_quality` — calibrate the provisional bounds | Not started |
| D8 | Full gate at ranks 1–5, both tiers, `format-check`; close out Task 8 | Not started |

**Open failures**, as measured by D3's full gate run (job `f3QJgiXFh3Ef`, 130
instances, reached #172 of 177 before its 20-minute window):

| Test | Verdict | Owner |
|---|---|---|
| `conforming_migrate` | np1 pass; **np2–5 abort** `unordered_map::at` | D5 |
| `conforming_operators` | np1 pass; **np2–5 abort** `unordered_map::at` | D6 |
| `conforming_determinism` | **np1 fails** `closure-idempotence`; np2–5 abort | D6 |
| `conforming_quality`, `markquality_conforming` | not yet reached | D5 / D7 |

All three aborts are **pre-existing and not caused by D3** — verified by rebuilding
with D3's library change stashed and re-probing at np2: byte-identical abort and the
identical np1 idempotence failure. They are almost certainly D1's re-halo defect
again (all three refine repeatedly; `test_conforming_determinism.cpp:516-517`
refines twice back-to-back with nothing in between).

**Everything else in the gate passes at ranks 1–5 on both backends**, including
`refine`, `refine_parallel`, `refine_splitedges`, `refine_conforming`,
`refine_closure`, `migrate_mesh`, `distribute`, `halo`, `loadbalance`, `io`,
`markquality_edge` and `markquality_curv`. D1, D2 and D3 are fixed.

---

## How to build and run (this is the whole environment recipe)

The Bash tool runs a **non-login** shell, where no `PrgEnv-*` module is loaded and
the Cray `CC` wrapper fails at configure. Everything below must go through
`bash -lc`, and `SPACK_USER_CONFIG_PATH` must be exported or spack silently drops
every Tuolumne system external.

```bash
bash -lc '
  export SPACK_USER_CONFIG_PATH=$HOME/.spack/tuolumne
  . /usr/WS2/stewartj/spack/share/spack/setup-env.sh
  spack env activate ~/spack_envs/tuolumne_trilinos/
  cd /g/g20/stewartj/research-bridges/tessera-dev/Tessera/build-tuolumne && make -j 48'
```

Build dir `build-tuolumne/` exists and is current. If `CMakeCache.txt` is ever
lost, reconfigure with `bash ../run_cmake_toulumne.sh` (never a bare `cmake .`,
which deletes the cache).

### Running tests — two harnesses

**Full sweep** (submit, then read `tessera-conforming-tests.<id>.out` in the repo
root):

```bash
flux batch scripts/tuolumne/run_conforming_tests.flux
```

That script is committed. It runs both tiers × {SERIAL, HIP} at np1–4 (`_np5`
excluded), with a 100 s per-test ctest timeout, on 1 pdebug node for 20 minutes.

**Isolated probe** — use this while debugging a single test. Copy
`probe.flux` below to the scratchpad, edit the `for` list, `flux batch` it, and
read `probe.<id>.out` in the repo root.

```bash
#!/usr/bin/env bash
#FLUX: --job-name=tessera-probe
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --queue=pdebug
#FLUX: --time-limit=15m
#FLUX: --output=/g/g20/stewartj/research-bridges/tessera-dev/Tessera/probe.{{id}}.out
set -uo pipefail
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"
cd "${TESSERA_BUILD_DIR}/tests"
P="--nodes=1 --exclusive --cores-per-task=1 --env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608 --unbuffered"
for spec in "tessera_test_refine_splitedges_SERIAL 2" "tessera_test_refine_conforming_SERIAL 1"; do
  set -- $spec
  echo "########## $1 np$2 ##########"
  timeout -k 5 100 flux run --ntasks $2 $P ./$1 2>&1 | grep -v "no version information"
  echo "########## exit=${PIPESTATUS[0]} ##########"
done
```

### Three harness traps that cost D0 a run — do not rediscover these

1. **A ctest timeout poisons every test after it.** ctest kills its own child,
   but the `flux run` sub-job keeps holding the node `--exclusive`, so every
   later `flux run` queues forever and also times out. In the D0 sweep this
   turned one genuine hang into eight bogus ones. **Always wrap a
   possibly-hanging `flux run` in `timeout -k 5 <sec>`** (as `probe.flux` does)
   — `timeout` signals `flux run`, which tears the sub-job down properly.
2. **A killed test loses all its stdout.** ctest reports `<end of output>` with
   nothing before it. That is *not* evidence the test hung before its first
   print. Use `--unbuffered` on `flux run` when you need partial output from a
   hang.
3. **`flux run` from the login node just queues** — it goes to the system
   instance and waits for a pdebug node, so an interactive one-off appears to
   hang. Always `flux batch`.

---

## D0 — Triage sweep — **DONE (2026-08-04)**

`scripts/tuolumne/run_conforming_tests.flux` (new, committed) submitted as job
`f3QH9jz6D1aw`; it hit its 20-minute limit after 86 of 165 tests, having been
poisoned by trap 1 above from test 79 onwards. An isolated re-probe (job
`f3QHQtUmFtpX`) established which of those failures were real.

### Verdicts — everything that actually ran

**Passing, np1–4, both backends** (32 registrations, 78 test instances):
`keys`, `data_model`, `connectivity`, `migrate`, `halo`, `refine`,
`refinement_mode`, `refine_closure`, `staleslice_guard`, `geometry`,
`stencil_topology`, `apply_stencil`, `reduce_faces`, `global_reduce`,
`distribute`, `migrate_mesh`, `refine_parallel`.

That is a substantial result on its own: **Tasks 1 and 2 are clean** —
`refinement_mode` and `refine_closure` both pass, so the `FaceField::UserBegin`
shift (risk point 1) and the closure patterns/tie-break/inverse are all sound.
`staleslice_guard` passing clears risk point 8. `distribute`, `migrate_mesh`,
`halo`, `apply_stencil` and `reduce_faces` all now run on a **`Conforming`**-mode
mesh by default and pass, so the wider face tuple does not disturb construction,
distribution, halo, geometry, or the operators (risk point 6 clear on HIP too).

**Failing:**

| Test | np1 | np≥2 | Nature |
|---|---|---|---|
| `refine_splitedges` (SERIAL) | Pass | **Hang** (100 s timeout) | D1 — **fixed** |
| `refine_splitedges` (HIP) | *unknown* — poisoned | *unknown* | D1 — **fixed**, np1–5 |
| `refine_conforming` (SERIAL) | **Fail, rounds 2–3** | **Abort** `std::out_of_range` | D2 — **fixed** (np1) / D3 — open (np≥2) |

**Never executed at all** (they sit after `refine_conforming` in ctest order and
the sweep never reached them): `conforming_migrate`, `loadbalance`, `io`,
`markquality_edge`, `markquality_curv`, `conforming_operators`,
`conforming_determinism`, `conforming_quality`, `markquality_conforming`.

### The `refine_conforming` np1 output, in full — this is the primary evidence

```
test_refine_conforming: distributed conforming refinement (size 1)
  [Serial] adaptive round1 ok   (it=1 F=596  euler=2    closure=261/596 =0.438 |S| hist=[335,108,15,0] blue lo1/lo2=6/9   | control euler=-136 badInc=414  tjunc=138)
  [Serial] adaptive round2 FAIL (it=1 F=1013 euler=-136 closure=357/1013=0.352 |S| hist=[656,140,23,2] blue lo1/lo2=12/11 | control euler=-314 badInc=1031 tjunc=248)
  [Serial] adaptive round3 FAIL (it=1 F=1606 euler=-328 closure=493/1606=0.307 |S| hist=[1113,225,13,1] blue lo1/lo2=7/6  | control euler=-426 badInc=1474 tjunc=295)
  [Serial] empty-mask   ok (V=162 E=480 F=320 closureChildren=0)
  [Serial] uniform-mask ok (V=642 E=1920 F=1280 closureChildren=0 |S| hist=[1280,0,0,0])
```

(`[Default]` repeats `[Serial]` identically — the same two rounds fail.)

**Risk point 9 is confirmed, exactly as Task 7 predicted it:** round 1 is
conforming (`euler=2`), rounds 2 and 3 are not. Round 2's `euler=-136` is
*numerically identical* to round 1's hanging-node **control** — i.e. from round 2
on, the conforming mesh has degraded to exactly the hanging-node mesh's defect
count. The closure still fires (`|S| hist` has non-zero entries every round), it
just only closes the level jumps created *this* round, leaving every jump
inherited from an earlier round open. Non-vacuity holds throughout (the control
fails conformity every round), so the test is measuring something real.

The `empty-mask` and `uniform-mask` cases pass, but note they run on a **fresh**
mesh, not on the refined one — so `empty-mask` here is *not* the idempotence
probe. `conforming_determinism`'s idempotence case is, and it has not run yet.

---

## D1 — Fix the `refine_splitedges` np≥2 hang

**Status: DONE (2026-08-04).** `refine_splitedges` passes SERIAL and HIP at
np1–5. Two independent test-side defects, the second only visible once the first
was fixed. **The second one very likely explains D3 as well — read *What
landed* before starting D3.**

**Reproduce.**

```
probe.flux with: "tessera_test_refine_splitedges_SERIAL 2"
→ exit=124 (timeout at 100 s), no output captured
```

np1 passes in 3.8 s and prints a complete, plausible report:

```
[Serial] round1 ok (refining=46 splitEdges=138 keptOnlyDiscovered=0 phase2a: 960 total vs 138 refining-only, x6.96)
[Serial] round2 ok (it=1 faces=731 localMidsSum=267 phase2a: 1374 vs 273, x5.03)
[Serial] round3 ok (it=1 faces=950 localMidsSum=219 phase2a: 2193 vs 219, x10.01)
[Serial] round4 ok (it=1 faces=1124 localMidsSum=174 phase2a: 2850 vs 174, x16.38)
```

**Leading hypothesis: the hang is in the test, not in the library.**
`refine_splitedges` is pinned to `RefinementMode::HangingNode2to1` (Task 7A), and
`refine_parallel` — same mode, same `refine()`, same extended Phase 2 — **passes
at np1, 2, 3 and 4**. So the shared Phase-2 code path does not deadlock. What
`refine_splitedges` has that `refine_parallel` does not is its own multi-rank
reference machinery: the partition-free round-1 reference, the per-rank
kept-side-discovery reduction, and `checkSplitEdgeCoverage()`'s coordinator
round trip. A collective inside a rank-conditional branch (`if (rank == 0)`, an
early `continue` on an empty local set, or a `++fails` early-out that skips a
later `MPI_Allreduce`) deadlocks at np≥2 and is invisible at np1.

**Procedure.**

1. Read `tests/test_refine_splitedges.cpp` and list every MPI call. For each,
   confirm it is reached by all ranks unconditionally — in particular that no
   `MPI_Allreduce`/`allToAllV` sits inside a loop whose trip count is a local
   quantity, and that no failure path returns before a collective.
2. If that audit is clean, get a stack. Rebuild with
   `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (already the default) and attach:
   inside a `flux batch` allocation, `flux run` the np2 job in the background,
   then `gdb -p <pid> -batch -ex bt` on both ranks. Cheaper first step: bisect
   by adding flushed per-phase prints to the test and re-probing with
   `--unbuffered`.
3. Also probe `tessera_test_refine_splitedges_HIP` at np1 and np2 — its D0
   verdicts were all poisoned and are unknown. If HIP np1 also hangs while
   SERIAL np1 passes, the cause is different and belongs to risk point 6, not
   here.

**Acceptance.** `refine_splitedges` passes SERIAL and HIP at np1–5. **Met.**

**What landed.**

The leading hypothesis above was right that the bug was in the test rather than
in the library, and right about the mechanism — *a collective inside a
rank-conditional branch* — but the audit in step 1 of the procedure misses it,
because the collective is not a bare `MPI_*` call: it is hidden in an **argument
to the `printf`**.

**Defect 1 — the hang.** `tests/test_refine_splitedges.cpp`, the rounds-2-4
loop:

```cpp
if ( rank == 0 )
    std::printf( "  [%s] round%d %s (it=%d faces=%lld ...",
                 ..., TesseraTest::globalOwnedFaces( mesh ), ... );
                 //   ^^^^^^^^^^^^^^^^ MPI_Allreduce, rank 0 only
```

`globalOwnedFaces()` (`tests/MeshInvariants.hpp:222`) is an `MPI_Allreduce`.
Rank 0 entered it at the end of round 1 and every other rank walked on into
round 2's `refine()`, whose first collective is Phase 1's `allToAllV` — a
deadlock. np1 cannot see it. Fixed by hoisting the call to a `const long long
gFaces` above the `if`, evaluated by every rank.

Localised by tracing: temporary per-rank `fprintf(stderr)` markers around every
phase of the test, run with `TESSERA_PROBE=1`. The decisive clue was that
rank 0's *round-1 print itself* never appeared while rank 1 had already printed
`round2 refine enter` — i.e. rank 0 was stuck **inside** the print, not before
or after it.

A scan of every `if ( rank == 0 )` block in `tests/` and `examples/` for a
collective (including ones reached through a helper) found **this as the only
occurrence**, so no sibling fix was needed.

**Defect 2 — a `std::out_of_range` abort, unmasked by the fix.** With the
deadlock gone, np2-5 got further and aborted in round 2 with
`unordered_map::at` on a non-zero rank — *the same exception as D3, in
`HangingNode2to1` mode, in a test with no closure at all.*

Root cause: `refine()` drops every ghost and clears the halo plans
(`src/Tessera_RefineParallel.hpp:926`), but its own Phase 3a needs the
**positions of both endpoints** of every midpoint the rank owns, in order to
interpolate that midpoint (`gid2lv.at( a )`, line 533). Midpoint ownership is
"lowest incident refining-face owner", so a rank can own the midpoint of an edge
one of whose endpoints it holds only as a *ghost* — and after the previous
`refine()` that ghost is gone. Refining twice with no rebuild in between
therefore throws at np >= 2. This is the already-documented README *Known
Issue* ("a distributed mesh must be re-haloed after `refine()`"), one step
sharper than it was recorded: the consequence is not just an inert
`haloExchange()`, it is a **throw from inside the next `refine()`**.

Fixed in the test, not the library, because the library's contract is already
the documented one and two other tests already follow it
(`test_conforming_determinism.cpp:289` and `test_conforming_quality.cpp:324`
both re-halo between rounds with exactly this idiom):

```cpp
std::vector<Rank> dest( mesh.numOwnedFaces(), static_cast<Rank>( rank ) );
migrate( mesh, halo, dest );   // identity: the Step-7 halo rebuild rides along
haloExchange( mesh, halo );
```

placed at the *end* of the round body — after every check and the print — so the
invariants still measure exactly what `refine()` produced, with no ghosts
present.

**Side effect, benign, worth knowing.** The identity `migrate()` permutes the
local face ordering, so round 2's children are assigned gids in a different
order, so a **gid-derived mask picks a different face set** in later rounds.
Visible as round 3's `localMidsSum` moving 219 -> 218 at np1. Nothing regressed:
the mesh is the same size (`faces=` is identical round-for-round at every rank
count) and all of this test's checks are reference-free coordinator-decided
ground truth, so they hold under any such permutation. But do **not** treat a
gid-masked test's per-round counts as rank-count invariants — that is
`conforming_determinism`'s job, and it uses a *geometric* mask for exactly this
reason.

**Result, np1-5 x {SERIAL, HIP}, all `exit=0`** (job `f3QHvwB5yGRu`).
Non-vacuity holds and strengthens with rank count — round 1's
`keptOnlyDiscovered` is 0 / 6 / 13 / 24 / 27 at np 1 / 2 / 3 / 4 / 5, so the
kept-side cross-boundary discovery that Task 3 exists to provide is genuinely
exercised. Global face counts are identical at every rank count
(458 / 731 / 950 / 1124 over the four rounds). Phase-2a message volume, for the
Task-8 measurement table: **x6.96, x5.03, x10.01, x16.38** total-vs-refining-only
over rounds 1-4 (rank-count independent).

**Also landed.** README *Known Issues*: the re-halo entry now states the
second-`refine()` throw, gives the identity-`migrate()` idiom as a code block,
and notes the gid-permutation side effect. The per-round `printf`s in this test
now `fflush( stdout )` so a later hang cannot swallow them (harness trap 2).

**Steer for D3 — read this before debugging D3 separately.** D3 is an
`unordered_map::at` abort on a non-zero rank at np>=2 in `refine_conforming`.
Defect 2 above is the *same exception, same rank-count threshold, same
round-2 onset*, and `test_refine_conforming.cpp`'s round loop (line 180) has
**no re-halo between rounds** either. So the first thing to try for D3 is adding
the identity-`migrate()` + `haloExchange()` idiom to that loop — it may well be
the whole of D3, and it is not a closure bug. Two more loops need the same
check when their turn comes: `test_conforming_determinism.cpp:516-517` refines
twice back-to-back with nothing in between, and `test_markquality_edge.cpp:361`.
D3's note that "the fix is to make the map complete (D2), not to soften the
`at()`" still stands for `midGid.at()` — but the map that is actually throwing
here is `gid2lv`, which is a different map and a different problem.

---

## D2 — Fix risk point 9: the split-edge map is this-round-only

**Status: DONE (2026-08-04).** `refine_conforming` passes SERIAL and HIP at np1:
all three adaptive rounds `euler=2`, plus empty-mask and uniform-mask. **The
central defect of Task 8, and it was three coupled defects, not one — read *What
landed*.** np≥2 is still D3.

**Reproduce.** `probe.flux` with `"tessera_test_refine_conforming_SERIAL 1"` →
`exit=1`, rounds 2 and 3 FAIL as quoted under D0.

**Root cause (established by reading the code in Task 7, confirmed by D0's
numbers).** Step 3b′ in `detail::refineImpl()`
(`src/Tessera_RefineParallel.hpp`) calls `closeFaces( newRed, midGid, … )`.
`midGid` is built entirely inside Phase 2 from **this round's** refinements: the
coordinator drops every edge with `!anyRefining` (see the 2b block, `if
( !a.anyRefining ) continue;`). But a kept face needs closing whenever its edge
is bisected **in the red layer**, which is a *persistent* condition.

Concretely: round 1 refines A, so its kept neighbour B's edge `(a,b)` is bisected
at `m` and B closes into two green children — correct. Round 2 un-closes B back
to red corners `(a,b,c)`; A's children are level 1, B is level 0, and the 2:1
fixpoint permits that, so B does not refine. In Phase 2a, B advertises `(a,b)`
with `refining = 0`; A's children advertise `(a,m)` and `(m,b)`, which are
*different keys*. `(a,b)` therefore has no refining incidence, is dropped, and is
absent from `midGid`. `closeFaces()` computes `|S| = 0` for B and passes it
through **unclosed** — the T-junction returns.

**Proposed fix (from Task 7; weigh it, do not apply blindly).** The persistent
split-edge map is recoverable **locally, with no communication and no new field**,
from the closure bookkeeping already stored. For a parent `P` with corners
`(a,b,c)` and its children, a parent edge is split **iff it does not appear as an
edge of any child**: green replaces `(a,b)` by `(a,m),(m,b)` while `(b,c)` and
`(c,a)` survive; blue removes two; red-closure removes all three; `|S| = 0`
removes none. The midpoint of a split parent edge `(x,y)` is then the unique
child corner `m` outside `{a,b,c}` with both `(x,m)` and `(m,y)` present as child
edges. So `unclose()` (in `src/Tessera_RefineClosure.hpp`) can return an
`EdgeKey → GlobalId` map alongside the red layer, and step 3b′ closes against
`midGid` **unioned with** it.

Three things to get right — the first two are called out in the design, the
third follows from D3 and is new:

1. The recovered map must be consulted for **kept faces only**. A face refined
   this round has its edges replaced anyway.
2. `countClosureChildren()` in step 3c must be given **the same union**, or the
   face-gid `MPI_Exscan` under-allocates and children collide.
3. The union must be applied in `refineLocal()`'s conforming path
   (`detail::refineLocalConforming()` in `src/Tessera_Refine.hpp`) too, not just
   the distributed one — `refine_closure` passes today only because its
   conforming case is a *single* `refineLocal()` call from a fresh mesh.

An alternative worth one paragraph of thought before committing to the local
reconstruction: have the Phase-2 coordinator treat a red edge with a **single**
incidence as split. It needs the midpoint gid, which only the finer side knows
and which the local reconstruction already has — so it is probably strictly
worse. Record whichever you choose, and why, in the *Decision record* of
`tasks/conforming-refinement.md`.

**Acceptance.** `refine_conforming` passes all three adaptive rounds plus the
empty-mask and uniform-mask cases, SERIAL, np1. The non-vacuity control must
still **fail** conformity every round (`control euler` stays negative) — if the
control starts passing, the fix has made the mask too weak and the test is
proving nothing.

**Also update.** `tasks/conforming-refinement.md`: rewrite the *Design →
Distributed algorithm* step 3b description and the *Phase 2 extension*
paragraph to describe the union, and add a Decision-record entry. Risk point 9
becomes a resolved finding. **Done.**

**What landed.**

The local reconstruction was the right call and the alternative is strictly worse
(Decision 8 in `tasks/conforming-refinement.md` records why). But the closure union
by itself gets round 2 only, and **two further defects had to be fixed for round 3**;
both were invisible before the union existed, and both are about the *red engine*
rather than the closure. All three are in the library, none in the tests.

**Defect 1 — the union (risk point 9 proper).** `unclose()` now also returns
`UncloseResult::splitEdges`, recovered by the new `recoverSplitEdges()` in
`src/Tessera_RefineClosure.hpp`. `refineImpl()` captures it as `persistentSplit`
(step 0c) and `emplace`s it into Phase 2's `midGid` before `countClosureChildren()`
and `closeFaces()`; `refineLocalConforming()` seeds its `midpointOf` with it.
`RefineResult::midpoints` therefore publishes the union, which is what makes
`checkClosureInverse` (it re-closes from `res.midpoints`) agree without any test
change.

**The uniqueness rule in the proposed fix above is wrong as written** — worth
knowing, because it reads convincingly. "The midpoint of `(x,y)` is the unique child
corner `m ∉ {a,b,c}` with `(x,m)` and `(m,y)` both present as child edges" is
ambiguous for **blue**: in the `q0 < q1` diagonal the children are `(A,q0,C)`,
`(q0,B,q1)`, `(q0,q1,C)`, so for parent edge `(B,C)` both `q1` (correct) and `q0`
(via `(q0,B)` and `(q0,C)`) satisfy it. The repair is to require each half-edge to
belong to **exactly one** child: an edge shared by two children is a *diagonal* of
the parent's fan, and fan-*boundary* edges — which is what the two halves of a split
parent edge are — appear exactly once. Verified by hand against all four patterns
including both blue diagonals. `recoverSplitEdges()` aborts if the answer is not
unique, which is also a cheap detector for a closure family split across ranks.

**Defect 2 — a refining coarse face minted a second, coincident midpoint.** Not
predicted anywhere, and *pre-existing since Step 6b*. If `(a,b)` is bisected at `m`
and the coarse face `K` owning `(a,b)` refines, Phase 2 allocates a **fresh**
midpoint for `(a,b)` — the fine side's `m` is invisible to it, because the fine
faces carry `(a,m)`/`(m,b)`, which are different keys. Two coincident vertices, and
the mesh is cracked. Fixed by the same union: `midOf( a, b )` now finds the
persistent entry, so the split reuses `m` and the red layer becomes *conforming*
there (edge `(a,m)` gains its second incidence from `K`'s child). Consequence for
`closeFaces()`: a fresh red child's two inherited boundary half-edges can now be
**old** edges and so can legitimately be bisected this round, which is why the
`freshChild` assertion is no longer "|S| must be 0" but "no split edge may touch a
vertex created this round" — hence the new `firstNewVertexGid` parameter.

**Defect 3 — the 2:1 propagation was blind across a hanging node.** Also
pre-existing. Both coordinator rules that matter (Phase 1's level comparison, Phase
2's split decision) require **two** incident faces for an edge key, and a hanging
node leaves exactly one on each side: `(a,b)` is advertised only by the coarse face,
`(a,m)` only by one fine face. So a level jump across a hanging node was never
bounded, and round 3 hit `closeFaces()`'s "an edge of a red face is bisected more
than once" abort. Fixed with `forEachSubEdge()` in `refineImpl()`: a persistently
split edge is advertised to the coordinator as its **two half-edges**, in Phase 1
and Phase 2 both, so the coarse face meets its true neighbours. Visible in the run
as the fixpoint doing real work for the first time — `it=1/2/3` over rounds 1/2/3,
where it was `it=1/1/1` before.

**The subtle half of defect 3, which cost the second build.** A half must be
advertised with `refining = 0` **regardless of the advertising face's own mark**. A
refining coarse face bisects the *whole* edge, at the midpoint it already has, and
bisects neither half. Advertising a half as refining makes the coordinator mint a
midpoint *for the half* — a spurious refinement that propagates and lands as
exactly the same "bisected more than once" abort one round later. The first attempt
had this wrong and produced a plausible-looking round 2 (`euler=2`, `F=1494`) that
aborted in round 3; with it fixed, round 2 is `F=1222`. **A conforming round that
passes every check can still be over-refining — compare `F` against the
hanging-node control's growth, not just `euler`.**

**Result, np1, both backends** (jobs `f3QJLhrfQLKh`, `f3QJMiKVuhm9`):

```
round1 ok (it=1 F=596  euler=2 closure=261/596 =0.438 |S| hist=[335,108,15,0]  blue lo1/lo2=6/9   | control euler=-136 badInc=414  tjunc=138)
round2 ok (it=2 F=1222 euler=2 closure=557/1222=0.456 |S| hist=[665,194,39,13] blue lo1/lo2=19/20 | control euler=-314 badInc=1031 tjunc=248)
round3 ok (it=3 F=2386 euler=2 closure=1100/2386=0.461 |S| hist=[1286,392,84,16] blue lo1/lo2=47/37 | control euler=-426 badInc=1474 tjunc=295)
empty-mask   ok (V=162 E=480 F=320 closureChildren=0)
uniform-mask ok (V=642 E=1920 F=1280 closureChildren=0 |S| hist=[1280,0,0,0])
```

Non-vacuity holds and is not weakened: the control still fails conformity every
round and its defect counts still *grow* (414 → 1031 → 1474). **For D7:** the
closure fraction is essentially flat across rounds (0.438 / 0.456 / 0.461) and both
blue diagonals stay populated, which is the first real evidence that the closure
cost and the shape bound are round-independent — the premise D7 has to confirm.

**No regressions.** 14 spot-checked test instances all `exit=0`, chosen to cover
everything the change could touch: `refine_closure` (SERIAL+HIP, the
`refineLocalConforming` path), `refine`, `refinement_mode`, `refine_parallel`,
`refine_splitedges`, `migrate_mesh`, `halo`, `distribute`, `apply_stencil`,
`staleslice_guard` at np1–4 across both backends. `HangingNode2to1` is provably
untouched: `persistentSplit` is always empty there, so `forEachSubEdge()` degrades
to the old single advertisement and the union is a no-op — its Phase-2a volumes are
byte-identical (`x6.96 / x5.03 / x10.01 / x16.38`, unchanged from D1).

**Also landed.** `tasks/conforming-refinement.md`: *Persistent split edges* section
under the distributed algorithm, step 0c/1/3/3b in the pseudocode, Decisions 8 and
9, risk point 9 marked resolved (with the original analysis preserved), and a new
*Known limit* recording that defects 2 and 3 remain in `HangingNode2to1` mode —
they cannot be fixed there without a new field or a new message round, and that
mode's tests cannot see them because they assert non-conformity anyway.

---

## D3 — Fix the `refine_conforming` np≥2 `unordered_map::at` abort

**Status: DONE (2026-08-04).** `refine_conforming` passes SERIAL and HIP at np1–5.
Two defects: D1's steer was right about the abort, and fixing it exposed a second,
**pre-existing library** defect in how `refine()` assigns edge gids. **Read *What
landed*.**

**Status when D3 started:** **D2 did not fix it and did not change it** — re-probed
after D2 at np2 and np3, still `std::out_of_range: unordered_map::at`, still on a
non-zero rank, still before any output flushes. So it is *not* the missing
persistent split edge; **start from D1's steer instead** (the missing re-halo in
`test_refine_conforming.cpp`'s round loop, line ~180 — `gid2lv.at()` in Phase 3a
needs both endpoint positions of every midpoint the rank owns, and one of them is
a ghost `refine()` already dropped). The "candidate sites" paragraph below was
written before D1 established that, and `midGid` — the map it points at — is now
complete, so treat `gid2lv` as the prime suspect rather than `midGid`.

**Reproduce.** `probe.flux` with `"tessera_test_refine_conforming_SERIAL 2"`:

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  unordered_map::at
flux-shell[0]: FATAL: task-exit: task rank 1 ... exited (Aborted)
########## exit=134 ##########
```

Note it aborts on **rank 1**, and before any test output flushes, so the round
it dies in is unknown. Re-run with `--unbuffered` and an explicit `fflush` (or
`std::endl`) after each round's print to find out.

**Candidate sites.** Search for `.at(` on a midpoint or gid map:
`midOf()` in `detail::refineImpl()` (`midGid.at( keyOf( a, b ) )`) is the prime
suspect — it is an `at()` on the very map D2 shows to be incomplete. Also check
the `midpointOf` lambda handed to `closeFaces()` and any `.at()` in
`tests/MeshInvariants.hpp`'s new checks. Whatever the site, the *fix* is to make
the map complete (D2), not to soften the `at()` into a `find()` — an `at()`
throwing here is the map being wrong, and silently substituting `invalid_gid`
would turn a loud failure into a corrupt mesh.

Also note this fires at np2 but not np1, so there is a partition-dependent
component: a rank holding only the kept side of an edge is the case np1 cannot
produce. That is exactly what Task 3 built the extended Phase 2 to cover, so
check whether the failing lookup is on a rank that should have received the gid
in round 2c.

**Acceptance.** `refine_conforming` passes SERIAL and HIP at np1–5. **Met**, at
both tiers' expense of nothing: no test was relabelled, and one assertion got
*stronger* (below).

**What landed.**

**Defect 1 — the abort was exactly D1's re-halo defect.** `gid2lv.at()` at
`src/Tessera_RefineParallel.hpp:611` (Phase 3a) needs the *positions* of both
endpoints of every midpoint the rank owns; midpoint ownership is "lowest incident
refining-face owner", so a rank can own the midpoint of an edge one of whose
endpoints it holds only as a ghost — and `refine()` has already dropped every ghost.
`test_refine_conforming.cpp`'s round loop had no rebuild between rounds, so round 2
threw at np ≥ 2. Fixed test-side with the documented identity-`migrate()` +
`haloExchange()` idiom, applied to **both** the conforming mesh and the
hanging-node control, placed after every check and the print so the invariants still
measure exactly what `refine()` produced. `midGid` was never involved — D3's
original "candidate sites" paragraph pointed at the wrong map, as D1 suspected.

**Defect 2 — `refine()` minted edge gids from a local exscan.** Unmasked by
defect 1's fix: `empty-mask` then failed at **np3, np4 and np5** on the topology
**checksum** alone — every count, `euler`, conformity, closure-inertness and
`closureInverse` held. Diagnosis, and the reason this is worth reading: only the
**edge** gids moved, and by an exactly explicable amount. Step 3f gid'd edges from an
`MPI_Exscan` over `nLocalE` — each rank's *local* edge count, which includes the
boundary edges it holds but does not own (edge owner = min incident face owner, so
the duplicate always sits on the higher rank). Every non-last rank holding such a
duplicate therefore opens a **gap** in the owned gid space, shifting every later
rank's block up: 23 / 44 / 76 gids pushed above the global max at np3 / np4 / np5,
exactly the boundary-edge count of ranks 1..n-2. np1 and np2 pass structurally —
at np2 every duplicate lands on the last rank, where it does no harm.

Worse than the gaps: the two sides of a boundary edge got **different gids for the
same edge** (each from its own rank's block). The old step-3f comment stated this as
if it were fine ("boundary edges carry distinct gids per rank but are matched by
EdgeKey"). It is not fine — `distribute()`, `MeshBuilder`, `migrate()`'s gid-keyed
`eById`/`eOwner` maps and the HDF5 writer/reader all treat an edge gid as a *global*
identifier, and vertex and face gids already are one. Edges were the only kind whose
gids were locally minted.

Fixed in the library, **for free**: each `EdgeKey` is routed to exactly one
coordinator, so the coordinator numbers its own distinct keys densely from an
`MPI_Exscan` over its key count and returns the gid in the reply of the round trip
step 3e already performs for ownership and level. One extra field on `EdgeOwnMsg`,
no extra message round. Refined edge gids are now dense in `[0, globalEdges)` and
identical on every rank holding the edge. Recorded as **Decision 10** in
`tasks/conforming-refinement.md`, and `docs/design.md`'s refine section now
describes the coordinator as assigning the gids.

**One assertion got stronger, deliberately.** The gid half of `empty-mask` was a
global **XOR** of owned gids (`topologyChecksum`), which is permutation-invariant and
cancels: at np4 and np5 it was **passing while the gid set was in fact wrong** —
shifting a contiguous, even-sized block by a fixed amount can cancel in XOR. It now
also compares the globally gathered, sorted owned-gid **set** per kind, and on
failure prints which kind and which gids were lost/gained. `case_empty` also reports
*which* condition failed (a named bit per condition) instead of only bumping a
count; that is what turned this from "np3 fails somehow" into a one-line diagnosis.
Had the weak checksum not been strengthened, defect 2 would have shipped hidden at
two of the five gate rank counts.

**Result** (final probe job `f3QJwCBao2jR`; gate job `f3QJgiXFh3Ef`):
`refine_conforming` `exit=0` at np1–5 × {SERIAL, HIP}, all three adaptive rounds
`euler=2`, plus empty-mask and uniform-mask. Non-vacuity intact — the
hanging-node control still fails conformity every round with growing defect counts
(414 → 1031 → 1474 at np1). The adaptive-round numbers are **byte-identical before
and after the edge-gid change** at every rank count, which is the cleanest evidence
that the change moved gids only and not a single refinement decision.

**No regressions.** The full gate (`regression` × {SERIAL, HIP} × ranks 1–5) was
run rather than spot-checks, because the edge-gid change is in the path *both* modes
share: 130 instances, everything green through test #172 except the three
never-before-executed conforming tests recorded under *Open failures*, whose
failures were shown to be pre-existing by rebuilding with the change stashed.
`loadbalance`, `io`, `markquality_edge` and `markquality_curv` — all of which
consume edge gids — pass at ranks 1–5 on both backends for the first time.

---

## D4 — Re-sweep and get a first verdict on the nine never-executed tests

**Status: mostly done, as a side effect of D3's gate run** (job `f3QJgiXFh3Ef`).
Seven of the nine now have a verdict: `loadbalance`, `io`, `markquality_edge` and
`markquality_curv` **pass** at ranks 1–5 on both backends; `conforming_migrate`,
`conforming_operators` and `conforming_determinism` fail as tabulated under *Open
failures*. Still unexecuted: **`conforming_quality` and `markquality_conforming`**
(the gate window ended at instance #172 of 177, and `conforming_quality` is `unit`,
so it is not in the gate at all — run `ctest -L unit` for it).

Remaining D4 work is therefore just those two, plus splitting the confirmed
failures across D5/D6, which the table above already does.

Nine registrations have **never been executed**: `conforming_migrate`,
`loadbalance`, `io`, `markquality_edge`, `markquality_curv`,
`conforming_operators`, `conforming_determinism`, `conforming_quality`,
`markquality_conforming`. Until they run, the size of the remaining work is
unknown and D5–D7 cannot be scoped.

**Procedure.** After D1–D3 land, resubmit
`flux batch scripts/tuolumne/run_conforming_tests.flux`. Expect it to need more
than one 20-minute allocation now that it will get further: if it times out
again, either raise `--time-limit` (pdebug's cap permitting) or split it — add a
`-R`/`-E` filter to run only the tests downstream of `refine_conforming`.

Record every verdict in the table under D0 and split the failures across D5–D7.
Do **not** start fixing before the whole list is in hand: Task 8's dependency
ordering exists because one upstream bug manifests as a dozen unrelated
downstream failures, and D2 in particular is expected to change most of these
verdicts on its own.

**What landed.** *(fill in)*

---

## D5 — Downstream conforming tests: `conforming_migrate`, `io`, `markquality_conforming`

**Status:** Not started (blocked on D4 for a verdict).

These three exercise the closure composed with redistribution, persistence and
marking. Their known risk points, from
`tasks/conforming-refinement.md` → Task 8:

- **Risk 3 — face gid collisions across rounds.** Retired parent gids are reused
  by un-close while children are allocated above the global max visible gid.
  Surfaces as duplicate gids in `checkOwnershipPartition` *only after* a
  `migrate()` brings two ranks together, so only in a multi-round test.
- **Risk 4 — non-local vertices named by closure children.** By design a closure
  child may name a vertex gid its rank does not hold until the halo is rebuilt.
  Anything assuming post-refine locality will abort or read garbage.
- **Risk 5 — sibling co-residency.** If `migrate()`'s round S fixup or
  `repairClosureCohesion()` misses a case, two ranks each restore the same
  parent → duplicate face, caught by `checkOwnershipPartition`. Non-vacuity:
  `conforming_migrate` fails if the fixup count is zero at ranks ≥ 2.
- **Risk 7 — I/O field-set mismatch.** Writer and reader disagreeing about the
  two closure datasets fails as a round-trip checksum mismatch, not a read
  error. `io` also asserts red-layer identity and then refines the read-back
  mesh once more.

Also collect here, from the test printouts: the `dest`-fixup counts (adversarial
and Zoltan2's), pre/post `loadBalance` max face count and max weighted load vs
ideal, and the actual on-disk size delta for a conforming vs hanging-node file
(predicted +32 B per owned face).

**What landed.** *(fill in)*

---

## D6 — `conforming_operators` and `conforming_determinism`

**Status:** Not started (blocked on D4).

`conforming_operators` is the payoff test — closed 1-ring fan, stencil topology,
`applyStencil` and `reduceVertexFromFaces` error at closure vs interior vertices
asserted separately. An empty closure-vertex set is a hard failure by design.

`conforming_determinism` has three cases and they need different judgement:

1. **Idempotence** — the sharpest probe of risk point 9, and the one case that
   should flip from failing to passing purely as a result of D2. If it still
   fails after D2, D2 is incomplete.
2. **Rank-count independence** vs an `MPI_COMM_SELF` reference, compared by
   quantised **position**. Read risk point 10 in
   `tasks/conforming-refinement.md` before touching this. If it fails *only* on
   the visible layer while the red layer, the `|S|` histogram, the
   closure-vertex set and V/E/F all agree, and `blueDiagMismatch > 0`, that is
   **not** a bug in the risk-point-2 sense — it is the blue tie-break comparing
   midpoint gids that the `MPI_Exscan` ordered differently at different rank
   counts. That is a **design decision to make and record**: leave it
   (documenting that the visible layer is rank-count dependent up to blue
   diagonals) or replace the tie-break with a geometric rule. No *gid*-based
   rule can be rank-count stable. Record the decision in
   `tasks/conforming-refinement.md`'s *Decision record* either way.
3. **Cross-mode equivalence under a uniform mask** — full topology, not counts.
   D0 already showed the uniform-mask closure is inert at np1
   (`|S| hist=[1280,0,0,0]`, `closureChildren=0`), which is a good sign for this
   case.

**What landed.** *(fill in)*

---

## D7 — `conforming_quality`: calibrate the provisional bounds

**Status:** Not started (blocked on D4, and meaningless before D2 — quality
measured on a mesh that has stopped being conforming measures nothing).

Task 7 registered this as **`unit`, deliberately outside the gate**, with bounds
derived from the patterns' ideal-parent geometry rather than measured: **min
angle ≥ 20°, radius ratio `Q` ≤ 4.0, closure fraction ≤ 0.50** over 8 adaptive
rounds on a shrinking geodesic cap.

Replace them with values justified by the measured per-round data the test
prints, then promote to `regression` at ranks 1–5 **only if** it is stable across
repeated runs. Promoting a test into the gate requires explicit user
confirmation of backends and ranks (`CLAUDE.md`).

**If the measured quality is genuinely unbounded with round count, that is a
design finding, not a tolerance to loosen** — stop, record it in
`tasks/conforming-refinement.md`, and report it. The whole justification for
making the closure transient is that the bound is fixed.

**What landed.** *(fill in)*

---

## D8 — Close out Task 8

**Status:** Not started.

1. `flux batch scripts/tuolumne/run_regression_minset.flux` — the real gate,
   `regression` × {SERIAL, HIP} × ranks **1–5**. Green.
2. `ctest -L unit --output-on-failure` — green.
3. `cmake --build build-tuolumne --target format-check` — clean. Note that
   several *untouched* files (e.g. `src/Tessera_Geometry.hpp`) already violate
   clang-format v21 and that predates this work; check touched files under both
   the v21 (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`) binaries.
4. Recompute the **new test totals** (they were 70/70 regression and 28/28 unit
   before the conforming work) and record them in the *Status* section of
   `tasks/conforming-refinement.md` and in the `CLAUDE.md` task-log row.
5. Decide the **default flip**. Task 7 flipped `Mesh`'s `Mode` default to
   `Conforming` before anything had run. If conforming cannot be made green,
   revert the default to `HangingNode2to1` (keeping conforming opt-in and fully
   built) and record why in README *Known Issues* and under *Open failures*.
   Reverting is preferable to leaving a broken default.
6. Fill in Task 8's *Report back* in `tasks/conforming-refinement.md`, including
   its deferred-measurements table, from the run output.
7. Housekeeping: `tuolumne2150-claude-3638145.core` is an untracked crash dump
   in the repo root from an earlier session — delete it or gitignore `*.core`.
   `.claude/settings.json` and `systems/tuolumne/spack.yaml` also had uncommitted
   modifications at the start of D0; check whether they should land.

**What landed.** *(fill in)*

---

## Findings log

*(append-only)*

- **2026-08-04 — D3.** `refine_conforming` green SERIAL+HIP at **np1–5** (probe
  `f3QJwCBao2jR`, gate `f3QJgiXFh3Ef`). Two defects. (1) The abort was exactly D1's
  re-halo defect — `gid2lv.at()` in Phase 3a needs both endpoint *positions* of every
  midpoint the rank owns and one is a ghost `refine()` already dropped; the round loop
  had no rebuild between rounds. Fixed with the identity-`migrate()` +
  `haloExchange()` idiom on both the conforming mesh and the control. `midGid` was
  never involved. (2) *Unmasked by that fix, pre-existing, library:* `refine()` gid'd
  its re-derived edges from an `MPI_Exscan` over each rank's **local** edge count,
  which includes the boundary duplicates a rank holds but does not own. Two
  consequences — the owned gid space gained a gap per duplicate on any non-last rank
  (an **empty-mask** refine changed the global owned edge-gid set at np3/4/5, by
  exactly 23/44/76 gids; np2 escapes because with two ranks every duplicate lands on
  the last rank), and the two sides of a boundary edge carried **different gids for
  the same edge** — which `distribute()`, `MeshBuilder`, `migrate()`'s gid-keyed edge
  maps and the HDF5 writer/reader all assume they do not, and which vertex and face
  gids already satisfied. Edges were the only kind gid'd locally. Fixed with **no
  extra message round**: each `EdgeKey` reaches exactly one coordinator, so the
  coordinator numbers its keys densely from an exscan over its key count and returns
  the gid in step 3e's existing ownership/level reply (Decision 10). **Trap worth
  remembering:** the check that should have caught this was a global **XOR** of owned
  gids, and XOR *cancels* — it was passing at np4 and np5 while the gid set was
  wrong, because shifting a contiguous even-sized block by a fixed amount is
  XOR-invisible. It now compares the gathered sorted gid **set** per kind and names
  the failing condition and the lost/gained gids; without that strengthening defect 2
  would have shipped hidden at two of the five gate rank counts. *A permutation- and
  cancellation-invariant checksum is not a set comparison — do not trust an XOR to
  detect a structured shift.* Adaptive-round numbers are byte-identical before and
  after the edge-gid change at every rank count (it moved gids, not decisions).
  Regression evidence is the **full gate** rather than spot-checks (both modes share
  the changed path): 130 instances green through #172 except the three
  never-before-executed conforming tests, which were proven pre-existing by
  rebuilding with the change stashed. D4 is now largely answered — `loadbalance`,
  `io`, `markquality_edge`, `markquality_curv` pass at ranks 1–5 both backends;
  `conforming_migrate` / `conforming_operators` / `conforming_determinism` abort at
  np≥2 (plus a np1 idempotence failure) and are D5/D6; only `conforming_quality` and
  `markquality_conforming` remain unexecuted.
- **2026-08-04 — D2.** `refine_conforming` green SERIAL+HIP at **np1** — all three
  adaptive rounds `euler=2` (jobs `f3QJLhrfQLKh`, `f3QJMiKVuhm9`). Risk point 9 was
  real and was **three coupled library defects**, all in the red engine's handling of
  a persistent hanging node, only the first of which Task 7 predicted. (1) The
  closure's split-edge map was this-round-only; `unclose()` now recovers the
  persistent map locally from the closure children (`recoverSplitEdges()`) and it is
  unioned into Phase 2's. Task 7's proposed uniqueness rule is **ambiguous for the
  blue pattern** and needed an "each half-edge belongs to exactly one child"
  qualifier — fan-boundary edges appear once, diagonals twice. (2) *Pre-existing:*
  a coarse face refining across a hanging node minted a fresh midpoint coincident
  with the fine side's, cracking the mesh; the same union makes it reuse the existing
  one, which also means a fresh red child can now legitimately have |S| > 0 on an
  inherited half-edge, so `closeFaces()`'s `freshChild` assertion was narrowed via a
  new `firstNewVertexGid` parameter. (3) *Pre-existing:* both coordinator rules need
  two incident faces for an edge key and a hanging node leaves one on each side, so
  the **2:1 propagation was blind across hanging nodes** and round 3 aborted on
  "bisected more than once"; a persistently split edge is now advertised as its two
  **half-edges** in Phases 1 and 2 (`forEachSubEdge()`). The fixpoint does real work
  for the first time: `it=1/2/3` over the three rounds, was `1/1/1`. **Trap worth
  remembering:** a half must be advertised with `refining = 0` even when the
  advertising face is refining — otherwise the coordinator mints a midpoint for the
  half, and the resulting over-refinement still passes every check for one round
  (`euler=2`, but `F=1494` where the correct answer is `F=1222`) before aborting in
  the next. *A conforming round that passes every check can still be over-refining;
  watch `F`, not just `euler`.* Non-vacuity unaffected (control defects still grow
  414→1031→1474). For D7: closure fraction is flat at 0.438/0.456/0.461 across
  rounds. No regressions in 14 spot-checked instances; `HangingNode2to1` is provably
  untouched (empty persistent map ⇒ both changes are no-ops, and its Phase-2a volumes
  are byte-identical). **D3 is unchanged by all of this** — re-probed, same abort, so
  follow D1's re-halo steer, not the `midGid` steer.
- **2026-08-04 — D1.** `refine_splitedges` green SERIAL+HIP np1-5 (job
  `f3QHvwB5yGRu`). Two test-side defects, the second hidden behind the first.
  (1) The hang: `TesseraTest::globalOwnedFaces( mesh )` — an `MPI_Allreduce` —
  was an *argument to a `printf` guarded by `if ( rank == 0 )`*, so rank 0
  deadlocked against everyone else's round-2 `refine()`. Hoisted above the
  branch. A sweep of every rank-0 block in `tests/` and `examples/` found no
  other instance. (2) Unmasked by that fix, an `std::out_of_range:
  unordered_map::at` in round 2 — **the same exception as D3, but in
  `HangingNode2to1` mode with no closure involved**: `refine()` drops all ghosts
  and clears the halo, yet Phase 3a needs both endpoint *positions* of every
  midpoint the rank owns, and across a partition boundary one of those endpoints
  is a ghost. So `refine()` twice with no rebuild in between throws at np>=2.
  Fixed test-side with the documented identity-`migrate()` + `haloExchange()`
  re-halo idiom (already used by `conforming_determinism` and
  `conforming_quality`); README's Known Issue sharpened to say the consequence
  is a throw, not just an inert `haloExchange()`. **This is very likely all or
  most of D3** — `test_refine_conforming.cpp`'s round loop lacks the same
  re-halo. Side effect: the identity migrate permutes face ordering and hence
  child gid assignment, so gid-derived masks select different faces in later
  rounds (round 3 `localMidsSum` 219 -> 218 at np1) — benign, but gid-masked
  per-round counts are not rank-count invariants. Non-vacuity strengthens with
  rank count (`keptOnlyDiscovered` 0/6/13/24/27 at np1-5); phase2a ratios
  x6.96/x5.03/x10.01/x16.38 for the Task-8 table.
- **2026-08-04 — D0.** Added `scripts/tuolumne/run_conforming_tests.flux` (both
  tiers × {SERIAL, HIP} × np1–4, 100 s per-test timeout, 1 pdebug node, 20 min).
  First execution of any Task 1–7 code. Job `f3QH9jz6D1aw` reached 86/165 before
  its wall limit; job `f3QHQtUmFtpX` re-probed the failures in isolation.
  **Result: 78 test instances pass** — all of Tasks 1 and 2 (`refinement_mode`,
  `refine_closure`), the whole pre-existing suite including `refine_parallel`
  and `migrate_mesh` at np1–4, and `staleslice_guard`. Risk points 1, 6 and 8
  are clear. **Three failures**, all downstream: `refine_splitedges` hangs at
  np≥2 (D1, probably in the test's own collectives — `refine_parallel` shares
  the same Phase 2 and passes), `refine_conforming` fails rounds 2 and 3 at np1
  (D2 — **risk point 9 confirmed exactly as predicted**, round 2's `euler=-136`
  equals round 1's hanging-node control), and `refine_conforming` aborts on rank
  1 at np2 with `std::out_of_range: unordered_map::at` (D3). Nine registrations
  never executed. Also learned three harness traps, documented above; the first
  — a ctest timeout leaving its `flux run` sub-job holding the exclusive node,
  which then times out every subsequent test — turned one real hang into eight
  bogus verdicts and is worth remembering.
