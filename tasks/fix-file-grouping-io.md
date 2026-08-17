# XDMF time-series grouping: one dataset in Paraview instead of N

**Status:** NOT STARTED

## Problem

A downstream solver (Beantik) calls `writeMesh( mesh, stem )` once per output
frame, each call with a different stem. Every call produces `<stem>.h5` plus a
standalone `<stem>.xmf`. In Paraview the result is N unrelated datasets: each
`.xmf` must be opened separately, there is no time slider over the sequence, and
stepping through the mesh evolution means loading and toggling N pipeline
objects by hand.

The cause is not a Paraview setting and not the filenames. The sidecar Tessera
writes carries **no time coordinate and no collection wrapper**, and
`writeMesh()` is handed no time and no series identity, so the library has no
way to state that N files are N timesteps of one thing. Paraview's XDMF readers
are not file-series readers — they will not infer a series from numeric
filenames the way `vtkFileSeriesReader`-backed readers do — so nothing
downstream can recover the grouping. See **Read this first** for the evidence.

End state: a run that writes N frames also writes **one master `.xmf`** which
Paraview opens as a single dataset with N timesteps on the time slider,
animatable, with the same attribute set (`v_gid`, `f_level`, `vu0…`, `fu0…`)
selectable across every step. The per-frame `.h5` layout on disk does not
change, so `readMesh()` keeps working on any individual frame.

**Out of scope**

- Changing the HDF5 layout, `readMesh()`, or `format_version` (still 2). The
  master `.xmf` is pure light data derived from what the writer already emits.
- Writing edges to XDMF. Still deliberately omitted
  ([src/Tessera_Xdmf.hpp:15-21](../src/Tessera_Xdmf.hpp#L15-L21)).
- Any change to the ship gate. The new test is `unit` tier, so the four-location
  gate definition (`docs/testing.md`, `tests/CMakeLists.txt`,
  `scripts/<system>/run_regression_minset.*`, `.github/workflows/ci.yml`) is
  untouched by this work.
- Spatial collections (one grid per rank). Tessera writes one dense global grid
  per frame; there is nothing to collect spatially.
- Changes inside Beantik. This design delivers the library API and updates
  `examples/02_mesh_pipeline` as the reference caller; adopting it in Beantik is
  that project's change.

## Read this first

The brief's framing — "they are not grouped together in paraview" — is accurate
about the symptom but suggests grouping is something that failed. Nothing ever
attempted it. Three specific facts a later session should not re-derive:

1. **There is no `<Time>` element anywhere in the emitted XML, and the grid is a
   lone `GridType="Uniform"`.**
   [src/Tessera_Xdmf.hpp:91-93](../src/Tessera_Xdmf.hpp#L91-L93) writes
   `<Xdmf Version="3.0"><Domain>` then `<Grid Name="Tessera"
   GridType="Uniform">`, and [src/Tessera_Xdmf.hpp:147-148](../src/Tessera_Xdmf.hpp#L147-L148)
   closes it. A reader handed this file has one timeless grid; there is no
   timestep for a time slider to show.

2. **`writeMesh()` cannot know a series exists.** Its signature is
   `writeMesh( const MeshT& mesh, const std::string& stem )`
   ([src/Tessera_HDF5Writer.hpp:77-79](../src/Tessera_HDF5Writer.hpp#L77-L79))
   and it returns `void`. No time argument, no frame index, no state carried
   between calls. The caller invents the stem (`frameStem()` in the example,
   [examples/02_mesh_pipeline/mesh_pipeline.cpp:107-114](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L107-L114)),
   so the *only* place in the system that knows which files form a sequence, and
   in what time order, is the caller. That is why the fix necessarily adds a
   caller-facing series object rather than being a pure change inside
   `writeXdmf()`.

3. **Paraview will not group the files for us.** The idiomatic and reliable XDMF
   mechanism is a temporal collection —
   `<Grid GridType="Collection" CollectionType="Temporal">` with one child
   `<Grid>` per timestep, each carrying its own `<Time Value="…"/>` and its own
   full `<Topology>`/`<Geometry>` — which is exactly what an adaptively
   remeshing series needs, since `Nv`/`Nf` differ frame to frame. Paraview's
   generic `.series` JSON sidecar is supported only by readers that declare
   file-series support, which the XDMF readers do not. A child grid with no
   `<Time>` has been reported to crash Paraview, so the `<Time>` element is
   mandatory, not decorative.

A fourth observation, deliberately **not** acted on by this design: the emitted
XML declares `Xdmf Version="3.0"` while using XDMF2 element spellings
(`TopologyType=`, `GeometryType=`, `NumberOfElements=`, `NumberType=`/
`Precision=`, e.g.
[src/Tessera_Xdmf.hpp:99-113](../src/Tessera_Xdmf.hpp#L99-L113)). This currently
works — the readers accept the legacy spellings — and changing it is a separate
risk with no bearing on grouping. It is why **V1 must confirm the master file in
more than one Paraview reader** (see R2).

## Approach

Keep the frame files exactly as they are and add a second, higher-level artifact
that names them in time order.

Three pieces, in the same header-only style as the rest of the library:

1. **A frame record.** The information `writeXdmf()` currently receives as seven
   loose arguments becomes one `XdmfFrame` struct, and `writeMesh()` **returns**
   it. Returning it is what makes the series possible without duplicating the
   writer's metadata derivation (`gcV.N`, `gcF.N`, the `vXdmf`/`fXdmf` field
   lists built at [src/Tessera_HDF5Writer.hpp:245](../src/Tessera_HDF5Writer.hpp#L245)
   and [src/Tessera_HDF5Writer.hpp:356](../src/Tessera_HDF5Writer.hpp#L356)) in a
   second place, where it would silently drift from the writer.

2. **A stateless master-file emitter.** `writeXdmfSeries( masterStem, steps )`
   takes a vector of `{ XdmfFrame, double time }` and writes the temporal
   collection. Stateless and MPI-free, so it is directly testable without
   building a mesh, and a caller that keeps its own frame bookkeeping can call it
   directly.

3. **A caller-held series handle.** `MeshSeries` accumulates frames and rewrites
   the master after every frame, so the master on disk always describes the
   frames that actually exist. A run killed at frame 40 leaves a master with 40
   timesteps, not a missing or truncated file.

### Why the master is rewritten every frame, not once at the end

An `.xmf` temporal collection is not append-friendly: the closing
`</Grid></Domain></Xdmf>` must move. The alternatives were (a) leave the file
unterminated until finalize — a crashed run then leaves invalid XML that
Paraview refuses outright, which is worse than the status quo; (b) write the
master only in a finalize call — a crashed or killed run yields no master at
all, and long HPC runs are killed routinely. Rewriting is O(frames) text on rank
0 per frame, immaterial against a collective HDF5 write (R4). The rewrite goes
to `<master>.xmf.tmp` and is `std::rename`d over the real path, so a Paraview
reload never observes a half-written file.

### Why frame stems stay caller-owned

`MeshSeries::write()` takes the frame stem from the caller rather than
generating `<base>_%06u`. The master lists each frame's file explicitly, so
frame names need not be numeric or even ordered — and Beantik keeps whatever
naming it already uses. The cost is one real constraint that must be enforced
loudly, not worked around: `writeXdmf()` references the HDF5 file by **basename**
([src/Tessera_Xdmf.hpp:40-44](../src/Tessera_Xdmf.hpp#L40-L44), used at
[src/Tessera_Xdmf.hpp:85](../src/Tessera_Xdmf.hpp#L85)), so an `.xmf` can only
reference `.h5` files sitting beside it. The master must therefore live in the
same directory as every frame it names, and `MeshSeries` hard-fails on a frame
stem whose directory differs from the master's.

### Conventions

| Choice | Decision |
|---|---|
| Header | New public API in a new header `src/Tessera_XdmfSeries.hpp`; add it to `src/Tessera.hpp` in alphabetical position (after `Tessera_Xdmf.hpp`, [src/Tessera.hpp:52](../src/Tessera.hpp#L52)) |
| Namespace | `Tessera::` for `XdmfFrame`, `XdmfTimeStep`, `XdmfField`, `writeXdmf`, `writeXdmfSeries`, `MeshSeries`. `XdmfField` **moves out** of `Tessera::detail` because it is now reachable through a public return type; its only two current uses are in `Tessera_Xdmf.hpp` and [src/Tessera_HDF5Writer.hpp:197](../src/Tessera_HDF5Writer.hpp#L197) |
| Time argument | Always an explicit `double time` from the caller. No default, no frame-index fallback, no NaN sentinel. A caller with no physical time passes the step index — an explicit choice at the call site |
| Optional time | Expressed as an **overload pair**, not a pointer or sentinel: `writeXdmf( stem, frame )` emits no `<Time>`; `writeXdmf( stem, frame, time )` emits one. Same for the two `writeMesh` overloads |
| Failure behavior | Loud. `MeshSeries::write()` throws `std::runtime_error` on a non-increasing time or a frame stem in a different directory from the master. `writeXdmfSeries()` throws if it cannot open the temp file or if `steps` is empty. No best-effort partial master |
| Time formatting | `%.17g`, so a `double` round-trips exactly through the XML |
| Child grid names | Every child keeps `Name="Tessera"`, as today — a stable grid name across timesteps is what lets Paraview track the same object through the collection |
| Comments | Units/ownership/monotonicity contracts on the declarations. Cite the XDMF temporal-collection shape as the provenance on `writeXdmfSeries()` |
| Test tier | `unit`, `SERIAL`, ranks `1;2;3` — new file `tests/test_xdmf_series.cpp`. The gate is untouched. Assertions are on XML text plus file existence; a Python `xml.etree` well-formedness check runs from the test via `python3` (present at `/usr/tce/bin/python3`; `pvpython` and `paraview` are **not** installed on this system, so no in-test Paraview check is possible) |
| License header | Every new file carries the project BSD-3-Clause `/* */` block with `SPDX-License-Identifier: BSD-3-Clause`, per `CLAUDE.md` |
| Formatting | Never run the formatter. Match surrounding style by hand |

### Deliberate deviations

- **Per-frame `.xmf` sidecars are kept.** The series writes a master *and* leaves
  each frame's own sidecar in place, so a single frame stays individually
  inspectable and the existing single-shot `writeMesh( mesh, stem )` behavior is
  unchanged for every current caller. The consequence is that the output
  directory still contains N+1 openable `.xmf` files; the master is the one to
  open, and it is the only one whose name has no frame index in it. This was
  chosen explicitly over suppressing sidecars in series mode.
- **No XInclude / XPointer reuse of topology between child grids.** Each child
  grid repeats its full `<Topology>` and `<Geometry>` pointing at its own `.h5`.
  This is correct for adaptive meshes (topology genuinely changes) and is the
  path with the fewest reader bugs; hoisting shared data into the `<Domain>` is
  reported to break. The XML is O(frames) but each child is ~10 lines of text
  referencing HDF5 — heavy data never enters the XML.
- **`writeMesh()`'s return type changes from `void` to `XdmfFrame`.** This is
  source-compatible with every existing caller (a discarded return value), so no
  caller is modified by T1 — but it is a change to a documented public signature
  and must be mirrored into `README.md` and `docs/design.md`.

## Current state

Everything described here is unbuilt. What exists:

- `writeMesh( const MeshT&, const std::string& )` → `void`, writes `<stem>.h5`
  collectively and then, on rank 0 after an `MPI_Barrier`, the sidecar
  ([src/Tessera_HDF5Writer.hpp:408-413](../src/Tessera_HDF5Writer.hpp#L408-L413)).
- `writeXdmf( stem, dim, scalarBytes, Nv, Nf, vFields, fFields )` →
  one timeless `GridType="Uniform"` grid
  ([src/Tessera_Xdmf.hpp:80-150](../src/Tessera_Xdmf.hpp#L80-L150)). Note it
  **returns silently if `fopen` fails** ([src/Tessera_Xdmf.hpp:87-89](../src/Tessera_Xdmf.hpp#L87-L89));
  that is a pre-existing quiet-degradation path this design does not fix, and
  the new emitter must not copy it.
- `detail::XdmfField { dataset, name, extent }`
  ([src/Tessera_Xdmf.hpp:33-38](../src/Tessera_Xdmf.hpp#L33-L38)), populated by
  the writer's user-field loops.
- No time concept, no collection, no series type, no master file, nothing that
  reads a written frame's metadata back for a restart.
- `examples/02_mesh_pipeline` writes N independent frames
  ([mesh_pipeline.cpp:155](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L155),
  [mesh_pipeline.cpp:221](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L221)) —
  it reproduces the reported symptom exactly and is the natural demonstration
  vehicle.
- `tests/test_io.cpp` exercises the round trip and deletes `<stem>.xmf`
  ([test_io.cpp:294-295](../tests/test_io.cpp#L294-L295),
  [test_io.cpp:523-524](../tests/test_io.cpp#L523-L524)) but asserts nothing
  about its contents. The XML has no test coverage at all today.

**Not read, deliberately:** nothing in Paraview's reader sources, and no Beantik
code (not in this repository). T4 is the first task that reads HDF5 root
attributes back from the series layer; earlier tasks must not open that path —
`MeshSeries` in T2 keeps every frame's metadata in memory and needs no reads.

## Progress log

`tasks/fix-file-grouping-io-progress-log.md`. Read it before implementing any
task, before changing a signature this design states, and before reopening a
question the design treats as settled — a completed task may have changed the
plan for a later one, and the `**Affects:**` line of each entry is the index to
that.

## Task sequence

### T1 — `XdmfFrame` record; `writeMesh()` returns it; byte-identical output — **NOT STARTED**

**Depends on:** none

**Fill in:**
- `src/Tessera_Xdmf.hpp` — move `XdmfField` from `Tessera::detail` to `Tessera`;
  add `struct XdmfFrame`; split the grid body out of `writeXdmf()` into
  `detail::writeXdmfGrid( fp, frame, indent )` and an overload
  `detail::writeXdmfGrid( fp, frame, indent, double time )`; replace the 7-arg
  `writeXdmf()` with `writeXdmf( const std::string& stem, const XdmfFrame& )`
  and `writeXdmf( const std::string& stem, const XdmfFrame&, double time )`.
- `src/Tessera_HDF5Writer.hpp` — extract the whole existing body except the
  final sidecar block into `detail::writeMeshH5( mesh, stem ) -> XdmfFrame`;
  public `writeMesh( mesh, stem )` and `writeMesh( mesh, stem, double time )`
  both call it, barrier, and emit the matching sidecar on rank 0, returning the
  frame. Update the `detail::XdmfField` use at line 197.

**Reference:** the exact bytes to preserve are
[src/Tessera_Xdmf.hpp:91-148](../src/Tessera_Xdmf.hpp#L91-L148); the metadata to
capture in `XdmfFrame` is the current argument list at
[src/Tessera_Xdmf.hpp:80-83](../src/Tessera_Xdmf.hpp#L80-L83) plus the
`h5name` basename computed at [src/Tessera_Xdmf.hpp:85](../src/Tessera_Xdmf.hpp#L85).

**Do:**
1. Define `XdmfFrame { std::string h5name; int dim; int scalarBytes; unsigned long long Nv, Nf; std::vector<XdmfField> vFields, fFields; }`. `h5name` is the **basename** of the HDF5 file (documented on the member: an `.xmf` can only reference `.h5` files in its own directory).
2. Factor the grid emitter with an `indent` parameter — the master nests one level deeper than a standalone file, and this is the only reason the parameter exists. Say so in a comment.
3. Keep the `dim != 2 && dim != 3` attribute-only branch ([src/Tessera_Xdmf.hpp:115-121](../src/Tessera_Xdmf.hpp#L115-L121)) verbatim inside the emitter.
4. Do not change any call site other than [src/Tessera_HDF5Writer.hpp:411-413](../src/Tessera_HDF5Writer.hpp#L411-L413) — `writeXdmf` has no other callers in `src`, `tests`, or `examples`, and the discarded `writeMesh` return value keeps `test_io.cpp:226`, `test_io.cpp:458`, `test_latlon_sphere.cpp:709`, `test_distributed_build.cpp:697`, `mesh_pipeline.cpp:155`, and `mesh_pipeline.cpp:221` compiling untouched.
5. Mirror the new `writeMesh` signature into [README.md:118](../README.md#L118) and [docs/design.md:1538-1545](../docs/design.md#L1538-L1545).

**Exit criterion:** save a copy of an `.xmf` produced before the change, then
`ctest -L regression -R "io_SERIAL"` passes at np1-5 and the newly produced
`.xmf` for the same case is **byte-identical** (`cmp` exits 0) to the saved copy.
Failure direction: with the timed overload called instead, `cmp` must report a
difference and the only difference must be an added `<Time Value=` line.

**Session prompt:**

`````markdown
Read `tasks/fix-file-grouping-io.md` and implement `T1` — introduce `XdmfFrame`,
have `writeMesh()` return it, and keep the emitted `.xmf` byte-identical.

Read these before starting, and skip the rest of the document:
- `tasks/fix-file-grouping-io.md`: the `T1` task entry, the **Conventions** table,
  the **Deliberate deviations** section, and risk `R3` (T1's whole reason for
  factoring the grid emitter into one function is so R3's fix is one-function-wide
  later).
- `src/Tessera_Xdmf.hpp` — the whole file (155 lines); `detail::XdmfField` at 33-38,
  `writeXdmf()` at 80-150, the exact bytes to preserve at 91-148.
- `src/Tessera_HDF5Writer.hpp:77-79` (the `writeMesh` signature), `:197` (the only
  `detail::XdmfField` use), `:245` and `:356` (the `vXdmf`/`fXdmf` field-list
  construction), `:407-413` (the barrier + rank-0 sidecar block to lift out).

`tasks/fix-file-grouping-io-progress-log.md` has no entries yet; yours is the first.

Decisions already made — do not reopen, and record them in
`tasks/fix-file-grouping-io-progress-log.md` under `## T1`:
- **The byte-comparison vehicle is `examples/02_mesh_pipeline`, not `test_io`.**
  `test_io.cpp` deletes its `.xmf` at lines 294-295 and 523-524, so it leaves
  nothing to `cmp`. The example leaves one `.xmf` per frame on disk.
- **Establish determinism before attributing any difference to your change.** In
  the baseline job, run the example twice into two different `--out` stems and
  `cmp` the corresponding frame `.xmf` files against each other. If those differ,
  stop and report — the byte-identity criterion is not measurable and the task
  needs a different vehicle.
- **The failure-direction check is temporary and reverted.** To prove the timed
  overload adds exactly one `<Time Value=` line and nothing else, temporarily
  switch one `writeMesh()` call in `mesh_pipeline.cpp` to the timed overload,
  capture the `cmp`/`diff` output, then revert that edit. T3 owns the permanent
  example change; do not leave a timed call site behind.
- **Scope of test running:** the full regression gate is not required. Build all
  targets — compilation is what proves the `void` → `XdmfFrame` return change is
  source-compatible with `test_io.cpp:226`, `test_io.cpp:458`,
  `test_latlon_sphere.cpp:709`, `test_distributed_build.cpp:697`,
  `mesh_pipeline.cpp:155` and `:221` — then run only `io_SERIAL` np1-5. `io_HIP` is
  not required: the sidecar is rank-0 text with no device involvement.

Constraints specific to this task:
- Do not change any call site other than `src/Tessera_HDF5Writer.hpp:411-413`.
- Keep the `dim != 2 && dim != 3` attribute-only branch
  (`src/Tessera_Xdmf.hpp:115-121`) verbatim inside the new emitter.
- `XdmfField` moves out of `Tessera::detail` into `Tessera`; `XdmfFrame` is new in
  `Tessera`. No new header in T1 — `src/Tessera_XdmfSeries.hpp` is T2's.
- Do not fix the silent `return` on `fopen` failure at `src/Tessera_Xdmf.hpp:87-89`.
  It is pre-existing and out of scope; T2's new emitter is where loud failure lands.
- Never run the formatter. Match surrounding style by hand.
- Out of scope, deliberately: `writeXdmfSeries()`, `MeshSeries`, any `<Time>`
  emission from a permanent call site, and the `Version="3.0"`-with-XDMF2-spellings
  mismatch (R3 — V1 decides that, not you).

Running on the cluster — this machine uses Flux:

- Do not run executables directly. No `flux run`, no bare `mpirun` or `srun`, and
  no invoking the binary or `ctest` on the login node — not a single-rank smoke
  test, not `--help`. Everything that executes goes in a job script submitted to
  the `pdebug` queue. Building is the exception: build on the login node.
- Build in a **login shell** with the Spack user config exported, per
  `systems/tuolumne/claude.md:52-59`:
  ```bash
  bash -lc 'export SPACK_USER_CONFIG_PATH=$HOME/.spack/tuolumne
            . /usr/WS2/stewartj/spack/share/spack/setup-env.sh
            spack env activate ~/spack_envs/tuolumne_trilinos/
            cd /g/g20/stewartj/research-bridges/tessera-dev/Tessera/build-tuolumne
            make -j 64'
  ```
  A non-login shell has no PrgEnv module, so the Cray CC wrapper fails; and without
  `SPACK_USER_CONFIG_PATH` the Tuolumne system externals silently disappear.
- Export `TESSERA_REPO` in the submitting shell before every `flux batch` —
  `export TESSERA_REPO=$(pwd)` from the repo root. Flux copies the script to a temp
  dir, so the resolver cannot locate the repo from its own path, and the committed
  runners die in seconds with an `unbound variable` error without it.
- Submission preambles are already written; copy them, do not invent one. Queue
  `pdebug`, `--nodes=1 --exclusive`, `-t 20m`, resolver sourced via
  `scripts/lib/tessera_env.sh`:
  - example runs: `scripts/tuolumne/run_mesh_pipeline_example.flux` — takes
    `mesh_pipeline` args straight through, output at
    `tessera-mesh-pipeline-example.<jobid>.out` in the submit directory.
  - ctest runs: `scripts/tuolumne/run_unit_tests.flux <LABEL> [CTEST_ARGS...]` —
    for this task `flux batch scripts/tuolumne/run_unit_tests.flux regression -R "io_SERIAL"`,
    output at `tessera-unit.<jobid>.out`.
  Request the shortest walltime the run needs; `pdebug` allows more, and a short
  limit makes a hang fail fast instead of holding an allocation.
- Submit, then poll every 30 seconds until the job leaves the queue, and continue
  the moment it does. Do not sleep for the walltime.
  ```bash
  jobid=$(flux batch scripts/tuolumne/run_unit_tests.flux regression -R "io_SERIAL")
  while flux jobs "$jobid" 2>/dev/null | grep -q "$jobid"; do sleep 30; done
  flux job status "$jobid"
  ```
- Leaving the queue is not success. Check the exit status above and read the job's
  stdout at the paths named before concluding anything. A job killed at the
  walltime limit or lost to a node failure disappears from the queue exactly like
  one that passed.
- If the job is still pending after ten minutes, stop polling and report the queue
  state instead of waiting silently.
- Cancel any job you started and are no longer waiting on (`flux cancel <jobid>`)
  before you finish.
- Copy the rank-to-GPU launch line from `scripts/tuolumne/run_mesh_pipeline_example.flux`
  exactly — `flux run --ntasks 4 --nodes=1 --exclusive --cores-per-task=16
  --env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608`. Do not simplify it
  and do not leave it to the default; Tuolumne has 4 GPUs per node, so a wrong
  binding oversubscribes one device and still returns plausible output. If you use
  a different rank count, say what the binding then was in the log entry.
- Have any job script you write echo its provenance to stdout before doing work —
  `spack env status`, the compiler version, the commit SHA, and the submit command.
  The committed runners already echo system, build dir and args; add the rest.
- Stop and report rather than work around. If the build fails twice for the same
  reason, or a job dies twice the same way, write up what you tried and what the
  error was, and stop. Do not loosen the byte-identity check, skip the
  failure-direction case, or drop rank counts — that silently changes what **DONE**
  means and the substitution is invisible in the diff.

Order of operations, because the baseline must be captured before you edit:
1. Build at current `HEAD`, submit `run_mesh_pipeline_example.flux --iters 1` twice
   with distinct `--out` stems, and save the resulting `.xmf` files outside the
   run directory. Confirm run-to-run byte-identity.
2. Implement T1. Rebuild. Re-run the example with the same args and the original
   `--out` stem, and `cmp` each frame `.xmf` against its saved baseline.
3. Do the temporary timed-overload check, then revert it.
4. Submit `io_SERIAL` at np1-5.

Exit criterion: `ctest -L regression -R "io_SERIAL"` passes at np1-5 and the newly
produced `.xmf` for the same case is **byte-identical** (`cmp` exits 0) to the saved
copy. Failure direction: with the timed overload called instead, `cmp` must report a
difference and the only difference must be an added `<Time Value=` line.

When done: mark `T1` **DONE** in `tasks/fix-file-grouping-io.md` with a **Met.**
paragraph stating what was actually verified — including the rank counts run and
which `.xmf` files were compared — and append a `## T1` section to
`tasks/fix-file-grouping-io-progress-log.md` covering the decisions above, the final
shape of `XdmfFrame` and both `writeXdmf`/`writeMesh` overload pairs, any bug only
running revealed, and an `**Affects:**` line naming the later task IDs your findings
change (T2 consumes `detail::writeXdmfGrid` and `XdmfFrame` directly, so any
departure from their stated signatures belongs there). Delete this
`**Session prompt:**` block from the `T1` entry in the same edit; it describes work
that is now finished.
`````

### T2 — `writeXdmfSeries()` + `MeshSeries`: a master `.xmf` Paraview opens as one timestepped dataset — **NOT STARTED**

**Depends on:** T1

**Fill in:**
- `src/Tessera_Xdmf.hpp` — `struct XdmfTimeStep { XdmfFrame frame; double time; }`
  and `void writeXdmfSeries( const std::string& masterStem, const std::vector<XdmfTimeStep>& steps )`.
- `src/Tessera_XdmfSeries.hpp` (new) — `class MeshSeries`.
- `src/Tessera.hpp` — add the include.
- `tests/test_xdmf_series.cpp` (new), `tests/CMakeLists.txt` — one `unit`,
  `SERIAL`, `RANKS 1;2;3` registration, following the block shape at
  [tests/CMakeLists.txt:537-550](../tests/CMakeLists.txt#L537-L550).

**Reference:** the master's required shape —
`<Grid Name="Tessera" GridType="Collection" CollectionType="Temporal">`
wrapping one full child `<Grid>` per step, each with its own
`<Time Value="…"/>`, `<Topology>`, `<Geometry>` and the identical attribute set.
Cite this shape and the "`<Time>` is mandatory or Paraview can crash" fact on
`writeXdmfSeries()` as provenance.

**Do:**
1. `writeXdmfSeries()`: throw on empty `steps`; open `masterStem + ".xmf.tmp"`, throw `std::runtime_error` naming the path if `fopen` fails (do **not** copy the silent-return at [src/Tessera_Xdmf.hpp:87-89](../src/Tessera_Xdmf.hpp#L87-L89)); emit the collection reusing `detail::writeXdmfGrid( fp, s.frame, indent, s.time )` per step; `fclose`; `std::rename` the temp over `masterStem + ".xmf"`, throwing if the rename fails.
2. `MeshSeries`: `explicit MeshSeries( std::string masterStem )`. Members: master stem, `std::vector<XdmfTimeStep>`, last time. Method `template <class MeshT> void write( const MeshT& mesh, const std::string& frameStem, double time )` — collective on `mesh.comm()`, documented as such.
3. `write()` order: validate, then `XdmfFrame f = writeMesh( mesh, frameStem, time )` (collective; also writes the per-frame sidecar), then on rank 0 only append the step, append one `"<stem> <time>\n"` line to `masterStem + ".xmfindex"` (flushed each frame — T4 consumes it; write it now so a run predating T4 is still restartable), and call `writeXdmfSeries()`.
4. Validation, on **every** rank before any I/O so the throw is symmetric and cannot deadlock: `time` strictly greater than the previous frame's (message naming both values); the directory component of `frameStem` equal to that of the master stem (message naming both, and stating why — an `.xmf` references `.h5` by basename).
5. Accessors: `std::size_t numFrames() const`, `const std::string& masterStem() const`.
6. Document in `README.md` (a `MeshSeries` snippet in the I/O part of Usage/API near [README.md:118](../README.md#L118)) and in the Parallel I/O section of `docs/design.md` ([docs/design.md:1528-1580](../docs/design.md#L1528-L1580)): what the master is, that it must sit beside the frames, and that the frame `.h5` layout is unchanged so `readMesh()` still reads any single frame.

**Exit criterion:** `ctest -L unit -R xdmf_series_SERIAL` passes at np1, 2 and 3.
The test builds a small icosphere, distributes it, writes three frames through
one `MeshSeries` at times `0.0, 0.5, 1.25`, and asserts on rank 0: the master
`.xmf` exists and `<master>.xmf.tmp` does not; it contains exactly one
`CollectionType="Temporal"`, exactly three `<Time Value=` occurrences whose
parsed values are `0.0, 0.5, 1.25` in that order, exactly three `<Topology`,
exactly three `<Geometry`, and one `Format="HDF"` reference per frame naming
that frame's basename; each named `.h5` exists; `python3 -c "import
xml.etree.ElementTree as E; E.parse(...)"` exits 0 on the master.
Failure direction, both asserted to throw `std::runtime_error` and to leave the
previously written master byte-unchanged: a fourth `write()` at time `1.25`
(non-increasing), and a `write()` whose frame stem is in a subdirectory of the
master's directory.

### T3 — `mesh_pipeline` writes a series, so the reported symptom is gone from the shipped example — **NOT STARTED**

**Depends on:** T2

**Fill in:** `examples/02_mesh_pipeline/mesh_pipeline.cpp` (the `run()` frame
writes at [lines 155](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L155) and
[221](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L221), and the
`frameStem()` helper at [lines 107-114](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L107-L114));
`README.md` ([line 572](../README.md#L572), the `mesh_pipeline` row).

**Do:**
1. Construct one `MeshSeries` per `run()` invocation, master stem
   `<out>_<tag>_np<size>` — i.e. `frameStem()` minus the `_frame<i>` suffix, so
   the master sits beside the frames as required and each
   backend/rank-count/refine-mode combination gets its own series.
2. Replace both `writeMesh()` calls with `series.write( mesh, frameStem(...), time )`.
   The example has no physical time: pass the frame index as a `double` and say
   so in a comment, since that is exactly the explicit-caller-choice the API
   requires.
3. Keep the existing per-frame `printf` lines; add one rank-0 line at the end of
   `run()` naming the master `.xmf` to open in Paraview.
4. Update the `mesh_pipeline` README row: it now writes one master `.xmf` plus
   per-frame `.h5`/`.xmf`, and the master is what to open.

**Exit criterion:** running the example (Tuolumne: `scripts/tuolumne/run_mesh_pipeline_example.flux`;
see [docs/build-and-run.md](../docs/build-and-run.md)) with `--iters 3` produces
exactly one `<out>_<tag>_np<size>.xmf` per execution-space tag, each containing 4
`<Time Value=` entries (frame 0 plus 3 iterations) with values `0,1,2,3`, and the
run exits 0. Failure direction: an `.xmf` naming an `.h5` that does not exist on
disk must be absent — assert every `Format="HDF"` reference in the master
resolves to an existing file in the master's directory.

### T4 — Restart: reopen an existing series and keep appending — **NOT STARTED**

**Depends on:** T2. Independent of T3; may be deferred indefinitely without
blocking anything else.

**Fill in:** `src/Tessera_XdmfSeries.hpp`, `tests/test_xdmf_series.cpp`.

**Do:**
1. Add a reopen constructor distinguished by a tag type, not a bool:
   `struct ReopenSeries {}; MeshSeries( std::string masterStem, ReopenSeries )`.
2. Read `<masterStem>.xmfindex` line by line for `(frameStem, time)`. Throw if
   the index is missing, malformed, or its times are not strictly increasing —
   a silently truncated series is exactly the quiet-degradation this project
   forbids.
3. Rebuild each `XdmfFrame` by reopening `<frameStem>.h5` **serially on rank 0**
   (`H5Fopen`, `H5P_DEFAULT` — not MPI-IO) and reading the root attributes with
   the existing helpers `detail::readIntAttr` / `detail::readU64Attr`
   ([src/Tessera_IoCommon.hpp:308-323](../src/Tessera_IoCommon.hpp#L308-L323)):
   `dim`, `scalar_bytes`, `Nv`, `Nf`, `n_user_v_fields` / `n_user_f_fields`, and
   `uv_ext_<j>` / `uf_ext_<j>` for the extents. Reconstruct the field display
   names by the same rule the writer uses — `"v" + "u<j>"` and `"f" + "u<j>"`
   ([src/Tessera_HDF5Writer.hpp:245](../src/Tessera_HDF5Writer.hpp#L245),
   [src/Tessera_HDF5Writer.hpp:356](../src/Tessera_HDF5Writer.hpp#L356)). Do not
   validate against the mesh template here: this constructor takes no mesh.
4. Rewrite the master immediately from the reconstructed steps, so a reopen that
   succeeds is visible on disk before any new frame is written.

**Additional information needed:** whether Beantik restarts by re-running from
frame 0 or by resuming mid-series. If it always re-runs from 0, this task is
unnecessary and should be closed rather than implemented — ask before starting.

**Exit criterion:** `ctest -L unit -R xdmf_series_SERIAL` passes at np1-3 with an
added case that writes two frames through one `MeshSeries`, destroys it,
constructs a second with `ReopenSeries`, writes a third frame, and asserts the
master then holds three `<Time Value=` entries in increasing order and that the
first two child grids are byte-identical to those the first object wrote.
Failure direction: reopening with the `.xmfindex` deleted, and with its two time
lines swapped into decreasing order, each throws `std::runtime_error` naming the
index path.

### V1 — Confirm in Paraview, in both reader families — **NOT STARTED**

**Depends on:** T2 (T3 gives a larger, more convincing dataset). Isolated
because it may block: neither `paraview` nor `pvpython` is installed on this
system, so this needs a machine with a Paraview install and cannot run in CI.

**Do:** open a T2- or T3-produced master `.xmf` in the Paraview GUI. Where the
open dialog offers a reader choice, verify **both** the XDMF3 temporal reader
(`Xdmf3ReaderT` — the variant that walks a temporal collection; `Xdmf3ReaderS`
is the single-file/spatial one) and the legacy `XDMFReader` (Xdmf2). For each:
confirm the Information tab lists N timesteps, the time slider animates through
all N, the surface renders at every step, and `v_gid` / `f_level` / `vu0` /
`fu0` are all selectable and change with the step. Record which readers worked,
with versions, in the progress log.

**Exit criterion:** a progress-log entry naming the Paraview version, each
reader tried, and for each the observed timestep count (must equal the number of
frames written) and whether all attributes coloured correctly. Failure
direction: if a reader shows 1 timestep, or 0, or errors, the entry records
which, and R2 or R3 becomes an open task rather than V1 being marked DONE.

## Known risks

**R1 — Attribute set drifts between frames.** Paraview flags attributes missing
at some timesteps as "partial", and a reported failure mode is the *first*
scalar in the file returning garbled values pulled from other arrays. Tessera's
attribute names come from the compile-time field pack, so within one run they
are constant — but two runs with different mesh template parameters must never
share a master. *Presents as:* correct geometry, wrong or flickering field
values, often only for one array. *Do:* `MeshSeries` never mixes frames from
different mesh types by construction (one object per `run()` in T3); if this is
ever seen, first diff the `<Attribute Name=` sets across child grids in the
master — they must be identical, in the same order.

**R2 — The wrong Paraview reader is used.** Paraview offers up to three XDMF
readers, and only the temporal XDMF3 reader walks a temporal collection. *
Presents as:* the master opens showing exactly **one** timestep — visually very
close to the symptom being fixed. *Distinguishing measurement:* count
`<Time Value=` in the master with `grep -c`. If the file has N and Paraview
shows 1, this is R2 (a reader-selection issue, documented in the README, not a
code bug); if the file itself has fewer than N, the bug is in
`writeXdmfSeries()` or in `MeshSeries`'s step accumulation.

**R3 — The `Version="3.0"` declaration with XDMF2 element spellings breaks the
XDMF3 temporal reader**, even though it is tolerated for a single uniform grid
today. *Presents as:* the master fails to parse or renders nothing under
`Xdmf3ReaderT` while the individual per-frame sidecars still open fine.
*Distinguishing from R2:* R2 shows one timestep and renders; R3 errors or
renders nothing. *Do:* do **not** pre-emptively rewrite the spellings — V1 is
what decides. If R3 fires, the minimal fix is the child-grid element names in
`detail::writeXdmfGrid()`, which is one function precisely because T1 factored
it.

**R4 — Rewrite cost on very long runs.** The master is rewritten every frame, so
total text written is O(frames²). At 10 000 frames that is ~10⁸ characters
cumulative on rank 0 — still small against 10 000 collective HDF5 writes, but
not free. *Presents as:* rank-0-only wall-clock creep late in a long run, with
`TIMER_WRITE_MESH` flat (the rewrite is outside it). *Do:* if measured as
material, add an explicit `flush()`-style cadence control as a new task; do not
silently switch to finalize-only writing, which reintroduces the
crashed-run-has-no-master failure this design rejected.

**R5 — A frame stem in a different directory from the master.** `.xmf` files
reference `.h5` by basename only, so a master and its frames in different
directories yield a master full of unresolvable references — which Paraview
reports as a missing-array or empty-dataset error, not as a path error.
*Presents as:* the master opens with the right timestep count but nothing
renders. *Do:* T2 step 4 makes this throw at `write()` time with both paths
named. If this is ever seen at runtime, that check is missing or was bypassed.
