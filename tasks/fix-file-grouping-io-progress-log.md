# XDMF time-series grouping — progress log

Session record for fix-file-grouping-io. Companion to `fix-file-grouping-io.md`,
which holds the design, the task sequence and the risks; this file holds what
actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `fix-file-grouping-io.md` can
cite it by ID. No dates: the order of the sections is the chronology. If a
session covers more than one task, name them all; if it belongs to no task, name
the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that invalidates
a later task is worthless if the session starting that task has to read the whole
log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could not
stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation departed
from the task's stated **Do** steps, and why — a task marked `**DONE**` that was
done differently than it was written is the quietest way for a design to stop
describing the code.

V1 in particular exists to be recorded here: the Paraview version, every reader
tried, and the timestep count each one reported.

## T1

**Decisions taken as given** (handed down with the task, recorded so a later
session does not relitigate them):

- **The byte-comparison vehicle is `examples/02_mesh_pipeline`, not `test_io`.**
  `test_io.cpp` deletes its sidecar
  ([test_io.cpp:294-295](../tests/test_io.cpp#L294-L295),
  [:523-524](../tests/test_io.cpp#L523-L524)), so it leaves nothing to `cmp`;
  the example leaves one `.xmf` per frame per exec space on disk.
- **Determinism was established before anything was attributed to the change.**
- **The failure-direction check was temporary and is reverted.** No timed call
  site is left in the example — T3 owns that.
- **Scope of test running:** all targets built; only `io_SERIAL` np1-5 run.
  `io_HIP` was not run — the sidecar is rank-0 text with no device involvement.

**Final shape.** `XdmfField` moved from `Tessera::detail` to `Tessera`
unchanged. New in `Tessera`:

```cpp
struct XdmfFrame {
    std::string h5name;   // BASENAME of the .h5, never a path
    int dim; int scalarBytes;
    unsigned long long Nv, Nf;
    std::vector<XdmfField> vFields, fFields;
};
```

Emitters, exactly as the design stated them:

```cpp
detail::writeXdmfGrid( std::FILE*, const XdmfFrame&, int indent );
detail::writeXdmfGrid( std::FILE*, const XdmfFrame&, int indent, double time );
Tessera::writeXdmf( const std::string& stem, const XdmfFrame& );
Tessera::writeXdmf( const std::string& stem, const XdmfFrame&, double time );
detail::writeMeshH5( const MeshT&, const std::string& stem ) -> XdmfFrame;
Tessera::writeMesh( const MeshT&, const std::string& stem ) -> XdmfFrame;
Tessera::writeMesh( const MeshT&, const std::string& stem, double ) -> XdmfFrame;
```

`indent` is the column the `<Grid>` line starts at in spaces — 2 standalone, 4
nested in a collection. Every line inside is written as `pad + relative indent`,
so the format strings differ from the originals only by a leading `%s`.

**Departures from the stated Do steps**, both internal and neither changing a
signature the design names:

1. Each overload pair is one emitter plus a thin pair of wrappers. The two
   timed/untimed spellings funnel into `detail::writeXdmfGridImpl( fp, frame,
   indent, const double* time )` and `detail::writeXdmfFile( stem, frame, const
   double* time )`. The *API* is an overload pair as the design requires; the
   pointer is private, and it is what makes byte-identity structural rather than
   a thing to keep re-checking. T2 must call the public
   `detail::writeXdmfGrid()` overloads, not `...Impl`.
2. `TESSERA_SCOPED_TIMER( TIMER_WRITE_MESH )` stayed in the two public
   `writeMesh()` overloads rather than moving into `writeMeshH5()`, so the timer
   still spans the HDF5 write *and* the sidecar exactly as before. A caller that
   calls `writeMeshH5()` directly is therefore untimed — T2's `MeshSeries`
   should go through `writeMesh()`.
3. `const int R = mesh.rank()` was deleted from `writeMeshH5()`: its only use
   was the rank-0 sidecar guard, which now lives in `writeMesh()` and reads
   `mesh.rank()` there. `comm` and `size` are still used inside.

**Verification, and one gap the prescribed check does not cover.** The example
run (4 ranks, `flux run --ntasks 4 --nodes=1 --exclusive --cores-per-task=16`,
the committed binding) produces 6 sidecars per run — Serial/Default/OpenMP ×
frame0/frame1, conforming — and all 6 are byte-identical to their pre-change
copies. But `mesh_pipeline` instantiates its mesh with `VertexFields<>`,
`EdgeFields<>`, `FaceFields<>` and `Dim=3`, so that comparison exercises
**neither** the user-field attribute loop **nor** the `dim != 2 && dim != 3`
attribute-only branch — the two places the reindentation could have gone wrong
and nothing in the repo would have noticed. No existing test leaves a
user-field `.xmf` on disk to compare (`test_io` is the only mesh with user
fields and it deletes its sidecar). Closed with a throwaway header-only probe
(`/usr/WS2/stewartj/t1_runs/probe/`): two 1-task programs including
`git show HEAD:src/Tessera_Xdmf.hpp` and the post-change header, emitting the
same four cases — dim 3, dim 2, dim 4, and empty field lists, each with a scalar
and a multi-extent user field on both centerings. All four `.xmf` byte-identical
old vs new. The same probe printed `detail::writeXdmfGrid( fp, frame, 4 )` and
its timed overload, confirming the nesting T2 needs.

**No bug was revealed by running.** Two things running *did* reveal, both
about the harness rather than the code:

- `/tmp` is node-local on Tuolumne. A job submitted from a `/tmp` directory
  fails to `chdir`, silently lands in the compute node's own `/tmp`, and still
  exits 0 — its output is simply unreachable afterwards. Job working
  directories must be on a shared filesystem; these runs used
  `/usr/WS2/stewartj/t1_runs`.
- `flux job wait` returns `Request requires owner credentials` for these jobs.
  `flux job attach <id>` blocks until exit and `flux job status <id>` then
  yields the exit code.
- Two `--out` stems differing in their **last** component cannot be compared
  byte-for-byte at all: the `.xmf` embeds the `.h5` basename, so the text
  differs for that reason alone. The determinism probe used `--out runA/mesh`
  and `--out runB/mesh` — distinct stems, shared basename.

**Affects:** T2 — `XdmfFrame` and both `detail::writeXdmfGrid()` overloads
landed exactly as the design stated them, so T2 needs no adjustment on that
front; note only that `writeXdmfGridImpl`/`writeXdmfFile` are private plumbing
to call *through*, not directly, and that `MeshSeries` should call `writeMesh()`
(timed) rather than `writeMeshH5()` so `TIMER_WRITE_MESH` still covers the
frame. T2's `tests/test_xdmf_series.cpp` is the first place a user-field `.xmf`
gets asserted-on in-repo — worth asserting the attribute block there, since the
throwaway probe that covered it in T1 is not committed. V1 — R3's fix stays
one-function-wide: every child grid's element spellings now come from
`detail::writeXdmfGridImpl()` alone.
