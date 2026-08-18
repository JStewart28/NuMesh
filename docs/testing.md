# Testing and the ship gate

Read this before running tests or changing what the gate covers.

---

## Minimum test set (ship gate)

The gate is defined by **label + backends + ranks**, not by enumerating test names.

**Gate definition (single source of truth):**
- Label: `regression`
- Backends: `SERIAL`, `HIP`
- Ranks: 1, 2, 3, 4, 5

**Run on Tuolumne (submit as batch job):**
```bash
flux batch scripts/tuolumne/run_regression_minset.flux
```

**Run interactively (from build dir, after env activation):**
```bash
ctest -L regression -R "SERIAL|HIP" --output-on-failure
```

**CI** runs `regression`/`SERIAL` at ranks 1–2 only (no GPU in hosted runners).
See [.github/workflows/ci.yml](../.github/workflows/ci.yml). This subset is stated
explicitly in CI — the HIP gate is Tuolumne-only.

**Promoting a test into the gate** requires explicit confirmation: confirm which
backends and ranks the test should run at before setting `TIER regression`.

---

## When to run the full gate

The full gate takes ~30 minutes. It blocks the session for that whole time, so it
is not the default way to check work.

**Default: run only the tests that cover the code you touched.** Select them with
`ctest -R <pattern>` at the ranks and backends the change can plausibly affect —
a serial-only change does not need the HIP tests, a single-rank change does not
need ranks 2–5. State in your report which subset you ran and why it is
sufficient.

**Run the full gate only when the change is broad enough to require it**, i.e. any
of:
- shared headers, core data structures, or anything included by most translation
  units,
- build/CMake, run plumbing, or the resolver,
- the gate definition itself,
- a change whose blast radius you cannot bound by reading the code,
- the final commit of a multi-commit feature, before declaring it done.

## Handoff protocol when the full gate is required

Do not sit idle waiting on a 30-minute job. When the full gate must run:

1. **Submit it from the repository root** so its output file lands in the repo:
   ```bash
   flux batch scripts/tuolumne/run_regression_minset.flux
   ```
   `#FLUX: --output=` in the gate script is a repo-root-relative path, so the CWD
   at submit time decides where the log goes. Never submit from `/tmp` or any
   scratch directory — those are reaped and the results are lost. The resulting
   `tessera-regression-gate.<jobid>.out` is covered by `*.out` in `.gitignore`, so
   it persists locally without being committed.
2. **Record progress** — update the task document and progress log under `tasks/`
   with what is done, what the gate is verifying, and the job id and output file
   name.
3. **Commit and push** the work as it stands.
4. **End the session with a handoff prompt** the user can paste into a fresh
   session. It must name the output file, the task document, and the remaining
   work — enough that the new session can read the gate results and finish
   without rediscovering context. Sketch:

   > Read `tessera-regression-gate.<jobid>.out` in the repo root — the full
   > regression gate for `<task id>` in `tasks/<doc>.md`. If it passed, <remaining
   > work>. If anything failed, diagnose and fix it, then re-run only the failing
   > tests.

---

## Backend suffixes

Test names carry a backend suffix so `-R <BACKEND>` selects one:

```
mesh_partition_SERIAL_np2   mesh_partition_HIP_np4
```

Backend availability per system: see [compile-and-run.md](compile-and-run.md#compute-backends).
