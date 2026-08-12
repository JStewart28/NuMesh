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

## Backend suffixes

Test names carry a backend suffix so `-R <BACKEND>` selects one:

```
mesh_partition_SERIAL_np2   mesh_partition_HIP_np4
```

Backend availability per system: see [build-and-run.md](build-and-run.md#compute-backends).
