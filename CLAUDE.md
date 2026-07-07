# Tessera — Project Control Document

**This is the framework control document.** Read it first in every session; follow
its rules to keep the framework self-consistent across sessions.

---

## New-checkout quickstart

```bash
hostname                        # e.g. tuolumne1004 → use docs/tuolumne/claude.md

# Activate environment (Tuolumne):
spack env activate ~/spack_envs/tuolumne_trilinos/

# Build (manual mode, default):
mkdir build-tuolumne && cd build-tuolumne
bash ../run_cmake_toulumne.sh
make -j $(nproc)

# Run the regression gate:
flux batch ../scripts/tuolumne/run_regression_minset.flux
```

Override build settings without editing committed files: create
`scripts/<system>/profile.local.sh` (gitignored) before sourcing the resolver.

---

## Background / task logs

Multi-phase or ongoing problems live in `tasks/`, one file per topic. Use the
template at [tasks/TEMPLATE.md](tasks/TEMPLATE.md). **At the start of a session
touching an active task, read the task file first and append progress as work
lands.**

Current task logs:

| Task | File | Status |
|------|------|--------|
| Milestone 1 — distributed triangle mesh | [tasks/milestone1_mesh.md](tasks/milestone1_mesh.md) | In progress (Steps 0–8, 10a done; regression gate 60/60, unit 28/28 — Step 10b next; Step 9 still deferred) |

---

## System detection

Before building or running, run `hostname` and match the prefix:

| Hostname prefix | System token | Per-system instructions |
|---|---|---|
| `tuolumne*` | `tuolumne` | [docs/tuolumne/claude.md](docs/tuolumne/claude.md) |
| *(any other)* | `local` | [docs/local/claude.md](docs/local/claude.md) |

The resolver (`scripts/lib/tessera_env.sh`) performs this match automatically.
Override with `export TESSERA_SYSTEM=<token>` before sourcing.

**Fallback rule:** unmatched hostname or a `docs/<system>/claude.md` missing a
required section → stop and ask the user before proceeding.

---

## Build & run profile

**Build mode** is orthogonal to which system — it governs how the env is activated,
where binaries live, and which gate runner applies.

| Mode | When | Binaries land in | Gate runner |
|---|---|---|---|
| `manual` | Default — hand-compiled out-of-tree | `build-<system>/` | `ctest -L regression` in build dir |

*(No spack package for Tessera yet. Only `manual` mode is active.)*

### Session-start flow

1. Check for `scripts/<system>/profile.local.sh`; if present, note it and do not
   re-ask for build mode or dirs this session.
2. If absent, **AskUserQuestion** for build mode + build directory, then write
   `profile.local.sh` with those choices.
3. Source the resolver: `. scripts/lib/tessera_env.sh`.

### Profile mechanism

| File | Committed? | Purpose |
|---|---|---|
| `scripts/<system>/profile.defaults.sh` | Yes | Per-system zero-config defaults |
| `scripts/<system>/profile.local.sh` | **No** (gitignored) | Per-checkout override |

The resolver sources `profile.defaults.sh` first, then `profile.local.sh`
(if present). `${VAR:=default}` ordering means env > local > defaults.

### Resolver knobs

| Variable | Default | Purpose |
|---|---|---|
| `TESSERA_SYSTEM` | auto from `hostname` | Override system token |
| `TESSERA_BUILD_MODE` | `manual` | Build mode |
| `TESSERA_BIN_MODE` | same as `BUILD_MODE` | Binary resolution mode |
| `TESSERA_BUILD_DIR` | `$TESSERA_REPO/build-<system>` | Out-of-tree build directory |
| `TESSERA_SPACK_ENV` | per-system default | Spack env path (future spack mode) |
| `TESSERA_NO_SPACK_ACTIVATE` | `0` | Set to `1` to skip spack activation |

### `tessera_exe` helper

```bash
source scripts/lib/tessera_env.sh
exe=$(tessera_exe examples/01_hello_tessera/hello_tessera)
flux run --ntasks 2 --nodes=1 --exclusive --cores-per-task=1 "$exe"
```

In `manual` mode, resolves `$TESSERA_BUILD_DIR/<relpath>`.

---

## Per-system runtime environment

Launch-time env vars that must reach Flux-submitted tasks live in **one** file per
system: `scripts/<system>/runtime_env.sh`. The resolver sources this automatically.

**Batch scripts must not re-export these vars inline** — sourcing the resolver is
sufficient and avoids drift.

Skipped when `TESSERA_NO_SPACK_ACTIVATE=1`. Omit the file entirely for systems
that need no launch-time exports.

---

## Compute backends

Tessera targets multiple Kokkos execution spaces:

| Backend | Kokkos space | Systems | In gate? |
|---|---|---|---|
| `SERIAL` | `Kokkos::Serial` | All | **Yes** |
| `OPENMP` | `Kokkos::OpenMP` | Tuolumne | No (diagnostic) |
| `HIP` | `Kokkos::HIP` | Tuolumne (MI300A APU) | **Yes** |

Which backends build and run is declared per-system in `docs/<system>/claude.md`.

Test names carry a backend suffix so `-R <BACKEND>` selects one:
```
mesh_partition_SERIAL_np2   mesh_partition_HIP_np4
```

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
See [.github/workflows/ci.yml](.github/workflows/ci.yml). This subset is stated
explicitly in CI — the HIP gate is Tuolumne-only.

**Promoting a test into the gate** requires explicit confirmation: confirm which
backends and ranks the test should run at before setting `TIER regression`.

---

## Plans

Plan-mode plan files are saved to `./plans/` in this repo.

---

## General guidelines

- **Checkpoint commits in plans.** Break large changes into stable, reviewable
  commits at intermediate milestones.
- **Formatter:** `cmake --build build-<system> --target format` (clang-format,
  style defined in `.clang-format`). Format-only check: `--target format-check`.
- **License/header convention:** every new source file (`.cpp`, `.hpp`, `.h`,
  `.cu`, `.sh`, `CMakeLists.txt`, `.cmake`) carries the project's BSD 3-Clause
  header. For cmake/shell files, use the `#`-box style in
  [cmake/FindCLANG_FORMAT.cmake](cmake/FindCLANG_FORMAT.cmake). For C++ files,
  use an equivalent `/* */` block with `SPDX-License-Identifier: BSD-3-Clause`.
- **README.md in sync:** update when a public API or an example's accepted
  arguments change.
- **Future Optimizations:** track in README; ask before adding an entry.
- **Known Issues:** record in README what fails, how it reproduces, and whether
  it predates current work.

---

## Maintaining this framework

### Adding a system

1. Create `docs/<system>/claude.md` with **all seven required sections**
   (Environment, Build-config args, Build command, Run command, Batch template,
   Non-test binaries, Backends).
2. Add a row to the hostname table above.
3. Add a `case` branch in `scripts/lib/tessera_env.sh`.
4. Add `scripts/<system>/profile.defaults.sh` (committed).
5. Add `scripts/<system>/runtime_env.sh` if launch-time env vars are needed.
6. Add `scripts/<system>/run_regression_minset.<scheduler>`.
7. Declare the system's backends in both this file and `docs/<system>/claude.md`.
8. Commit an env snapshot under `docs/<system>/` if one is used (e.g. spack.yaml).

### Invariants — never violate

- **No inline runtime-env in batch scripts** — source the resolver instead.
- **Gate definition is single-sourced:** label + backends + ranks must match
  identically across this file, `tests/CMakeLists.txt`, every
  `scripts/<system>/run_regression_minset.*`, and `.github/workflows/ci.yml`.
  Changing the gate is deliberate and requires updating all four locations.
- **`profile.local.sh` is never committed** (enforced by `.gitignore`).
- **`profile.defaults.sh` is always committed.**
- **Env snapshots stay in sync** with the live environment in the same change.
- **Example argument changes mirror into README.**
- **New files carry the license/SPDX header.**
- **A failing test is never silently excluded from the gate** — label it `unit`,
  record it in README Known Issues, and report it.
