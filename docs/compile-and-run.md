# Building and running Tessera

Read this before building, running an example, or submitting a job. System
detection (hostname → system token) is in [CLAUDE.md](../CLAUDE.md); per-system
machine facts are in `systems/<system>/claude.md`.

---

## New-checkout quickstart

```bash
hostname                        # e.g. tuolumne1004 → use systems/tuolumne/claude.md

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

Which backends build and run is declared per-system in `systems/<system>/claude.md`.

---

## Formatting

The repo has clang-format targets — `cmake --build build-<system> --target format`
and `--target format-check` (style in `.clang-format`). **Do not run them as part
of normal work**: the user formats by hand or asks for the target explicitly.
