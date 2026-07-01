# Tuolumne — Tessera System Guide

Tuolumne is an AMD MI300A (APU) HPC cluster. Hostname pattern: `tuolumne*`.

## 1. Environment

Activate the spack environment before building or running anything:

```bash
spack env activate ~/spack_envs/tuolumne_trilinos/
```

No module loads are required. The runtime env vars (MPICH GPU-aware communication,
HIP/HMM settings, OpenMP placement) live in
[scripts/tuolumne/runtime_env.sh](../../scripts/tuolumne/runtime_env.sh) and are
sourced automatically by the resolver — do not re-export them in batch scripts.

## 2. Build-config args

See [run_cmake_toulumne.sh](../../run_cmake_toulumne.sh) for the full CMake
invocation. Key flags:

| Flag | Value | Why |
|---|---|---|
| `CMAKE_CXX_COMPILER` | `CC` | Cray C++ wrapper (MPI + HIP-enabled) |
| `CMAKE_C_COMPILER` | `cc` | Cray C wrapper |
| `CMAKE_HIP_COMPILER` | `amdclang++` | AMD HIP compiler for MI300A |
| `CMAKE_CXX_COMPILER_LAUNCHER` | `ccache` | Incremental build caching |
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` | Optimized with debug symbols |
| `MPIEXEC_EXECUTABLE` | `$(which flux)` | ctest launches tests via flux run |
| `MPIEXEC_NUMPROC_FLAG` | `run;--ntasks` | Flux run flag for rank count |
| `MPIEXEC_PREFLAGS` | `--nodes=1;--exclusive;--cores-per-task=1;--env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608` | Flux resource binding per test + per-task static-TLS surplus |

The `MPIEXEC_*` overrides make `ctest` the single entry point: each test is
launched as `flux run --ntasks N --nodes=1 --exclusive --cores-per-task=1
--env=GLIBC_TUNABLES=... <exe>`. Without the resource binding, CMake's FindMPI
auto-detects `srun`, which deadlocks at ≥3 ranks because unbound ranks contend on
the MI300A APU during Kokkos::initialize.

The `--env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608` enlarges glibc's
surplus static-TLS block **for the launched task only**. The Cray `CC` wrapper
links libsci and the ROCm/HIP runtime dlopens large-TLS libraries, which otherwise
abort every binary at load with *"libsci_cray_mp.so.6: cannot allocate memory in
static TLS block"*. It is injected here rather than in `runtime_env.sh` because the
same variable **segfaults the Cray linker** if present in the build environment —
so it must reach the run task without polluting the compiler/linker env.

## 3. Build command

```bash
# First-time configure + build:
spack env activate ~/spack_envs/tuolumne_trilinos/
mkdir build-tuolumne && cd build-tuolumne
bash ../run_cmake_toulumne.sh
make -j $(nproc)

# Incremental rebuild:
cd build-tuolumne && make -j $(nproc)
```

## 4. Run command for binaries

Using the resolver's `tessera_exe` helper:

```bash
source scripts/lib/tessera_env.sh
flux run --ntasks <N> --nodes=1 --exclusive --cores-per-task=1 \
    $(tessera_exe <relpath-to-binary>) [args...]
```

## 5. Job-scheduler batch template

Submit batch jobs via `flux batch`. Batch scripts must source the resolver first
so runtime_env.sh is applied:

```bash
#!/usr/bin/env bash
#FLUX: --job-name=tessera-<job>
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --output=tessera-<job>.{{id}}.out

set -euo pipefail

source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

# Branch on TESSERA_BIN_MODE if needed, then launch:
flux run --ntasks <N> --nodes=1 --exclusive --cores-per-task=1 \
    $(tessera_exe <relpath>) [args...]
```

See [scripts/tuolumne/run_regression_minset.flux](../../scripts/tuolumne/run_regression_minset.flux)
for the gate wrapper.

## 6. Running non-test binaries (examples)

When asked to run an `examples/` program, ask for the example name and any
arguments, then plug into section 4 or 5.

Example — run `01_hello_tessera` interactively on 2 ranks:

```bash
source scripts/lib/tessera_env.sh
flux run --ntasks 2 --nodes=1 --exclusive --cores-per-task=1 \
    $(tessera_exe examples/01_hello_tessera/hello_tessera)
```

## 7. Backends

| Backend | Kokkos space | Enabled | In gate |
|---|---|---|---|
| `SERIAL` | `Kokkos::Serial` | Yes | **Yes** |
| `OPENMP` | `Kokkos::OpenMP` | Yes | No (diagnostic) |
| `HIP` | `Kokkos::HIP` (MI300A APU) | Yes | **Yes** |

The Kokkos build in the spack env provides Serial, OpenMP, and HIP. Tessera
inherits all three via `find_package(Kokkos)`. The MI300A architecture is
`Kokkos_ARCH_AMD_GFX942`.

Test names carry a backend suffix: select one with `ctest -R SERIAL` or
`ctest -R HIP`. The gate runs both: `ctest -L regression -R "SERIAL|HIP"`.
