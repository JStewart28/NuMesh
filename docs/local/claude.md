# Local — Tessera System Guide

Local workstation or laptop build. No HPC scheduler; Serial and OpenMP backends
only. Matched when `hostname` does not match `tuolumne*`.

## 1. Environment

No spack or module loads are needed. Ensure Kokkos, Cabana, and MPI are
installed and on `CMAKE_PREFIX_PATH`.

A typical install path if built from source:
```bash
export CMAKE_PREFIX_PATH=$HOME/local/tessera-deps:$CMAKE_PREFIX_PATH
```

## 2. Build-config args

See [run_cmake.sh](../../run_cmake.sh). Key differences from Tuolumne:
- No Cray wrappers — use the system/default C++ compiler.
- No HIP flags.
- No `MPIEXEC_*` overrides — let CMake's FindMPI auto-detect `mpirun`.

## 3. Build command

```bash
# First-time configure + build:
mkdir build-local && cd build-local
bash ../run_cmake.sh [-DCMAKE_PREFIX_PATH=<deps-install>]
make -j $(nproc)

# Incremental rebuild:
cd build-local && make -j $(nproc)
```

## 4. Run command for binaries

```bash
source scripts/lib/tessera_env.sh
mpirun -n <N> $(tessera_exe <relpath-to-binary>) [args...]
```

## 5. Job-scheduler batch template

No scheduler. Run interactively via `mpirun`:

```bash
source scripts/lib/tessera_env.sh
mpirun -n <N> $(tessera_exe <relpath>) [args...]
```

## 6. Running non-test binaries (examples)

When asked to run an `examples/` program, ask for the example name and any
arguments, then plug into section 4.

Example — run `01_hello_tessera` on 2 ranks:

```bash
source scripts/lib/tessera_env.sh
mpirun -n 2 $(tessera_exe examples/01_hello_tessera/hello_tessera)
```

## 7. Backends

| Backend | Kokkos space | Enabled | In gate |
|---|---|---|---|
| `SERIAL` | `Kokkos::Serial` | Yes | **Yes** |
| `OPENMP` | `Kokkos::OpenMP` | If Kokkos built with OpenMP | No |
| `HIP` | `Kokkos::HIP` | No | No |

The local gate covers `SERIAL` only. HIP is Tuolumne-only.
Run: `ctest -L regression -R SERIAL --output-on-failure`
