# Tessera

A distributed non-uniform mesh library built on [Cabana](https://github.com/ECP-copa/Cabana)
and [Kokkos](https://github.com/kokkos/kokkos).

---

## Usage / API

> **Note:** Tessera is in early development. The API is not yet stable.

### Entry points

*(To be documented as the API is implemented.)*

### Minimal end-to-end example

See [examples/01_hello_tessera/hello_tessera.cpp](examples/01_hello_tessera/hello_tessera.cpp)
for a build-verification example that initializes Kokkos and MPI.

### Example programs

| Example | Directory | Arguments | Description |
|---|---|---|---|
| `hello_tessera` | `examples/01_hello_tessera/` | *(none)* | Prints the active Kokkos backend and MPI rank count. Build verification only. |

---

## Dependencies and Build Notes

### Dependencies

| Dependency | Notes |
|---|---|
| [Kokkos](https://github.com/kokkos/kokkos) | ≥ 4.0; Serial, OpenMP, and HIP execution spaces |
| [Cabana](https://github.com/ECP-copa/Cabana) | MPI-enabled build required |
| MPI | Cray-MPICH on Tuolumne; OpenMPI on local workstations |
| CMake | ≥ 3.21 |
| clang-format | For `make format` / `make format-check` |

### Build (Tuolumne — AMD MI300A)

```bash
spack env activate ~/spack_envs/tuolumne_trilinos/
mkdir build-tuolumne && cd build-tuolumne
bash ../run_cmake_toulumne.sh
make -j $(nproc)
```

### Build (local workstation)

```bash
mkdir build-local && cd build-local
bash ../run_cmake.sh -DCMAKE_PREFIX_PATH=<deps-install>
make -j $(nproc)
```

### Design limitations

*(None documented yet.)*

---

## Future Optimizations

*(None documented yet.)*

---

## Known Issues

*(None documented yet.)*
