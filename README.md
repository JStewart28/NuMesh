# Tessera

A distributed, MPI- and GPU-aware **unstructured triangle-mesh library** built on
[Cabana](https://github.com/ECP-copa/Cabana) and
[Kokkos](https://github.com/kokkos/kokkos).

Tessera provides the local, halo-able mesh machinery for evolving closed surfaces:
entity storage, full connectivity, a hand-built 1-deep MPI halo, split-based
adaptive refinement, optional migration/load-balancing, and parallel
Paraview-readable I/O. It is a dependency of the [Canopy](https://github.com/) FMM
solver and the Beatnik/rocketrig Rayleigh–Taylor problem; the global
Birkhoff–Rott/FMM velocity solve lives downstream in Canopy and is **not** part of
Tessera.

> **Status: early development — design reference.** This README and
> [docs/design.md](docs/design.md) document the *design* of milestone 1 (a single
> rising bubble). The API below is the target the
> implementation is being built against (see `plans/compiled-hopping-torvalds.md`
> and the `tasks/milestone1_mesh.md` handoff contract). Sections marked *(planned)*
> are not yet implemented.

---

## Documentation

Detailed descriptions of the algorithms and design decisions used in this
repository — entity storage and AoSoA layout, global ID scheme, ownership and the
1-deep halo, adaptive refinement, load balancing, and parallel I/O — live in
**[docs/design.md](docs/design.md)**. This README covers the public API, build
instructions, and per-example arguments; the design rationale is not duplicated
here.

Per-system build/run instructions (one file per machine) live under
[systems/](systems/).

---

## Usage / API

The full pipeline below is implemented and gate-tested end to end (see
`examples/02_mesh_pipeline/` for a complete, runnable version with CLI args).

```cpp
#include <Tessera.hpp>

using namespace Tessera;
using MeshT = Mesh<double, /*Dim=*/3, VertexFields<>, EdgeFields<>, FaceFields<>,
                   MemSpace, ExecSpace>;
// An optional 8th parameter selects the refinement conformity contract:
//   ..., ExecSpace, RefinementMode::HangingNode2to1>   // default: 2:1 hanging nodes
//   ..., ExecSpace, RefinementMode::Conforming>        // no T-junctions (in progress)
// See docs/design.md → Adaptive refinement → Refinement modes.

MeshT mesh( MPI_COMM_WORLD );
buildIcosphere( mesh, /*subdivisions=*/3 );        // initial coarse closed surface,
                                                     // replicated on every rank

auto faceOwner = facePartitionByAxis( mesh, /*axis=*/2 );  // deterministic geometric
                                                             // partition of the faces
MeshHalo<MemSpace> halo;
distribute( mesh, halo, faceOwner );                // cut to owned + 1-deep ghost layer,
                                                     // build the halo exchange plans
haloExchange( mesh, halo );                         // fill the ghost layer

auto vort = mesh.vertexSlice<Tessera::VertexField::Vorticity>();   // typed Cabana slice
// ... fill/evolve owned vertices ...
haloExchange( mesh, halo );                         // refresh ghosts (whole field pack)

refine( mesh, halo, face_refine_mask );             // conforming 2:1 split refinement;
                                                     // clears `halo` as a side effect

loadBalance( mesh, halo );           // internal Zoltan2 rebalance + halo rebuild (optional)
// or, external (e.g. Canopy-driven):
//   auto c = ownedFaceCentroids( mesh );
//   std::vector<Rank> dest = external_partition( c );
//   migrate( mesh, halo, dest );   // also rebuilds the halo
haloExchange( mesh, halo );                         // required again after refine/migrate

writeMesh( mesh, "bubble_0000" );    // bubble_0000.h5 + bubble_0000.xmf
```

### Example programs

| Example | Directory | Arguments | Description |
|---|---|---|---|
| `hello_tessera` | `examples/01_hello_tessera/` | *(none)* | Prints the active Kokkos backend and MPI rank count. Build verification only. |
| `mesh_pipeline` | `examples/02_mesh_pipeline/` | `--subdiv N` `--axis N` `--balance` `--iters N` `--frac F` `--seed N` `--out STEM` | End-to-end demo: build an icosphere, partition + distribute it, then run `N` iterations of random-percent refinement, writing a new `.h5`/`.xmf` frame after each iteration so the mesh evolution can be stepped through in Paraview. Runs the pipeline once per available Kokkos execution space (Serial, plus the platform default and OpenMP where distinct). Also demonstrates the profiling API (see below): each iteration is one reporting window. |

### Profiling

Tessera carries an optional, level-gated wall-time profiler
(`Tessera_Profiling.hpp`, header-only). Every instrumented region is an RAII scoped
timer that (1) pushes a `Kokkos::Profiling` region — so external Kokkos-aware tools
(Kokkos Tools, `rocprof`, Nsight) see the named region for free — and (2)
accumulates `MPI_Wtime` wall time into a process-local registry. It is **compiled
out entirely** at level 0; the default build enables nothing.

**Enable it (CMake):**

| Option | Default | Meaning |
|---|---|---|
| `Tessera_ENABLE_PROFILING` | `OFF` | Master switch. `OFF` is an authoritative kill switch — it forces the effective level to 0 regardless of `Tessera_PROFILING_LEVEL`. |
| `Tessera_PROFILING_LEVEL` | *(empty)* | `0`/`1`/`2`/`3`. Empty resolves to `1` when profiling is enabled, else `0`. |

When the effective level is `> 0`, `TESSERA_ENABLE_PROFILING` and
`TESSERA_PROFILING_LEVEL=<n>` are added as compile definitions on the `Tessera`
interface target (inherited by every consumer). The `run_cmake*.sh` scripts default
to level 0; override on the command line, e.g.
`bash ../run_cmake_toulumne.sh -DTessera_PROFILING_LEVEL=2`.

**Verbosity levels** — higher levels are strictly additive (a level-*n* build emits
every region at levels ≤ *n*):

| Level | Adds | Example regions |
|---|---|---|
| 1 | Top-level phases | `build_icosphere`, `partition`, `distribute`, `halo_exchange`, `refine`, `migrate`, `load_balance`, `write_mesh`, `read_mesh`, `mark_*` |
| 2 | Major sub-phases | `refine_2to1_balance`, `refine_local_rebuild`, `migrate_round_*`, `distribute_csr_rebuild`, `write_datasets`, `lb_zoltan2_solve` |
| 3 | Comm rounds / device kernels | `refine_advertise_alltoallv`, `mark_edge_length_kernel`, `write_hyperslabs`, `read_hyperslabs` |

**Reporting is caller-driven — Tessera has no timestep loop.** The library owns the
mechanism; the downstream application decides the cadence. Two registries back every
region: a resettable **window** and a monotonic **lifetime**. The caller-facing
macros (all no-ops at level 0):

```cpp
TESSERA_RESET_TIMERS();               // clear the window registry
TESSERA_PRINT_TIMERS( comm );         // collective: rank 0 prints the window
                                      //   min/max/mean/imbalance per region
TESSERA_PRINT_TIMERS_TOTAL( comm );   // collective: rank 0 prints the lifetime total
```

A simulation prints and resets every `ts` steps for a per-window breakdown, and
prints the lifetime total once at shutdown:

```cpp
for ( int step = 0; step < nsteps; ++step ) {
    // ... Tessera refine / migrate / haloExchange / writeMesh ...
    if ( step % ts == 0 ) {
        TESSERA_PRINT_TIMERS( comm );   // aggregated over this window
        TESSERA_RESET_TIMERS();         // start the next window
    }
}
TESSERA_PRINT_TIMERS_TOTAL( comm );     // whole-run aggregate
```

`examples/02_mesh_pipeline/` wires exactly this pattern, treating each adaptive
refinement iteration as one window. All three print/reset macros are collective on
the passed communicator, so every rank must call them.

### Geometry & stencil operators

A narrow, physics-free surface for building local surface operators (surface
gradient, Laplace–Beltrami, curvature, vertex normals/areas) on top of the mesh.
**Tessera owns the traversal, the assembly, and the raw geometry; the caller owns
the weights and every convention** — the weight scheme (cotangent vs RBF-FD vs
uniform), the normal orientation, the area definition (barycentric/Voronoi/⅓),
the curvature sign. Tessera never sees a physics parameter. This split is what
lets one interface hold both operator families; when the discretization is
settled it is one weight-builder in the caller, and *zero* changes in Tessera.

Headers: `Tessera_Geometry.hpp`, `Tessera_Stencil.hpp`, `Tessera_FieldReduce.hpp`,
`Tessera_Reduction.hpp` (all folded into `<Tessera.hpp>`).

**Raw geometric primitives** — pure geometry, no convention. Because a `Mesh` is
not capturable into a device kernel and its connectivity stores vertex *global
ids*, the primitives read a lightweight, device-capturable accessor built once
from the mesh:

```cpp
auto geom = buildMeshGeometry( mesh );   // captures the position slice + per-face /
                                         // per-edge LOCAL vertex indices
// inside a KOKKOS_LAMBDA, per face f / edge e / corner c (0,1,2):
Scalar A   = faceArea( geom, f );              // ½‖(p1−p0)×(p2−p0)‖
Scalar n[3]; faceNormalRaw( geom, f, n );      // unnormalized (p1−p0)×(p2−p0); the
                                               //   outward sign is the caller's choice
Scalar v[3]; edgeVector( geom, e, v );         // p[v1]−p[v0]
Scalar cot = cotangentAtCorner( geom, f, c );  // (u·v)/‖u×v‖ (triangle geometry only)
```

**k-ring stencil topology** — a CSR of the k-ring vertex neighbours (`k=1` and
`k=2` both supported), built by edge BFS over the existing connectivity:

```cpp
auto stencil = buildVertexStencil( mesh, /*k=*/2 );   // VertexStencil<MemSpace>
```

**Weighted-stencil apply** — halo-correct, GPU-resident. Applies a caller-built
weight View aligned to the stencil CSR: `out(i) = Σ_j w(i,j)·in(j)` over owned
vertices. **The caller builds `w`** (that is the convention) and must
`haloExchange` `in` first so ghost neighbour values are current:

```cpp
Kokkos::View<Scalar*, MemSpace> w( "w", stencil.csr.get().numEntries() );
buildMyWeights( mesh, geom, stencil, w );   // caller's cotangent / RBF-FD / … fill
haloExchange( mesh, halo );                  // refresh ghost `in` before apply
applyStencil( mesh, stencil, w, in_slice, out_slice );   // owned vertices only
```

**Face→vertex reduce** — Tessera iterates a vertex's incident faces; the caller's
device functor `op(v, f, geom, faceSlice, vertSlice)` defines the accumulation
(no atomics — one thread per owned vertex). This is the primitive behind vertex
normals and vertex areas, whose conventions stay in the caller:

```cpp
reduceVertexFromFaces( mesh, geom, faceSlice, vertSlice, MyAreaOrNormalOp{} );
```

**Global scalar reduction** — a single-sourced `MPI_Allreduce(MPI_MIN)` over
`mesh.comm()`, e.g. for an adaptive-timestep min-reduce:

```cpp
Scalar dt = globalMin( mesh, local_dt_estimate );
```

**Validity.** `MeshGeometry` and `VertexStencil` are generation-guarded like mesh
slices (see *Slice/handle validity*): both are stamped with the mesh generation at
build time and abort on use after a topology op. **Rebuild them with
`buildMeshGeometry`/`buildVertexStencil` after any `distribute`/`migrate`/`refine`/
`loadBalance`**; they survive `haloExchange` (topology-preserving). Note also that a
solver's *operator consistency* at irregular (non-valence-6) vertices is a property
of the weights, not the apply — document and convergence-test that on the caller's
weight-builder; `applyStencil` is exact arithmetic and is correctness-tested against
an analytic field.

---

## Dependencies and Build Notes

### Dependencies

| Dependency | Notes |
|---|---|
| [Kokkos](https://github.com/kokkos/kokkos) | ≥ 4.0; Serial, OpenMP, and HIP execution spaces |
| [Cabana](https://github.com/ECP-copa/Cabana) | MPI-enabled build required |
| [Trilinos](https://github.com/trilinos/Trilinos) | Zoltan2 (geometric MultiJagged) for internal load balancing |
| [HDF5](https://www.hdfgroup.org/) | Parallel (`+mpi`) build, for mesh I/O *(added to the spack env in Step 0)* |
| MPI | Cray-MPICH on Tuolumne (GPU-aware); OpenMPI on local workstations |
| CMake | ≥ 3.21 |
| clang-format | For `make format` / `make format-check` |

### Build (Tuolumne — AMD MI300A)

```bash
spack env activate ~/spack_envs/tuolumne_trilinos/
mkdir build-tuolumne && cd build-tuolumne
bash ../run_cmake_toulumne.sh
make -j $(nproc)
```

`run_cmake_toulumne.sh` resolves and passes `-DHDF5_ROOT=<prefix>` automatically
(via `spack location -i hdf5`, falling back to the known Cray path). This is
required on Tuolumne because the Cray parallel HDF5 is a spack **external** and
is not view-linked onto `CMAKE_PREFIX_PATH`; a bare `find_package(HDF5)` would
otherwise silently resolve the OS serial `/usr/lib64` build. CMake fails loudly
(`HDF5_IS_PARALLEL` guard) rather than configuring against a non-parallel HDF5.

### Build (local workstation)

```bash
mkdir build-local && cd build-local
bash ../run_cmake.sh -DCMAKE_PREFIX_PATH=<deps-install>
make -j $(nproc)
```

### Design limitations

- Milestone 1 implements **split-based refinement only** (no edge collapse/flip).
- Topology surgery (pinch-off / merger of surfaces) is not implemented; the data
  model is component-agnostic to leave room for it.

---

## Future Optimizations

- Incrementally patch the `HaloExchangePlan` index maps after refinement instead of
  a full `rebuild()` each 2:1-balance iteration.

---

## Known Issues

- **A distributed mesh must be re-haloed after `refine()` before `haloExchange()`.**
  Distributed `refine()` (Step 6b) leaves each rank holding only its refined *owned*
  entities and clears the halo. The general (non-replicated) halo rebuild now exists
  (Step 7, inside `migrate()`), but it is currently coupled to `migrate()` and is not
  yet invoked automatically at the end of `refine()`; calling `haloExchange()` on a
  freshly-refined-but-not-migrated mesh is a no-op on an empty plan, not a synced
  ghost layer. Factoring the rebuild into a standalone `rebuildHalo()` that `refine()`
  also calls is a tracked follow-up.
- **Adaptive `refine()` is non-conforming (bounded hanging nodes).** A partial
  refine mask leaves T-junctions bounded to a 2:1 level jump; the owned-only Euler
  number equals 2 only for a uniform (conforming) refine. A fully conforming
  triangulation is **in progress**: `Mesh`'s optional 8th template parameter
  `RefinementMode` now exists, and `RefinementMode::Conforming` selects a face
  layout carrying the closure bookkeeping members — but the closure itself is not
  implemented, so `refine()`/`refineLocal()` on a `Conforming`-typed mesh abort
  with a "not implemented" diagnostic. `RefinementMode::HangingNode2to1` (the
  current default, and the only mode any test or example uses) is unaffected. See
  [tasks/conforming-refinement.md](tasks/conforming-refinement.md) for the design
  and remaining tasks.
- **`buildVertexStencil(mesh, 2)` (k=2) is incomplete within one hop of a partition
  boundary.** Tessera's halo is **1-deep**, which fully covers a k=1 stencil but not a
  k=2 one: for an owned vertex whose 2-ring reaches beyond the ghost layer, the missing
  outer-ring neighbours are silently absent from its CSR row rather than reported as an
  error. The marked set is therefore only correct for vertices whose entire 2-ring is
  held locally (all interior vertices on a single rank; interior-of-partition vertices
  in a distributed run). Workaround: either widen the halo to depth ≥ k before building
  the stencil, or restrict k=2 stencils to single-rank runs. k=1 stencils are unaffected.
- **Edge user fields are reset by `refine()`/`refineLocal()`.** Edges are re-derived
  from the new face connectivity, so any per-edge user data is re-initialized (M1
  carries no edge user state through AMR). Vertex and face user fields are preserved
  (interpolated / inherited).
