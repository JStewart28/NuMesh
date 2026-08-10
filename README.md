# Tessera

A distributed, MPI- and GPU-aware **unstructured triangle-mesh library** built on
[Cabana](https://github.com/ECP-copa/Cabana) and
[Kokkos](https://github.com/kokkos/kokkos).

Tessera provides the local, halo-able mesh machinery for evolving closed surfaces:
entity storage, full connectivity, a hand-built MPI halo of configurable depth,
split-based adaptive refinement, optional migration/load-balancing, and parallel
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
halo (including halo depth), adaptive refinement, load balancing, and parallel I/O
— live in **[docs/design.md](docs/design.md)**. This README covers the public API, build
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
//   ..., ExecSpace, RefinementMode::Conforming>        // default: no T-junctions
//   ..., ExecSpace, RefinementMode::HangingNode2to1>   // opt-in: 2:1 hanging nodes
// See docs/design.md → Adaptive refinement → Refinement modes.

MeshT mesh( MPI_COMM_WORLD );
buildIcosphere( mesh, /*subdivisions=*/3 );        // initial coarse closed surface,
                                                     // replicated on every rank

auto faceOwner = facePartitionByAxis( mesh, /*axis=*/2 );  // deterministic geometric
                                                             // partition of the faces
MeshHalo<MemSpace> halo;
distribute( mesh, halo, faceOwner, /*depth=*/1 );   // cut to owned + a `depth`-deep ghost
                                                     // layer, build the halo exchange
                                                     // plans. Pass depth=k once, at setup,
                                                     // for a k-ring operator; refine() and
                                                     // migrate() PRESERVE it thereafter
haloExchange( mesh, halo );                         // fill the ghost layer

auto vort = mesh.vertexSlice<Tessera::VertexField::Vorticity>();   // typed Cabana slice
// ... fill/evolve owned vertices ...
haloExchange( mesh, halo );                         // refresh ghosts (whole field pack)

refine( mesh, halo, face_refine_mask );             // 2:1-balanced red split, plus the
                                                     // conforming closure in the default
                                                     // mode; REBUILDS `halo` on the way
                                                     // out, at its recorded depth, so it
                                                     // may be called again immediately and
                                                     // haloExchange() is meaningful
                                                     // straight afterwards

// mesh.haloDepth() reports the depth in force. rebuildHalo( mesh, halo, depth ) rebuilds
// the ghost layer in place from the owned entities -- refine()/migrate() call it
// themselves, so it is only needed when something else changed the owned set.

splitEdges( mesh, halo, edge_split_mask );          // bisect EXACTLY the marked edges (an
                                                     // EDGE mask, sized numOwnedEdges());
                                                     // every incident face becomes 2, 3 or
                                                     // 4 children, conforming on exit with
                                                     // no closure and no 2:1 pass. Also
                                                     // rebuilds `halo`. REMESH family --
                                                     // see "Editing families" below

loadBalance( mesh, halo );           // internal Zoltan2 rebalance + halo rebuild (optional)
// or, external (e.g. Canopy-driven):
//   auto c = ownedFaceCentroids( mesh );
//   std::vector<Rank> dest = external_partition( c );
//   migrate( mesh, halo, dest );   // also rebuilds the halo
haloExchange( mesh, halo );                         // re-sync the field pack (refine() and
                                                     // migrate() leave ghost values already
                                                     // equal to the owners')

writeMesh( mesh, "bubble_0000" );    // bubble_0000.h5 + bubble_0000.xmf
```

### Mesh generators

Tessera ships two closed-surface generators. Both produce a **triangle soup** —
a flat `positions` array plus a flat per-face `triangles` index array — which
`buildFromTriangleSoup()` turns into a fully-connected replicated mesh. The
`build*` wrappers do both steps. A caller is free to supply its own soup
instead; the soup interface exists precisely for that.

```cpp
TriangleSoup<double> s1 = generateIcosphere<double>( /*subdivisions=*/3 );
TriangleSoup<double> s2 = generateLatLonSphere<double>( /*nLat=*/33, /*nLon=*/64 );
buildFromTriangleSoup( mesh, s2 );          // or, in one step:
buildLatLonSphere( mesh, /*nLat=*/33, /*nLon=*/64 );
buildIcosphere( mesh, /*subdivisions=*/3 );
```

| Generator | Counts | Character |
|---|---|---|
| `generateIcosphere(n)` / `buildIcosphere` | `V=12, E=30, F=20` at `n=0`; each level `V'=V+E, E'=2E+3F, F'=4F` | Nearly isotropic, nearly equilateral, almost every vertex valence 6 |
| `generateLatLonSphere(nLat,nLon)` / `buildLatLonSphere` | `V = 2 + (nLat−2)·nLon`, `F = 2·nLon·(nLat−2)`, `E = 3·nLon·(nLat−2)` | **Anisotropic** by construction; the two poles have valence `nLon` |

**Why both.** An icosphere is *too good* a test surface. A lat/lon sphere
stretches its triangles toward the poles and puts two valence-`nLon` outliers on
the surface, so it exercises what an icosphere never reaches: quality-based
marking, the cotangent weight at a high-valence vertex, stencil rows of very
different lengths, and the poles as valence outliers. At `(33,64)` the measured
max/min triangle-area ratio is **10.21** and the max/min edge-length ratio
**14.41** (against ~1 for an icosphere).

`generateLatLonSphere` parameters and guarantees:

- **`nLat`** is the number of latitude rings **including both poles**, `nLat >= 3`;
  **`nLon`** is the number of meridians, `nLon >= 3`. Anything smaller throws
  `std::invalid_argument`. `nLat == 3` is the degenerate-but-legal bipyramid:
  two pole fans, no interior quads, `2·nLon` faces.
- **Vertex ordering** is deterministic and documented, so a consumer can address
  a vertex arithmetically: index `0` is the north pole, then ring
  `j = 1 .. nLat−2` each contributing `nLon` vertices in ascending `i` — so
  `ring(j,i) == 1 + (j−1)·nLon + i` — then the south pole last, at `V−1`.
  Positions are
  `(sin θ·cos φ, sin θ·sin φ, cos θ)` with `θ_j = πj/(nLat−1)` and
  `φ_i = 2πi/nLon` for `i = 0 .. nLon−1` — **no seam duplicate**; `i` wraps.
- **The poles are exact**, written literally as `(0,0,+1)` and `(0,0,−1)`.
  `sin(π)` is not zero in floating point, so a south pole evaluated from the
  formula is off the unit sphere in its last bits *and different for different
  `φ`* — which is how a duplicate-pole bug hides.
- **Winding** is CCW seen from outside, matching `generateIcosphere()`.
- **The quad diagonal is fixed:** each interior quad
  `(i,j)-(i+1,j)-(i+1,j+1)-(i,j+1)` is split along the `(i,j+1)-(i+1,j)`
  diagonal, the same way around the whole sphere. Not adaptive — a caller
  wanting a different triangulation flips edges. This convention is what fixes
  the valence histogram: `2` poles at `nLon`, `2·nLon` vertices at valence 5
  (rings `1` and `nLat−2`), `(nLat−4)·nLon` at valence 6, and for `nLat == 3` a
  single ring of `nLon` valence-4 vertices.
- **Reproducibility caveat.** Unlike the icosphere, whose positions come from a
  rational base table plus `sqrt`, these come from `sin`/`cos` at computed
  angles, which may differ in the last bit across libm implementations and
  platforms. Positions are **not** guaranteed bit-reproducible across machines,
  so compare against a gold file generated elsewhere with a tolerance. Within
  one run they are identical on every rank, which is what the replicated coarse
  build needs.

A **distributed** lat/lon generator is a natural follow-on once
`buildFromTriangleSoupDistributed` exists (the canonical key is just
`{j·nLon + i, invalid_gid}`); it is deliberately not built here. Other
generators (torus, plane, cylinder, cube-sphere) on demand.

### Editing families

Tessera has **two disjoint families of topological edit, and a mesh belongs to
exactly one of them**:

| Family | Operations | Invariant maintained | `Level` semantics |
|---|---|---|---|
| **Hierarchical** | `refine()`, `refineLocal()` | 2:1 level balance; conforming closure | `Level` is **authoritative** |
| **Remesh** | `splitEdges()` *(`collapseEdges()`, `flipEdges()`, `compact()` to follow)* | conformity and manifoldness only | `Level` is **advisory** |

`refine()`'s whole design rests on the level model, and that model is coherent
only because `refine()` performs the uniform 1→4 red split. Bisecting *one* edge
of a triangle produces two children whose edges have mixed levels: no single
integer describes them, and a 2:1 level-difference invariant is not the right
statement about the result. So a `splitEdges()` child **inherits its parent's
level**, and the mesh is thereafter not 2:1-level-meaningful.

Interleaving the two on one mesh is therefore **unsupported, and enforced rather
than documented**. The mesh carries an `EditFamily` tag (`mesh.editFamily()`),
`None` until its first topological edit and then fixed; each entry point throws a
`std::runtime_error` naming **both** families when the tag disagrees:

> `Tessera::refine: this mesh belongs to the Remesh
> (splitEdges/collapseEdges/flipEdges/compact) editing family, and refine belongs
> to the Hierarchical (refine/refineLocal) family. The two are DISJOINT and must
> not be interleaved on one mesh: the hierarchical family maintains the 2:1 level
> balance and the conforming closure with Level authoritative, while the remesh
> family maintains conformity and manifoldness only and leaves Level advisory (a
> child inherits its parent's level). Build a fresh mesh for the other family.`

A one-line check that turns a subtle wrong answer into an immediate abort.
Extending the level model to anisotropic bisection (per-edge levels with a
compatible balance rule) is a much larger design that no known consumer needs; it
is recorded under *Future Optimizations* below, not attempted. See
`docs/design.md` → *Edge-addressed splitting* and
[tasks/edge-split.md](tasks/edge-split.md).

### Example programs

| Example | Directory | Arguments | Description |
|---|---|---|---|
| `hello_tessera` | `examples/01_hello_tessera/` | *(none)* | Prints the active Kokkos backend and MPI rank count. Build verification only. |
| `mesh_pipeline` | `examples/02_mesh_pipeline/` | `--subdiv N` `--axis N` `--balance` `--iters N` `--frac F` `--seed N` `--refine-mode {hanging,conforming}` `--out STEM` | End-to-end demo: build an icosphere, partition + distribute it, then run `N` iterations of random-percent refinement, writing a new `.h5`/`.xmf` frame after each iteration so the mesh evolution can be stepped through in Paraview. Runs the pipeline once per available Kokkos execution space (Serial, plus the platform default and OpenMP where distinct). `--refine-mode` selects the `RefinementMode` the mesh type is instantiated with (default `conforming`); the mode tag is part of the frame stem, so a `hanging` and a `conforming` run of the same `--out` can be compared frame by frame in Paraview. Also demonstrates the profiling API (see below): each iteration is one reporting window. |

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
| 2 | Major sub-phases | `refine_2to1_balance`, `refine_local_rebuild`, `migrate_round_*`, `halo_round_*`, `distribute_csr_rebuild`, `write_datasets`, `lb_zoltan2_solve` |
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
`k=2` both supported), built by edge BFS over the existing connectivity. Rows are
complete for every owned vertex provided the mesh's **halo depth is ≥ k**, and
that is checked: `k > mesh.haloDepth()` throws `std::invalid_argument` naming both
numbers, rather than returning silently short rows on a partition boundary.

```cpp
distribute( mesh, halo, faceOwner, /*depth=*/2 );     // once, at setup
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

**Global scalar reductions** — single-sourced `MPI_Allreduce` wrappers over
`mesh.comm()`. Each takes a per-rank scalar and returns the global result on
every rank: **Tessera owns the collective, the caller owns the local value.** All
are collective — every rank must call them.

```cpp
Scalar dt     = globalMin( mesh, local_dt_estimate );   // adaptive timestep
Scalar cfl    = globalMax( mesh, local_cfl_estimate );  // CFL-style bound
Scalar volume = globalSum( mesh, local_volume );        // enclosed volume
bool   ok     = globalAllFinite( mesh, local_verdict ); // NaN/Inf tripwire
```

`globalAllFinite` is an `MPI_Allreduce(MPI_LAND)` returning true iff *every* rank
passed true, so every rank aborts on the step that produced the NaN rather than
one rank diverging silently. It takes a **verdict, not data** — the local
"everything I hold is finite" sweep is the caller's Kokkos reduction over
whichever fields it cares about, which is knowledge Tessera does not have. There
is deliberately no device-side all-finite helper.

> **`globalSum` floating-point reproducibility — read this before writing a
> cross-rank test.** `MPI_SUM` is not associative in floating point, so a
> `double` result is **not** bitwise reproducible across rank counts, nor across
> runs on a GPU partial-sum path. A quantity carried for the whole run (an
> enclosed volume, a reduced minimum edge length) will differ in its low bits
> between a 4-rank and a 5-rank run, and anything scaling off it — an adaptive
> timestep — diverges from there. **Integer sums are exact and reproducible.**
> Do not write a cross-rank bitwise comparison over a floating-point `globalSum`.
> Fixed-order/compensated summation is out of scope.

The scalar type is mapped to its `MPI_Datatype` by a specialization set; an
arithmetic type with no mapping (e.g. `long double`) is a **compile** error, not
a runtime MPI error.

**Global entity counts** — owned entities partition the global mesh, so a SUM of
the owned counts is the global count. Reduced as `long long`, exact since
integer:

```cpp
long long V = globalOwnedVertices( mesh );
long long E = globalOwnedEdges( mesh );
long long F = globalOwnedFaces( mesh );
long long X = globalOwnedEuler( mesh );   // V - E + F; 2 for a closed conforming surface
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
- Compress the conforming-mode closure bookkeeping. A closure child currently
  stores its red parent outright — `ClosureParent` (one `GlobalId`) plus
  `ClosureParentVerts` (three more), so 4 × 8 B on a ~78 B face core, in memory
  and on disk. A (parent gid, pattern + child-index byte) pair would halve that
  and reconstruct the parent's corners from the sibling group instead; the cost
  is that un-close stops being local per child, which is what makes the current
  encoding cheap in `migrate()`. Worth doing only if face memory becomes the
  binding constraint.
- Unify the two editing families by extending the level model to **anisotropic
  bisection**. Today `refine()` and `splitEdges()` are disjoint families (see
  *Editing families*) because a single integer `Level` cannot describe a face
  whose edges were bisected unevenly: `refine()`'s 2:1 invariant is stated in
  level differences and is only coherent because it performs the uniform 1→4 red
  split. Making them interleavable means **per-edge levels plus a compatible
  balance rule**, which then has to be maintained by the mark-propagation
  fixpoint, the closure patterns, and the HDF5 format alike — a much larger design
  than the guard it would replace, and one no known consumer needs (the driving
  consumer, Beatnik's z-model remesher, is entirely edge-addressed and never calls
  `refine()`). Until then the guard turns the mistake into an immediate abort,
  which is the cheap 95% of the value. Design notes:
  [tasks/edge-split.md](tasks/edge-split.md) → Decision 1's alternative.

---

## Known Issues

- **`markByQuality` on an anisotropic surface marks the *equator* of a lat/lon
  sphere, not the poles.** *(Not a defect — recorded here because the naive
  expectation is inverted, and the inversion is easy to mistake for a bug.)*
  Intuition says the pole region of a UV sphere is the "bad" one, because that is
  where the triangles degenerate as `nLon` grows. It depends entirely on which
  way the mesh is anisotropic. At `(nLat=33, nLon=8)` the meridional step is
  `dθ = π/32` (chord `0.0981`, latitude-independent) while the in-ring step is
  `dφ = 45°`, giving a ring-edge chord of `0.7654·sin θ`. So the ring edges — and
  with them the triangle areas — are **longest at the equator** and shrink to
  nothing at the poles: the polar triangles are the small, nearly *isotropic*
  ones and the equatorial ones carry the 7.8:1 stretch. `EdgeLengthCriterion`
  marks long edges, so it marks the equator, exactly per its documented contract.
  Measured at `(33,8)` with `maxLen = 0.4`, identical at ranks 1–5 on both
  backends: 352 of 496 faces marked; of the 128 faces entirely inside the polar
  caps (`|z| ≥ cos 28.125°`) **none** are marked, and of the 192 entirely inside
  the equatorial band (`|z| ≤ cos 56.25°`) **all** are.
  `CurvatureCriterion` behaves the same way round: at `maxAngle = 40°` it marks
  160 faces, all in the equatorial band and none in the polar caps; at `20°` it
  marks 384, of which only 16 (the two pole fans themselves, 8 faces each) are
  polar. The lesson for a consumer driving AMR from a lat/lon mesh is that
  "polar" and "badly shaped" are not the same set, and which one a criterion
  selects is decided by the `dθ`/`dφ` ratio. Pinned by
  `tests/test_latlon_sphere.cpp` against closed-form latitude cut points rather
  than measured output.
- **Conforming refinement is the `Mesh` default.** *(Not a defect — recorded here
  because it changes what a default-spelled `Mesh` does.)* The whole
  `RefinementMode::Conforming` path — closure kernel, distributed `refine()`,
  `migrate()`/`loadBalance()`, HDF5 round-trip, `markByQuality` — is implemented,
  registered across the suite, and **verified**: the ship gate is 190/190 (180/180
  when this was written; the ten `latlon_sphere` entries came later) and the
  diagnostic tier 62/62 on SERIAL and HIP at **ranks 1–5**, over multiple successive
  adaptive rounds, and the shape-quality bounds are measured rather than assumed (the
  worst radius ratio saturates by round 11 and is flat through round 16 while the mesh
  grows 5.7×). Because it is the default, a `Mesh<...>` spelled with seven template
  arguments gets conforming refinement; a consumer that wants the previous behaviour
  must spell `RefinementMode::HangingNode2to1` explicitly, which every pre-existing
  test in the gate now does. Under `HangingNode2to1` a partial refine mask leaves
  T-junctions bounded to a 2:1 level jump, so the owned-only Euler number equals 2
  only for a uniform refine; that is the *contract* of the mode, not a defect. The
  visible mesh is now rank-count reproducible as well: the blue closure diagonal is
  chosen **geometrically** (shorter diagonal, i.e. the midpoint of the longer split
  edge joins its opposite corner), so nothing about the closure reads an
  `MPI_Exscan`-derived gid. This closed the last recorded limit of conforming mode.
  Design: [tasks/conforming-refinement.md](tasks/conforming-refinement.md) and
  `docs/design.md` → *Adaptive refinement*; verification evidence:
  [tasks/conforming-refinement-debug.md](tasks/conforming-refinement-debug.md).
- **`RefinementMode::HangingNode2to1` does not track hanging nodes across refine
  rounds.** Two consequences, both long-standing and neither visible to that mode's
  own tests (which assert non-conformity anyway), found while fixing the conforming
  closure. First, the 2:1 mark propagation cannot see across a hanging node: its
  coordinator rule needs an edge to have two incident faces, and a hanging node
  leaves the coarse side holding `(a,b)` while the fine side holds `(a,m)`/`(m,b)`
  — different keys, one incidence each — so a sequence of *adaptive* rounds can
  drive a level jump larger than 2:1. Second, when the coarse face itself refines,
  `refine()` mints a fresh midpoint for `(a,b)` rather than reusing the `m` that is
  already there, leaving two coincident vertices. A uniform mask hits neither.
  `RefinementMode::Conforming` fixes both, because it can recover the persistent
  split-edge map locally from the closure bookkeeping; `HangingNode2to1` keeps no
  such record and would need a new face field or an extra message round.
- **Edge user fields are reset by `refine()`/`refineLocal()`/`splitEdges()`.** Edges
  are re-derived from the new face connectivity, so any per-edge user data is
  re-initialized (M1 carries no edge user state through AMR or remeshing). Vertex and
  face user fields are preserved (interpolated / inherited).
