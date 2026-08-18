# Rising-Bubble Architecture: Tessera / Canopy / solver split

**Status:** design document + implementation record. Deliverable for the Milestone‑1
closed‑surface Z‑model rising bubble.
**Date:** 2026‑07‑13 (design); **2026‑07‑14 (Tessera geometry/assembly APIs implemented — see
§"Implementation status").**
**Authoritative sources:** `background.md` (physics + milestone scope, collaborator‑owned),
plus reconnaissance of four codebases (citations throughout).

> **Update 2026‑07‑14 — the five new Tessera entry points are now implemented, tested, and on
> `main`.** The "narrowest interface" (§Q1) shipped: `buildMeshGeometry`/`MeshGeometry` +
> raw geometric primitives (`Tessera_Geometry.hpp`), `buildVertexStencil`/`applyStencil`
> (`Tessera_Stencil.hpp`), `reduceVertexFromFaces` (`Tessera_FieldReduce.hpp`), and `globalMin`
> (`Tessera_Reduction.hpp`), each with a unit test (SERIAL+HIP where applicable). One signature
> difference from the sketch below is load‑bearing and documented in §"Implementation status":
> the primitives take a device‑capturable `MeshGeometry` accessor, **not** the `Mesh` itself.

**Decisions taken as input (from Jason, this session):**
1. **Discretization:** design so *both* the RBF‑FD family (zmodel‑kokkos / zmodel‑unstructured)
   and the forthcoming cotangent family are expressible behind one interface; **Tessera owns
   neither weight scheme.** Isolating this choice is a first‑class evaluation criterion.
2. **Tessera core is settled:** Cabana‑AoSoA storage, the compile‑time field pack, and the
   reallocate‑on‑topology‑change behavior are fixed foundations. I design *on top* and *flag*
   hazards; I do not propose rewriting the core.
3. **`zmodel-unstructured` (Python):** noted, not deeply inventoried.
4. **Beatnik `risingbubble` branch:** treated as an abandoned spike; it does not anchor the
   recommendation.

A convention word means the caller (solver/driver), not Tessera, fixes it. **[inferred]** tags
mark anything reasoned rather than observed.

---

## Provenance of the codebases (who is authoritative on what)

| Code | Path | Language / stack | Role |
|---|---|---|---|
| Tessera | this repo | C++ / Kokkos / Cabana AoSoA, MPI | the mesh substrate under design |
| zmodel‑kokkos | `~/research-bridges/zmodel-kokkos` | C++ / Kokkos, **no Cabana, no MPI**, single‑GPU | reference solver (Ian May) |
| zmodel‑unstructured | `~/research-bridges/zmodel-unstructured` | Python / numpy | earlier reference, same math lineage |
| Canopy | `~/research-bridges/Canopy` | C++ / Kokkos / Cabana, MPI, Zoltan2 | FMM far‑field |
| Beatnik + rocketrig | `~/spack_envs/tuolumne_beatnik/beatnik` | C++ / Kokkos / Cabana::Grid, heFFTe, MPI | existing structured Z‑model solver |

**The central provenance fact that shapes Q1:** the two zmodel reference codes *already disagree
with each other and with `background.md`* on the local‑operator conventions:

- **State variable.** `background.md` §4/§6 says "potential formulation… a single per‑vertex
  scalar." zmodel‑kokkos stores a **3‑vector** `mu` plus position, 6 doubles/vertex
  (`zmodel-kokkos/src/ZModel.h:166`, split at `ZModel.hpp:301-302`). The Python code likewise
  carries a vector `mu` (`zmodel-unstructured/BRIntegral.py:74` `find_mu`, `:94` `mu_to_omega`).
  **Neither reference implements a scalar‑potential state.** This is a direct contradiction with
  `background.md`, not a nuance — see Open Questions.
- **Vertex normals.** zmodel‑kokkos: tangent cross‑product `D1Z × D2Z` from RBF gradients
  (`ZModel.hpp:64-73`). zmodel‑unstructured: **area‑weighted average of incident face normals**
  (`Mesh.py:403-410`). Same author, two codes, two conventions.
- **Local operators.** Both reference codes use **RBF‑FD** surface gradients over a **2‑ring**
  stencil (`zmodel-kokkos/src/SurfRBF.h:54`, `SurfRBF.hpp:15-243`; `zmodel-unstructured/Mesh.py:474`
  `rbf_surf_grad`), assembled into sparse operator matrices. `background.md` §2/§6 describes
  **1‑ring cotangent weights**. These are different operator *families*, not different constants.
- **Vertex areas.** `background.md` implies per‑vertex areas (γ = sheet‑strength × *local area*).
  zmodel‑kokkos stores **no per‑vertex area** at all — `omega` is integrated as a continuous field
  by face quadrature (`BirkhoffRott.hpp:273-337`); only `getFaceArea` exists (`Mesh.h:241`).

This is exactly the "moving spec" the task warns about, and it is *already* moving between two
copies of the reference. The design consequence is unambiguous: **Tessera must not bake in a
weight scheme, a normal convention, an area definition, or the choice of state variable.** Those
belong to the solver and, ultimately, to the collaborator's oracle.

---

## Phase 0 — Reconnaissance findings

### 0.1 zmodel‑kokkos: operation inventory (the ground truth for *what math runs*)

Data layout: **raw `Kokkos::View`s, no Cabana, no MPI, single execution space**
(`Mesh.h:38-110`; exec space `SurfRBF.h:37`). Reference mesh is a unit sphere; the physical
surface `Z` is a separate field, and several operators hard‑assume the unit‑sphere reference
(`SurfRBF.hpp:219`, `Mesh.h:217`).

| Operation | Location | Convention (quoted / paraphrased) | Topology needed | R/W |
|---|---|---|---|---|
| Conormal / tangent frame (ξ,η) | `Mesh.h:207-239` | built from **reference unit‑sphere normal** `(vx,vy,vz)`; "assumes unit sphere as reference mesh" (`:217`) | vertex only | reads ref coords |
| RBF‑FD surface gradient operators (∂x,∂y,∂z) | `SurfRBF.hpp:15-243`, applied `:428-448` | RBF kernel `exp(-s·d²)+w·d³`, `s=1.0`, `w=0.001` (`:231-242,347-348`); `num_polys=6` (`:328`); **2‑ring** stencil (`SurfRBF.h:54`); normal component projected out (`:218-228`) | 2‑ring via edge BFS (`Mesh.hpp:935-968`) | sparse CRS ops |
| Surface gradients of Z (D1Z,D2Z) | `ZModel.hpp:261-263,56-62` | RBF grad of Z, contracted with (ξ,η) | 2‑ring | vertex |
| Vertex unit normal | `ZModel.hpp:64-73` | `normal = D1Z × D2Z`, normalized — **tangent cross‑product, not face average** | 2‑ring (via grads) | vertex |
| First fundamental form / metric | `ZModel.hpp:75-80` | `h11,h22,h12` = dots of D1Z,D2Z; **det has a bug** (see below) | 2‑ring | vertex `[4]` |
| Sheet strength `omega` from `mu` | `ZModel.hpp:82-91` | `ν = (ξ·mu, η·mu)/√det`; `omega = ν2·D1Z − ν1·D2Z` | 2‑ring | reads `mu`, writes `omega` |
| Mean curvature | `ZModel.hpp:95-170`, formula `:165-168` | `½·tr(g⁻¹·II)` using **inverse metric weights**, from gradients of the normal field; **not** cotangent, **not** Laplacian‑of‑position | 2‑ring | vertex scalar |
| Laplace–Beltrami (viscous reg.) | `SurfRBF.hpp:516-534`, used `ZModel.hpp:363` | **RBF div‑grad** (gradient then divergence), not cotangent | 2‑ring | vertex `[3]` |
| Bernoulli forcing scalar ("mu_flux") | `ZModel.hpp:329-338` | `|u_BR|² − ¼|omega|² + 2·g·Z_z` (`:337`) | vertex | vertex scalar |
| Surface gradient of Bernoulli forcing (+surface tension) | `ZModel.hpp:342-360` | RBF gradient, Atwood‑scaled; tension `4σ·∇(curvature)` accumulated | 2‑ring | vertex `[3]` |
| Birkhoff–Rott velocity (global) | `BirkhoffRott.hpp:520-557`, kernel `BirkhoffRott.h:127-152` | `u = (1/4π) Σ [ω×(q−p)] / (\|q−p\|² + ε²)^{3/2}`; **ε added, then ^{3/2}** (`:146-147`); `ε = 0.15·radius` (`ProblemSetup.h:37`) | all‑to‑all over faces | vertex |
| Face area (BR quadrature only) | `Mesh.h:241-264` | `½·\|(Z1−Z0)×(Z2−Z0)\|` | face | — |
| Uniform 1→4 refinement + tree/flatten/remap | `Mesh.hpp:675-767,549-673,769-785` | red split; midpoints normalized to sphere | edges/faces | large |
| Time integration | `TimeIntegrator.hpp:181-227`, step `:269-404` | **embedded adaptive multi‑stage RK ("3*+", 5 stages)** with PI step control + reject/retry — **not a plain RK3** | global DOF vector | — |

**BR is *not* behind a swappable interface:** the two integrators are chosen by
`if constexpr(false)` (`BirkhoffRott.hpp:530`) and recursion depth is `#define NUM_RECURSE 1`
(`:9`), so it is effectively direct O(N_vert × N_face) with a 1‑level tree — editing source is the
only way to change it. There is **no FMM** here.

**Bugs to flag (the reference numerics are currently suspect):** `d_metric` is width‑4
(`ZModel.h:177`) but index 4 is written/read (`ZModel.hpp:79,83,84,168`) — out of bounds; and the
determinant squares the never‑written index 3 instead of `h12` at index 2 (`ZModel.hpp:79-80`).
These matter because the collaborator's oracle (§6) will presumably be *correct*, so the C++
reference cannot be trusted as the numerical target — only as a structural guide. **[inferred]**

**Cross‑check against `background.md` §2 (gaps both ways):**

- In §2 and in code: normals ✅, mean curvature ✅, Laplace–Beltrami ✅ (as RBF div‑grad), surface
  gradient of Bernoulli forcing ✅, BR sum ✅.
- In §2 but **weak/absent** in code: **vertex areas** (none stored); **"1‑ring" locality**
  (operators are 2‑ring); **scalar potential** (state is vector `mu`); **cotangent** Laplace–Beltrami
  (RBF‑FD instead).
- In code but **not in §2** (extra scope to be aware of): the conormal frame, first fundamental
  form, the `mu→omega` mapping, gradients of the *normal* field, an entire uniform‑AMR/tree
  subsystem, a BulkFlow particle path (inactive, `num_particles=0`), and adaptive error‑controlled
  time stepping.

### 0.2 Tessera: what exists, real vs stub

**Everything in the Milestone‑1 surface is real, non‑stubbed code.** No `throw`/TODO/`#if 0`
placeholders in `src/`.

- **Storage:** three Cabana AoSoAs (`Tessera_Mesh.hpp:74-76,188-190`). Field pack is a
  **compile‑time template parameter** (`Tessera_Fields.hpp:41-46,60-82`); **no runtime add‑field**.
  Field access: `mesh.vertexSlice<Tessera::userVertexField<0>()>()` → a `GenerationHandle`-wrapped
  Cabana slice (`Tessera_Mesh.hpp:184-190`; indices `Tessera_Fields.hpp:87-144`).
- **Connectivity (all real):** face→3 verts / 3 edges, edge→2 verts / ≤2 faces, vertex→edges and
  vertex→faces CSR (`Tessera_MeshBuilder.hpp:113-116,158-163,191-229`; `Tessera_CsrAdjacency.hpp:41-79`).
  No vertex→vertex or face→face list (1‑ring reached via vertexEdges/vertexFaces).
- **Differential geometry:** **none exposed at design time.** The only geometric computation was a
  transient face normal used for curvature‑based refinement marking (`Tessera_MarkQuality.hpp:266-275`)
  — produced, consumed, discarded. No area, no vertex normal accessor, no curvature, no
  Laplace–Beltrami, no gradient/divergence operator. **[UPDATE 2026‑07‑14]** The raw‑geometry and
  generic‑assembly layer of the design (§Q1) now exists: `faceArea`/`faceNormalRaw`/`edgeVector`/
  `cotangentAtCorner` (`Tessera_Geometry.hpp`), `buildVertexStencil`/`applyStencil`
  (`Tessera_Stencil.hpp`), `reduceVertexFromFaces` (`Tessera_FieldReduce.hpp`), `globalMin`
  (`Tessera_Reduction.hpp`). These remain **pure geometry + generic assembly** — still no
  convention, no vertex‑normal/area definition, no curvature, no weight scheme in Tessera.
- **Halo exchange:** real, one‑deep, generic over any AoSoA/field pack
  (`Tessera_HaloExchange.hpp:134-136`, whole‑tuple `MPI_Type_contiguous` `:172-175`). Does **not**
  resize. Documented invalidation contract at `:50-52`.
- **Refinement:** split‑only. Single‑rank red 1→4 (`Tessera_Refine.hpp:206-209`) and distributed
  2:1‑balanced (`Tessera_RefineParallel.hpp:150-152`). **No edge collapse, no edge flip anywhere.**
  Distributed refine **clears the halo and does not rebuild it** (`:712-714`) — Known Issue.
- **Migration / partition / LB:** device migrate primitive (`Tessera_Migrate.hpp:76-78`), public
  mesh migrate + ghost rebuild (`Tessera_MeshMigrate.hpp:154-156`), geometric static partition
  (`Tessera_Distribute.hpp:101-147`), **Zoltan2 (Trilinos) genuinely linked**
  (`Tessera_Zoltan2Balancer.hpp:55-57,177-185`, multijagged `:147`). External vs internal LB is
  already the contract: external = call `migrate(dest)` with your own map; internal = `loadBalance()`
  composes Zoltan2 → migrate. `ownedFaceWeights` returns **all 1.0** (`Tessera_MeshMigrate.hpp:818`)
  — LB currently balances face *count*, not work.
- **I/O:** real collective parallel HDF5 + XDMF (`Tessera_HDF5Writer.hpp:50`, `Tessera_Xdmf.hpp:80`),
  full round trip. Paraview sees the triangle surface + vertex/face fields (edges round‑trip in HDF5
  but are not in the XDMF grid).
- **Bubble init:** `buildIcosphere(mesh, subdiv)` (`Tessera_MeshBuilder.hpp:237-244`) →
  golden‑ratio icosahedron subdivided and projected to the unit sphere (`Tessera_Icosphere.hpp:77-160`).

**The ownership hazard is real and pervasive, but now guarded.** Every count‑changing op
**reallocates and reassigns** the AoSoAs, and this still **silently dangles any *bare*
caller‑held slice/tuple/CSR/key‑View**:
`migrate` builds `AoSoAType migrated(...)` then `aosoa = migrated;` (`Tessera_Migrate.hpp:236,275`);
distributed refine `resize`+`deep_copy` all three (`Tessera_RefineParallel.hpp:412-413,600-601,628-629`)
and clears halo plans (`:719-722`); mesh migrate the same (`Tessera_MeshMigrate.hpp:543-575`) and
*replaces* halo plans (`:673-680`); `distribute` the same (`Tessera_Distribute.hpp:291-338`). What
changed: `Mesh` now carries a monotonic **`generation()`** counter
(`Tessera_Mesh.hpp:144-145,329`), bumped at every one of those reallocation sites, and
`vertexSlice()`/`edgeSlice()`/`faceSlice()` (plus the CSR/key‑View `...Handle()` accessors) return a
**`GenerationHandle`** (`Tessera_GenerationGuard.hpp`) stamped with the generation at capture
time. Copying a stale handle — e.g. capturing it into a `KOKKOS_LAMBDA` after one of the ops above
has run — aborts with a diagnostic instead of silently reading dangling storage; see
§"Ownership‑hazard remediation" below for the mechanism. `RegisteredBufferPool`
(`Tessera_RegisteredBufferPool.hpp:50-90`) stabilizes only **MPI staging buffers**, not entity
storage — it is unrelated to (and unaffected by) the new guard. The one topology‑preserving op that
does *not* resize is `haloExchange` (`Tessera_HaloExchange.hpp:133`), so it never bumps
`generation()` and handles survive it by construction.

#### Ownership-hazard remediation

- **Generation token.** `Mesh::_generation` (`Tessera_Mesh.hpp:329`) is a `std::size_t`, exposed
  read‑only via `generation()` (`:144`) and bumped by the public `bumpGeneration()` (`:145`). It is
  bumped by `resizeVertices/Edges/Faces` (`:148-162`), `setOwnedCounts` (`:124-130`), the key‑View
  replacement methods `setEdgeKeys`/`setFaceKeys` (`:298-307`), and the CSR replacement methods
  `rebuildVertexFaces`/`rebuildVertexEdges` (`:257-270`), which are now the sanctioned way to
  reallocate the CSR (they wrap `detail::fillCsr` from `Tessera_CsrAdjacency.hpp`). Every
  reallocation call site in `Tessera_Distribute.hpp`, `Tessera_MeshMigrate.hpp`,
  `Tessera_RefineParallel.hpp`, `Tessera_Refine.hpp`, and `Tessera_MeshBuilder.hpp` was switched
  from directly reassigning `mesh.edgeKeys()`/`mesh.faceKeys()` or calling
  `detail::fillCsr(mesh.vertexFaces(), ...)` to calling these bumping methods.
- **Generation-stamped wrapper.** `GenerationHandle<Underlying>` (new file
  `Tessera_GenerationGuard.hpp`) stores the wrapped handle plus the generation captured at creation
  and a pointer to the mesh's live counter. `operator()` forwards to the underlying handle
  unconditionally — identical device codegen, no per‑element cost. Validation happens in the copy
  constructor/copy‑assignment, guarded to run **host‑side only**
  (`#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)`, the same idiom `Kokkos::View`
  itself uses): copying a handle — which is exactly what happens when one is captured by value into
  a `KOKKOS_LAMBDA` — re‑checks it against the mesh's current generation and calls `std::abort()`
  with a diagnostic on mismatch. The check (and its call sites) compile out entirely when
  `Tessera_ENABLE_DEBUG_CHECKS` is off (new CMake option, default **ON**, mirroring the existing
  `Tessera_ENABLE_PROFILING` pattern in `CMakeLists.txt`) — release builds pay nothing.
- **Uniform invalidation contract.** Every reallocation site now carries an `// INVALIDATION: ...`
  comment reusing `HaloExchangePlan`'s existing wording (`Tessera_HaloExchange.hpp:51-52`), extended
  from "the plan" to "every slice/CSR/key‑View handed out before the change." See
  `Tessera_Distribute.hpp:270-274`, `Tessera_MeshMigrate.hpp:527-532,673-675`,
  `Tessera_RefineParallel.hpp:408-411,716-719`, `Tessera_Refine.hpp:417-420`, and
  `Tessera_Migrate.hpp:265-274` (the last documents that this mesh‑agnostic primitive cannot bump
  `generation()` itself and a direct caller must call `mesh.bumpGeneration()`).
- **`haloExchange` stays outside the invalidating set** — it never calls `bumpGeneration()`, so a
  `GenerationHandle` taken before it is used is still valid after it returns; this is exercised
  directly by `tests/test_staleslice_guard.cpp`.
- **Re‑slice helper.** `vertexSlices<M...>()`/`edgeSlices<M...>()`/`faceSlices<M...>()`
  (`Tessera_Mesh.hpp:211-225`) return a `std::tuple` of `GenerationHandle`s for a whole field set in
  one call, so "re‑slice at the top of every stage" is a single structural call rather than tribal
  knowledge.
- **New test.** `tests/test_staleslice_guard.cpp` (registered `TIER unit`, `BACKEND SERIAL`,
  `tests/CMakeLists.txt`) covers both directions: a slice held across a generation bump aborts when
  copied (checked in a forked child process, since `std::abort()` would otherwise kill the test
  binary), and a slice held across `haloExchange()` does not abort.

### 0.3 Beatnik: swappable BR, everything else grid‑bound

- **BR is a clean runtime‑polymorphic strategy.** One pure virtual method:
  `virtual void computeInterfaceVelocity(node_view zdot, node_view z, node_view o) const = 0`
  (`BRSolverBase.hpp:29`, `node_view = Kokkos::View<double***,MemorySpace>` `:27`). Three backends —
  `ExactBRSolver` (`ExactBRSolver.hpp:54-55`), `CutoffBRSolver` (`CutoffBRSolver.hpp:55`),
  **`FmmBRSolver` wrapping `Canopy::Solver`** (`FmmBRSolver.hpp:69-70,85-86`). Factory dispatch on an
  enum (`CreateBRSolver.hpp:27,33-60`, FMM branch `#ifdef BEATNIK_ENABLE_CANOPY`). **The FMM backend
  was added without touching the solver core** — empirical proof the BR axis is swappable.
- **Constructor contract a backend must match:** `(const pm_type&, const BoundaryCondition&,
  double epsilon, double dx, double dy, Params)` (`ExactBRSolver.hpp:71-73` etc.).
- **But the interface is grid‑flavored:** every backend takes `ProblemManager` and enumerates points
  via `localGrid()->indexSpace(Own(),Node(),Local())` (`FmmBRSolver.hpp:153,335`), and the FMM backend
  **aborts if periodic** (`FmmBRSolver.hpp:110-119`). The `node_view` triple‑array is a structured
  layout; `FmmBRSolver` repacks it into a particle AoSoA internally (`:150-196`).
- **ε threading:** stored per‑backend (`FmmBRSolver.hpp:611`), passed via factory, rescaled by
  `sqrt(dx·dy)` in the solver (`Solver.hpp:191`), added **un‑squared** in the kernel
  (`Operators.hpp:134` "matlab code doesn't square epsilon"); FMM maps `cfg.softening = sqrt(epsilon)`
  (`FmmBRSolver.hpp:588-604`).
- **Deeply grid‑coupled elsewhere:** `Cabana::Grid::UniformMesh<double,2>` (`SurfaceMesh.hpp:37`),
  grid arrays everywhere (`ProblemManager.hpp:84-86`, `ZModel.hpp:79-81`, `TimeIntegrator.hpp:33-35`),
  grid halos (`ProblemManager.hpp:128-129`), **heFFTe periodic spectral solve** (`ZModel.hpp:183,238-305`),
  periodic BCs (`ExactBRSolver.hpp:112-121`, `SurfaceMesh.hpp:158-174`). RK is **TVD RK3**
  (`TimeIntegrator.hpp:80-140`), but its stages are `parallel_for` over grid index spaces (`:88-140`)
  — the *math* is mesh‑agnostic, the *implementation* is not factored out. Silo I/O
  (`SiloWriter.hpp:30-31`). ~5k lines total; 3 BR backends; 3 solver orders.

### 0.4 Canopy: scalar Laplace FMM; ε‑aware vector BR is a rewrite

- **API:** `Canopy::Solver<MemorySpace,ExecutionSpace,Scalar,P_ORDER,NComps>`
  (`Canopy_Solver.hpp:104-106`), `setup`/`solve` over a **single particle AoSoA — sources == targets**
  (`:191-239`). Per‑source data = `NComps` scalar charges (`Canopy_P2P.hpp:889-890`). Outputs are
  **internally‑owned Views** `potential()` (`Canopy_Solver.hpp:474`) and `gradient()` (`:475`), zeroed
  each solve (`:209-218`).
- **Ownership / residency:** caller allocates the AoSoA but **Canopy permutes and migrates it in
  place** (`particles = migrated;` `Canopy_TreePartitioner.hpp:766`). GPU‑resident throughout; MPI
  passes device pointers (`Canopy_P2P.hpp:572-578`); host mirrors only for tree metadata.
- **Kernel:** the pipeline is templated on `KernelType` but **`LaplaceKernel` (1/r) is the only one**
  and `Solver` hard‑codes it (`Canopy_Solver.hpp:112`; `Canopy_LaplaceKernel.hpp:150-151`). The
  far‑field is a **spherical‑harmonic Greengard–Rokhlin expansion mathematically hard‑wired to
  unsoftened 1/r** (`Canopy_LaplaceKernel.hpp:230-881`, `Canopy_SphericalCoefficients.hpp:23-120`).
  **ε (Plummer softening) lives only in near‑field P2P** (`Canopy_P2P.hpp:803,883-885`); the config
  explicitly documents the far field as unsoftened (`Canopy_Solver.hpp:70-77`). Gradient is by
  **finite differences of the potential** (`Canopy_LaplaceKernel.hpp:851-880`, `TODO: analytical`).
- **Vector output:** none. `NComps>1` = N *independent* scalar solves; no cross product
  (`Canopy_LaplaceKernel.hpp:134-135` comment "3 for Biot‑Savart via three parallel Laplace solves"
  is aspirational). No `cross`/`curl`/`biot` code exists.
- **Partitioner:** Zoltan2 RCB (multijagged) on octree‑leaf centroids over a globally‑replicated tree
  (`Canopy_TreePartitioner.hpp:362-400`, `Canopy_TreeBuilder.hpp:1160-1180`). **Canopy owns and
  imposes its own partition and migrates particles to its owners** — there is **no API to supply an
  external ownership map**, and multijagged is non‑deterministic across calls (`Canopy_Solver.hpp:538-543`).
- **Port estimate for the ε‑aware vector BR (§4 of background):** near‑field P2P → moderate (softening
  exists; add the cross product `u += γ×d·(r²+ε²)^{-3/2}` in `Canopy_P2P.hpp:867-1055`, widen charge to
  a 3‑vector). Far‑field → **near‑total rewrite**: the harmonic expansion of 1/r *cannot represent*
  `(r²+ε²)^{-3/2}`; an ε‑aware far field needs a different basis (e.g. Cartesian Taylor of the
  regularized kernel), replacing `Canopy_LaplaceKernel.hpp` + `Canopy_SphericalCoefficients.hpp`.
  **The tree, Zoltan2 partitioner, comm plan, and sweep drivers are kernel‑agnostic and reusable** if
  the new kernel keeps the `KernelType::` static‑method surface (`Canopy_DownwardSweep.hpp:1099,1636,1688,2029`).

---

## Q1 — What belongs in Tessera's API?

**The test, applied strictly:** an op is Tessera's iff it is fixed by *mesh topology + geometry + a
generic field* and would be written identically for different physics on the same mesh. The moment a
*convention* (normal orientation, area definition, weight scheme, curvature sign) or a *parameter*
(ε, Atwood, σ) enters, it leaves Tessera. Given the "isolate the moving spec" mandate, I split each
operator into a **Tessera‑owned traversal/assembly mechanism** and a **caller‑owned kernel/convention**.

### Q1 table

| Operation | zmodel‑kokkos site | generic / convention‑dependent / problem‑specific | Owner | Rationale | C++ signature sketch |
|---|---|---|---|---|---|
| 1‑ring / k‑ring neighbor access | `Mesh.hpp:935-968` (BFS) | generic | **Tessera** | pure topology; needed by every operator and both weight families | `auto vertexEdges()/vertexFaces()` (exist, `Tessera_Mesh.hpp:156-159`); add `ringNeighbors(v,k)` |
| Face area, face normal (raw geometry) | `Mesh.h:241-264`; normal impl varies | generic *magnitude*, convention‑dependent *sign/orientation* | **Tessera** (unsigned), caller (orientation) | `½‖(p1−p0)×(p2−p0)‖` is identical for all physics; the outward‑sign choice is a convention | `KOKKOS_FN Scalar faceArea(f); KOKKOS_FN Vec3 faceNormalRaw(f)` |
| Vertex normal | `ZModel.hpp:64-73` (cross‑prod) **vs** `Mesh.py:403-410` (area‑avg) | **convention‑dependent** | **caller**, via a Tessera reduction primitive | the two references already disagree; Tessera must not pick | caller functor + `reduceVertexFromFaces(field, op)` |
| Vertex area (barycentric / Voronoi / ⅓‑triangle) | absent in kokkos; `Mesh.py:444` | **convention‑dependent** | **caller**, via a Tessera face→vertex reduce | γ = ω·area needs *an* area; which one is the oracle's call | `reduceVertexFromFaces(...)` *(as shipped — the scatter and vertex‑normal reduce are one primitive with a caller op)* |
| Surface gradient of a **generic scalar/vector field** | `SurfRBF.hpp:428-448` | generic *operator application*; convention‑dependent *stencil weights* | **split**: Tessera owns apply + stencil gather; **caller owns weights** | this is the crux — see below | `applyStencilOperator(op, inField, outField)` where `op` is caller‑built |
| Laplace–Beltrami | `SurfRBF.hpp:516-534` (RBF) **vs** cotangent (forthcoming) | **convention‑dependent weights**, generic assembly | **split** (same as above) | RBF‑div‑grad and cotangent are *different weight fills* over the *same* 1/2‑ring gather | as above; `op` carries the weights |
| Mean curvature | `ZModel.hpp:165-168` | **convention‑dependent** (sign, weighting) + problem‑adjacent | **caller** | `½tr(g⁻¹II)` vs cotangent‑Laplacian‑of‑position vs area‑gradient are all "mean curvature"; sign must match oracle | caller functor over Tessera gradients/normals |
| `mu → omega` (sheet strength) | `ZModel.hpp:82-91` | **problem‑specific** (metric contraction in conormal basis) | **solver** | encodes the Z‑model state representation | solver‑lib only |
| Bernoulli forcing + surface tension | `ZModel.hpp:329-360` | **problem‑specific** (ε‑free but Z‑model‑specific algebra) | **driver/solver** | `‖u‖²−¼‖ω‖²+2gZ_z`, `4σ∇κ` are the physics | driver/solver only |
| Birkhoff–Rott velocity | `BirkhoffRott.hpp:520-557` | **problem‑specific kernel**, generic *point set + partition* | **Canopy** (kernel) + Tessera (points) | the N‑body sum is Canopy's; Tessera supplies positions + γ | see Q3 |
| Halo exchange (1‑deep) | n/a (serial) | generic | **Tessera** (exists) | pure topology/comm | `haloExchange(mesh, halo)` (`Tessera_Distribute.hpp:411`) |
| Split refinement + field interpolation | `Mesh.hpp:675-767` | generic *topology*; convention‑dependent *interpolation* | **Tessera** (topology) + caller (policy) | policy hook already exists (`Tessera_RefinePolicy.hpp`) | `refine(mesh, halo, mask, policy)` (exists) |
| Repartition / migrate / LB | n/a | generic | **Tessera** (exists) | topology + comm | `migrate(mesh,halo,dest)` / `loadBalance(...)` |
| Global min‑reduce (adaptive dt) | `TimeIntegrator.hpp` (serial) | generic reduction | **Tessera thin wrapper** or driver | one `MPI_Allreduce`; trivial, but must be single‑sourced | `Scalar globalMin(mesh.comm(), local)` |
| I/O (connectivity + fields) | `VTKWriter.hpp` | generic | **Tessera** (exists) | HDF5/XDMF already generic over field pack | `writeMesh(mesh, stem)` (exists) |

### The crux: the local‑operator seam (`operator` vs `weights`)

The RBF‑FD family and the cotangent family differ in **how the per‑neighbor weights are computed and
over what stencil**, but they are *identical* in shape once assembled: **a sparse linear operator that
maps a per‑vertex field to a per‑vertex field, whose sparsity pattern is a k‑ring gather.** That
observation is what lets one interface hold both:

- **Tessera owns:** (1) the k‑ring **stencil topology** (offsets + neighbor indices — it already has
  the CSR machinery, `Tessera_CsrAdjacency.hpp`); (2) the **apply** of a weighted stencil operator to a
  generic field, halo‑correct and GPU‑resident (`out[i] = Σ_j w[i,j]·in[j]`); (3) the raw geometric
  inputs a weight builder needs (edge vectors, face areas, unsigned face normals, cotangents of
  triangle angles — all pure geometry).
- **Caller owns:** the **weight fill** — a device functor `w[i,j] = f(geometry at i,j)` that *is* the
  convention. Cotangent weights, RBF‑FD weights, uniform weights are three implementations of that one
  functor. Tessera never sees ε, never picks a sign, never chooses barycentric vs Voronoi.

Concretely, the Tessera entry points (the *only* new operator surface Tessera needs). **These
shipped 2026‑07‑14; the block below is the AS‑IMPLEMENTED signature** (the design sketch took
`const MeshT&`; the reality is the `MeshGeometry` accessor — see §"Implementation status" for why):

```cpp
// 1. Stencil topology: k-ring neighbor CSR for vertices (k=1 or 2 covers both families).
//    Returns a generation-guarded CSR wrapper (VertexStencil).                [Tessera_Stencil.hpp]
template <class MeshT> VertexStencil<mem> buildVertexStencil(MeshT& mesh, int k);

// 2. Raw geometric primitives a weight-builder consumes (pure geometry, no convention). They read a
//    lightweight device-capturable accessor built once from the mesh, NOT the mesh itself, because a
//    Mesh is not device-capturable and its connectivity stores vertex gids.    [Tessera_Geometry.hpp]
template <class MeshT> MeshGeometry<MeshT> buildMeshGeometry(MeshT& mesh);
KOKKOS_FN Scalar faceArea(const MeshGeometry<MeshT>&, LocalIndex f);
KOKKOS_FN void   edgeVector(const MeshGeometry<MeshT>&, LocalIndex e, Scalar out[3]);
KOKKOS_FN void   faceNormalRaw(const MeshGeometry<MeshT>&, LocalIndex f, Scalar out[3]); // unnormalized
KOKKOS_FN Scalar cotangentAtCorner(const MeshGeometry<MeshT>&, LocalIndex f, int corner); // triangle only

// 3. Weighted-stencil apply over a generic field (the halo-correct, GPU-resident workhorse).
//    out(i)=Σ_j w(j)*in(nbr(j)) over OWNED vertices; caller haloExchanges `in` first. [Tessera_Stencil.hpp]
template <class MeshT, class WeightsView, class InSlice, class OutSlice>
void applyStencil(MeshT&, const VertexStencil<mem>&, WeightsView w, InSlice in, OutSlice out);

// 4. Vertex<-face reduce, for areas and vertex normals (caller gives the accumulation op).
//    op(v, f, geom, faceSlice, vertSlice) runs per incident face.           [Tessera_FieldReduce.hpp]
template <class MeshT, class FaceSlice, class VertSlice, class ReduceOp>
void reduceVertexFromFaces(MeshT&, const MeshGeometry<MeshT>&, FaceSlice, VertSlice, ReduceOp);

// 5. Global reduction for adaptive dt.                                       [Tessera_Reduction.hpp]
template <class MeshT, class Scalar> Scalar globalMin(const MeshT&, Scalar local);
```

The **weights themselves are built by the solver library**, not Tessera:

```cpp
// Solver-lib code, NOT Tessera. This is where the convention lives, and the single
// place that changes when the collaborator's Python lands:
buildCotangentLaplacianWeights(mesh, stencil, w);   // OR
buildRbfFdWeights(mesh, stencil, w);                // both call Tessera's geometric primitives
```

This satisfies the mandate exactly: **the weight scheme, the normal convention, the area definition,
the curvature sign, and the state variable are all outside Tessera.** When the oracle arrives, the
change is one weight‑builder + one normal functor + one area functor in the solver library — not ten
places, and *zero* places in Tessera.

### The narrowest interface

**The narrowest interface Tessera can expose such that the rising‑bubble solver is expressible without
Tessera knowing anything about bubbles is: (a) the existing mesh substrate — Cabana‑AoSoA storage with
a compile‑time field pack, both‑way connectivity, generic 1‑deep `haloExchange`, split `refine`,
`migrate`/`loadBalance`, and HDF5/XDMF `writeMesh`; plus (b) five new geometry‑and‑assembly entry
points that carry no physics: `buildVertexStencil`, the raw geometric primitives (`faceArea`,
`edgeVector`, `faceNormalRaw`, `cotangentAtCorner`), `applyStencil`, `reduceVertexFromFaces`, and
`globalMin`.** Everything convention‑bearing — cotangent vs RBF weights, normal orientation, area
definition, mean‑curvature sign, `mu→omega`, the Bernoulli forcing, ε — is assembled *above* that line
in the solver library. Tessera sees "a sparse stencil operator applied to a generic field on a
distributed triangle mesh," and nothing about closed surfaces, sheet strength, or bubbles.

**On the irregular‑vertex note (§ Second milestone):** Laplace–Beltrami being first‑order and
pointwise‑inconsistent at non‑valence‑6 vertices is a property of the *weights* and the *mesh*, not of
the apply. It therefore does **not** change where the operator lives — `applyStencil` is exact
arithmetic regardless — but it *does* dictate documentation and testing: the weight‑builder (solver
lib) must document the inconsistency, and Tessera's `applyStencil` test must use an analytic field so a
failure is unambiguously an apply/halo bug, never expected operator inconsistency. Put the convergence
test on the weights (solver lib), the correctness test on the apply (Tessera).

### Implementation status (2026‑07‑14)

All five entry points are implemented on `main`, header‑only, and folded into `<Tessera.hpp>`.

- **Files:** `src/Tessera_Geometry.hpp` (group 1 + the accessor), `src/Tessera_Stencil.hpp`
  (groups 2+3), `src/Tessera_FieldReduce.hpp` (group 4), `src/Tessera_Reduction.hpp` (group 5).
- **The accessor, and why the signature changed.** The design sketched `faceArea(mesh, f)`. That is
  not implementable on device: a `Mesh` holds AoSoAs, an `MPI_Comm`, and a raw generation pointer and
  is **not capturable into a `KOKKOS_LAMBDA`**, and the face/edge connectivity stores vertex **gids**
  (`FaceField::Verts = GlobalId[3]`) with no device‑side gid→local map. So `buildMeshGeometry(mesh)`
  returns a small **device‑capturable `MeshGeometry`** bundling the position slice + per‑face/per‑edge
  *local* vertex indices (built host‑side from a gid→local map, exactly as `Tessera_MarkQuality.hpp`
  already did), and the primitives read that. This is pure mechanism — no convention crossed the line.
- **Generation‑guard integration (per the §0.2 contract).** `MeshGeometry` holds its position slice as
  the same `GenerationHandle` `Mesh::vertexSlice()` hands out, so **copying the accessor into a
  `KOKKOS_LAMBDA` re‑validates it** and aborts if it was built before a topology op. `VertexStencil`
  wraps its CSR in a `GenerationHandle` stamped via a new minimal read‑only `Mesh::generationPtr()`
  accessor (`Tessera_Mesh.hpp`); `applyStencil` validates it host‑side before launch. Both must be
  **rebuilt after any `distribute`/`migrate`/`refine`/`loadBalance`**; both survive `haloExchange`.
  This is the only change to a core header — a one‑line accessor, no storage/field‑pack change.
- **Tests** (`tests/`, all `TIER unit`): `test_geometry` (analytic two‑triangle mesh, exact values),
  `test_stencil_topology` (k=1/k=2 vs an independent edge‑BFS reference), `test_apply_stencil`
  (analytic `f(p)=p_x`, uniform weights, reference from positions independent of the field‑halo path —
  a failure is unambiguously an apply/halo bug), `test_reduce_faces` (⅓‑area identity → total surface
  area, partition‑independent), `test_global_reduce` (min placed at different ranks). SERIAL+HIP where
  device kernels apply; `global_reduce` is SERIAL‑only (host MPI). The distributed tests run ranks 1–5.
- **Batch runner:** `scripts/tuolumne/run_all_tests.flux` (pdebug, 20 min) runs the whole suite —
  SERIAL at ranks 1–5, HIP at ranks 1–4 (4 GPUs/node). Verified green: **117/117 tests passed**.
- **README:** a new "Geometry & stencil operators" section documents the API and the
  Tessera‑owns‑traversal / caller‑owns‑weights‑and‑conventions split.
- **Still outside Tessera, as designed:** weight schemes, vertex‑normal/area/curvature conventions,
  `mu→omega`, the Bernoulli forcing, ε. Those live in Solverlib (Q2 option c).

---

## Q2 — Extend Beatnik, or build a new solver library?

### The options

**(a) Generalize Beatnik over mesh type + add a `rising_bubble` example.**
*Reused:* the RK3 math, the `Order` dispatch idea, the BR interface shape, Silo I/O, the driver
skeleton. *Rewritten:* `SurfaceMesh`, `ProblemManager`, `TimeIntegrator`, `ZModel`, `SiloWriter`, and
the BR backends must all become mesh‑type‑generic — but recon shows these are **not** factored behind
an abstraction today; they consume `Cabana::Grid` arrays and grid index spaces directly
(`ProblemManager.hpp:84-120`, `TimeIntegrator.hpp:33-140`, `ZModel.hpp:79-305`). *Blast radius:* every
core header. *Risk to rocketrig:* high — rocketrig's low/medium orders depend on the heFFTe periodic
spectral path (`ZModel.hpp:238-305`) that has no unstructured analogue, so the abstraction must
preserve two divergent code paths through the same templates. *Paper story:* "we generalized an
existing solver" — muddied by carrying periodic‑spectral machinery the bubble never uses. The
`risingbubble` branch was an attempt at exactly this and was abandoned (per Jason) — a data point
against.

**(b) New solver library depending on Beatnik's mesh‑agnostic components.**
*Reused (in principle):* RK3, `BRSolverBase`, `Operators::{dot,cross,BR}`. *Problem:* recon shows
almost nothing is *actually* mesh‑agnostic in a reusable form — `TimeIntegrator` and `ProblemManager`
are grid‑bound, and the genuinely grid‑free pieces are ~200 lines of small inline functions
(`Operators.hpp:58-167`, of which only `dot`/`cross`/`BR` are grid‑free). To "depend on" them you must
first refactor Beatnik to expose them, which *is* option (a)'s blast radius wearing a different hat.
*Maintenance:* a hard dependency on a fast‑moving structured‑grid code, for 200 lines. *Paper story:*
awkward ("depends on Beatnik but uses ~5% of it").

**(c) New standalone solver library; Beatnik untouched.**
*Reused:* the *designs*, not the code — replicate Beatnik's proven **BR‑behind‑a‑virtual** pattern
(`BRSolverBase.hpp:29`, factory `CreateBRSolver.hpp`), port the ~200 lines of grid‑free math, reuse the
**FmmBRSolver→Canopy wrapping pattern** (`FmmBRSolver.hpp:150-196,588-604`) as a template for the new
Canopy binding, and reuse TVD‑RK3's weights. *Rewritten:* a small state container over Tessera slices,
an RK3 that AXPYs over vertex slices instead of grid index spaces, the Z‑model update over Tessera
operators, and validation hooks. *Blast radius in Beatnik:* zero. *Risk to rocketrig:* zero. *Isolates
the moving spec:* best — the state representation, weight scheme, and ε all live in one new small code
you fully control, with none of Beatnik's periodic/spectral constraints leaking in. *Milestone‑2 path:*
cleanest — AMR, Zoltan2 repartition, adaptive‑dt min‑reduce, and disjoint components are all Tessera
concerns the new solver simply calls; nothing in Beatnik's structured assumptions is in the way.
*Paper story:* strongest — "a portable unstructured closed‑surface Z‑model solver on Tessera+Canopy,"
cleanly reusable, no structured‑grid baggage.

### Recommendation

**Option (c): a new standalone solver library, Beatnik untouched.** The decisive facts: (1) Beatnik's
only cleanly reusable asset is the *BR‑strategy design*, and designs are cheap to replicate but
expensive to depend on; (2) its structured‑grid coupling is load‑bearing and pervasive
(`SurfaceMesh.hpp:37`, heFFTe `ZModel.hpp:183`, periodic BCs `ExactBRSolver.hpp:112-121`), so any
"extend" path drags periodic/spectral machinery into a free‑space bubble; (3) (c) uniquely gives a
single, self‑owned place for the moving spec and the cleanest Milestone‑2 runway; (4) it carries zero
risk to existing rocketrig results, which matters for a code with users. Reuse Beatnik as a **reference
design and a source of small ported kernels**, and reuse its Canopy‑wrapping backend as the template
for the new ε‑aware binding.

### The strongest case against my own recommendation

The honest counter‑argument is **duplication and drift**. Beatnik and the new library will both carry a
`BRSolverBase`‑shaped interface, a TVD‑RK3, `dot/cross/BR` primitives, and a Canopy binding. Today
there is exactly one FMM‑over‑Canopy integration in existence (`FmmBRSolver.hpp`); (c) creates a second,
and the ε‑aware vector kernel work — the single longest pole (background §"longest‑lead item") — will
be needed by *both* the flat rocketrig FMM and the bubble. If that kernel lives only in the new
library, rocketrig can't share it; if it lives in Canopy (where it belongs), then the "standalone"
library still shares Canopy with Beatnik anyway, weakening the independence argument. There is a real
world in which **(b)** — extracting Beatnik's BR interface + RK + Operators into a tiny shared
`zmodel-core` header library that *both* Beatnik and the new solver depend on — eliminates that
duplication and gives one home for the BR contract and the RK weights, at the cost of a modest,
bounded refactor of Beatnik (move ~250 lines, no behavior change). If the collaborator's method turns
out to share more with rocketrig than background.md implies (same kernel, same RK, only the mesh
differs), (b) ages better than (c).

**Why I still land on (c):** the shared asset that actually matters — the ε‑aware vector Biot–Savart
expansion — belongs in **Canopy**, not in a shared solver‑core, and once it's in Canopy *both* solvers
get it regardless of (b) vs (c). What's left to "share" is ~250 lines of RK/interface boilerplate whose
duplication is cheaper than a dependency edge between two research codes on different meshes with
different BC models. Duplication of 250 lines is a smaller long‑run tax than coupling two codebases'
release cycles. I'd revisit if the refactor to extract `zmodel-core` proves trivial *and* the
collaborator confirms the kernel/RK are truly shared — that's the tripwire for switching (c)→(b).

---

## Q3 — End‑to‑end walkthrough (Q1 + Q2 adopted)

Reference config (background §7): potential formulation, explicit RK3, regularized free‑space ε‑aware
BR via Canopy FMM, Laplace–Beltrami viscous regularization, split AMR, surgery off. Libraries:
**Tessera** (mesh + operators), **Solverlib** (new standalone, option c), **Canopy** (FMM).

> **Caveat on "potential formulation":** background.md §4/§6 says the state is a single per‑vertex
> scalar, but *both* reference codes carry a **vector `mu`**. The walkthrough is written against a
> **generic per‑vertex state field** whose width (1 scalar vs 3‑vector) is a Solverlib compile‑time
> choice, so the design is correct either way. This is the #1 method‑owner question.

### Init

- **Mesh:** Solverlib driver calls `buildIcosphere(mesh, subdiv)` (`Tessera_MeshBuilder.hpp:237-244`)
  → replicated unit sphere; then `facePartitionByAxis` (`Tessera_Distribute.hpp:101`) →
  `distribute(mesh, halo, owner)` (`:169`) → `haloExchange(mesh, halo)`. Tessera **builds and owns**
  the AoSoAs.
- **AoSoA field pack (compile‑time, in the driver):** the `Mesh` type is instantiated with vertex user
  fields = `{ state[W], velocity[3], omega[3], gamma[3], curvature, forcing, work... }` where `W` is 1
  or 3. Because the pack is a template parameter (`Tessera_Fields.hpp:41-46`), **the driver declares it
  once**; Solverlib and Canopy receive the concrete `Mesh` type. Tessera owns the storage; Solverlib
  owns the *meaning* of each user field.
- **Per‑vertex state:** position (core field, `VertexField::Position`) + `state[W]` (the potential /
  `mu`). Everything else (velocity, omega, gamma, curvature, forcing) is derived scratch, recomputed
  each RK stage.

### One RK3 stage (annotated)

```
# Solverlib::step(dt):  TVD-RK3, 3 stages, weights from Beatnik TimeIntegrator.hpp:80-140 (ported)
for stage in {0,1,2}:
  # --- refresh handles: MANDATORY after any topology op last step (see hazard note) ---
  pos   = mesh.vertexSlice<Position>()                      # [Tessera] re-slice, no host copy
  state = mesh.vertexSlice<State>()

  # (1) LOCAL: geometry + operators  --- all one-deep-halo ---
  haloExchange(mesh, halo)                                  # [Tessera] LOCAL comm (1-deep)
  reduceVertexFromFaces(mesh, geom, faceSlice, vnormal, AreaWeightOp) # [Tessera reduce + Solverlib op]
  reduceVertexFromFaces(mesh, geom, faceSlice, varea, ThirdAreaOp)    # [Tessera reduce + Solverlib area convention]
  applyStencil(mesh, stencil, W_grad, state, dstate)        # [Tessera] surface gradient of state
  omega = muToOmega(state, metric, vnormal)                 # [Solverlib] PROBLEM-SPECIFIC (ZModel.hpp:82-91)
  applyStencil(mesh, stencil, W_lap, state, lap_state)      # [Tessera] Laplace-Beltrami apply
  curvature = meanCurvature(vnormal_grads, metric)          # [Solverlib] convention-dependent sign

  # (2) assemble Canopy source strength gamma = omega * local_area
  gamma = omega * varea                                     # [Solverlib] per-vertex, on device

  # (3) GLOBAL: Birkhoff-Rott via FMM
  canopy.setup<Position,Gamma>(mesh.vertices(), nOwned)     # [Canopy] GLOBAL: tree + Zoltan2 + migrate (!)
  canopy.solve<Position,Gamma>(mesh.vertices(), grad=false) # [Canopy] GLOBAL FMM
  br_vel = canopy.velocity()                                # [Canopy] internally-owned View
  # NOTE: br_vel indexed in CANOPY's post-migration order, not Tessera's — see MPI section.

  # (4) LOCAL: forcing + state RHS
  forcing = bernoulli(br_vel, omega, pos) + tension(curvature)  # [Solverlib] PROBLEM-SPECIFIC (ZModel.hpp:329-360)
  applyStencil(mesh, stencil, W_grad, forcing, grad_forcing)    # [Tessera] surface gradient of forcing
  d_state = zmodelRhs(grad_forcing, lap_state, ...)             # [Solverlib] d(state)/dt
  d_pos   = br_vel + bulk                                       # [Solverlib] d(position)/dt = velocity

  # (5) RK3 combine (AXPY over owned vertex slices, NOT grid index spaces)
  rk3_combine(stage, pos, state, d_pos, d_state, dt)       # [Solverlib] over Tessera slices

# (6) adaptive dt for next step
dt_next = globalMin(mesh, local_dt_estimate)              # [Tessera thin wrapper] GLOBAL min-reduce
```

**Local vs global:** everything in (1),(4),(5) is local (1‑deep halo) or vertex‑local; **only (3) FMM
and (6) min‑reduce are global.** This matches background §2 exactly — one long‑range interaction, one
global sync.

### Boundaries (what crosses, by value / reference / handle)

- **Tessera → Solverlib:** the `Mesh&` (reference) and Cabana **slices** (lightweight handles, ~by
  value but aliasing device memory). Slices are valid only between topology ops (hazard below) —
  this is now enforced by a guard, not just a discipline (see §0.2's "Ownership‑hazard
  remediation").
- **Solverlib → Canopy:** the vertex AoSoA (by reference) via `setup/solve<Position,Gamma>`. **Canopy
  needs, per vertex: position (`double[3]`) and γ = sheet‑strength × local area (a 3‑vector).**
  **Solverlib computes γ** in step (2); Tessera supplies the area, Solverlib supplies ω and the product.
- **Canopy → Solverlib:** `velocity()` — an internally‑owned View (handle, by const‑ref). Solverlib
  copies/reads it device‑side into the mesh `velocity` slice.
- **Ownership hazards (flagged):**
  1. **Tessera reallocation:** after *any* `refine`/`migrate`/`loadBalance` (AMR or LB steps),
     **every previously held bare slice dangles** (`Tessera_Migrate.hpp:236,275`,
     `Tessera_RefineParallel.hpp:412-413`). Solverlib **must re‑slice at the top of every stage** and
     must never cache a slice across a topology op — and now **there is a guard**: Tessera's
     `GenerationHandle` (`Tessera_GenerationGuard.hpp`, see §0.2's "Ownership‑hazard remediation")
     aborts with a diagnostic if a stale slice is copied (e.g. captured into a `KOKKOS_LAMBDA`) after
     a topology op, instead of silently reading dangling storage. Solverlib should still use
     `mesh.vertexSlices<...>()` (`Tessera_Mesh.hpp:211-225`) to re‑slice everything it needs at the
     top of every stage as a single structural call — the guard catches a violation of that
     discipline, it doesn't replace having the discipline. **[remedy implemented in Tessera —
     Tessera_Mesh.hpp, Tessera_GenerationGuard.hpp]**
  2. **Canopy in‑place migration:** `canopy.setup` **permutes and migrates the AoSoA in place**
     (`Canopy_TreePartitioner.hpp:766`) — it reorders and *moves across ranks* the very vertices Tessera
     owns. This is the sharpest hazard: after `canopy.setup`, Tessera's connectivity/CSR no longer
     matches the AoSoA row order, and vertices may have changed rank. **See MPI section for the
     mandatory reconciliation.** A callee (Canopy) resizing/reordering the caller's AoSoA is exactly the
     orphaning hazard the task names — **Canopy must operate on a *copy* / a separate particle AoSoA,
     not Tessera's mesh AoSoA.** (Beatnik's `FmmBRSolver` already does this — it repacks nodes into a
     private particle AoSoA, `FmmBRSolver.hpp:150-196` — and Solverlib must do the same.)

### MPI: partitioning, migration, halos, and the two‑partition problem

- **Tessera partitions by mesh** (faces by geometry or Zoltan2 on centroids), keeping connectivity +
  1‑deep halos coherent (`Tessera_Distribute.hpp`, `Tessera_MeshMigrate.hpp`).
- **Canopy partitions by octree leaves via its own Zoltan2 RCB and migrates particles to its owners**
  (`Canopy_TreePartitioner.hpp:362-400,766`), with **no API to accept an external map**. The two
  partitions **will not agree in general.**
- **Reconciliation (the critical design point):** Solverlib must treat Canopy as a black‑box N‑body
  service over a *transient particle set*, not over the mesh:
  1. Pack `{position, γ}` from Tessera‑owned vertices into a **separate particle AoSoA** (never hand
     Canopy the mesh AoSoA), carrying an origin tag = `{owning_rank, local_index}` (or global vertex id)
     per particle so results can be routed home.
  2. `canopy.setup/solve` — Canopy freely repartitions/migrates *its copy*.
  3. Scatter `velocity()` **back to the originating vertex** using the origin tag (an all‑to‑all keyed
     by `{rank,local}` — Tessera's `allToAllV` (`Tessera_AllToAllV.hpp:63`) is the right primitive).
  Tessera's mesh partition and halos are **never perturbed** by Canopy. Cost: one extra pack + one
  all‑to‑all per FMM call — acceptable, and it fully decouples the two partitioners.
  - *They must agree:* never. *May differ:* always. *Must be reconciled:* every FMM call, via the
    origin‑tag round‑trip. **[inferred design]**
- **When vertices move ranks (Tessera LB):** `migrate` rebuilds connectivity + 1‑deep halo
  (`Tessera_MeshMigrate.hpp:665-669`) and reassigns AoSoAs. Solverlib re‑slices and rebuilds the vertex
  stencil (`buildVertexStencil`) after LB. **Known Issue to route around:** distributed `refine` clears
  the halo and does **not** rebuild it (`Tessera_RefineParallel.hpp:712-714`); the rebuild currently
  only happens inside `migrate`. So the loop order after an AMR step must be **refine → migrate (to
  rebuild halo) → re‑slice → rebuild stencil**, or Solverlib must call a halo rebuild explicitly (none
  exists standalone today — flag).

### AMR (split‑based, surgery off)

- A refinement step lands **between timesteps** (not mid‑RK): mark (`markByQuality`,
  `Tessera_MarkQuality.hpp`) → `refine(mesh, halo, mask, policy)` (`Tessera_RefineParallel.hpp:150`).
- **What it invalidates:** all slices (realloc), the halo (cleared, `:712-714`), the vertex stencil
  (new vertices/valences), any cached vertex normals/areas, and Canopy's tree (rebuilt next `setup`
  anyway). New midpoint vertices get fields via the **RefinePolicy** (`Tessera_RefinePolicy.hpp`) —
  Solverlib must supply a policy that interpolates the *state* field correctly (linear default exists;
  a curvature‑aware policy is a hook, not yet implemented). Edge user fields are **reset** by refine
  (`Tessera_Refine.hpp:64-66`) — Solverlib must keep no authoritative state on edges.
- **Irregular vertices:** split AMR creates non‑valence‑6 vertices; Laplace–Beltrami is first‑order
  inconsistent there (background §"Second milestone"). This is expected numerics — documented on the
  Solverlib weight‑builder, not a Tessera bug (see Q1 note).

### GPU residency — where data could accidentally hit the host, and the guard

1. **Tessera slices** are device Views — `applyStencil`, `reduceVertexFromFaces`, `haloExchange` are all
   device kernels + GPU‑aware MPI (`Tessera_HaloExchange.hpp:172-175`). No host copy. ✅
2. **Canopy** is GPU‑resident with device‑pointer MPI (`Canopy_P2P.hpp:572-578`). ✅
3. **The pack/scatter reconciliation** (particle AoSoA build + origin‑tag all‑to‑all): must be a device
   kernel + GPU‑aware `allToAllV`. Tessera's `allToAllV` is **host‑staged** (`Tessera_AllToAllV.hpp:63`,
   `std::vector`), so a naïve use **round‑trips through host every FMM call** — the single most likely
   accidental host landing. *Remedy: use a device‑resident all‑to‑all for the velocity scatter (the
   same GPU‑aware pattern `haloExchange`/`migrate` already use), not the host `allToAllV`.* Flag as a
   Solverlib requirement. **[inferred]**
4. **Zoltan2 LB** gathers centroids to host by design (`Tessera_Zoltan2Balancer.hpp`) — but LB is
   between‑step and infrequent, so a host touch there is acceptable (it does not occur in the RK loop).
5. **I/O** (`writeMesh`) is collective HDF5 — host staging is inherent and out of the timestep loop. ✅
6. **Validation diagnostics** (below) must reduce on device and only bring **scalars** to host.

### Validation hooks (for the oracle when it lands)

- **Enclosed volume:** `V = (1/3) Σ_faces (centroid · areaNormal)` — a device reduction over faces
  using Tessera's `faceArea`/`faceNormalRaw` + a `globalMin`/sum wrapper. Attach as a Solverlib
  diagnostic callable each step; target relative error ~1e‑13 (background §6). Bring only the scalar to
  host.
- **Interface trajectory / cross‑section:** dump `writeMesh(mesh, stem_step)` at checkpoints (XDMF for
  Paraview) and a reduced centroid‑height time series.
- **Max sheet strength:** `globalMax(‖omega‖)` device reduction each step.
- **Oracle harness:** because the parallel state is distributed, comparison to the single‑rank oracle
  needs a **canonical global ordering** — dump by global vertex id (HDF5 already writes dense global
  numbering via `MPI_Exscan`, `Tessera_HDF5Writer.hpp`), then diff against the serial reference
  offline. Put these hooks behind a compile‑time/​runtime flag so they cost nothing in production runs.

---

## Assumptions I made

1. **RK3 = TVD RK3** as in Beatnik (`TimeIntegrator.hpp:80-140`), not zmodel‑kokkos's embedded adaptive
   5‑stage scheme (`TimeIntegrator.hpp:181-227`); background §7 says "explicit RK3," which matches
   Beatnik, not the reference C++. The adaptive‑dt min‑reduce (§ Second milestone) is a *step‑size*
   control layered on RK3, not the reference's embedded error estimator.
2. **Sources == targets** for the BR sum (interface vertices are both), consistent with all three codes.
3. **γ is a per‑vertex 3‑vector** = ω·(local area), assembled by Solverlib; Canopy consumes it as its
   per‑particle strength. This presumes the ε‑aware vector kernel is added to Canopy (see below).
4. **The compile‑time field pack is declared in the driver** and threaded as a concrete `Mesh` type to
   Solverlib and Canopy; no runtime field addition is needed for Milestone 1.
5. **Canopy must not operate on Tessera's mesh AoSoA directly** — Solverlib builds a private particle
   AoSoA. This is inferred from Canopy's in‑place migration, and mirrors Beatnik's existing FMM backend.
6. The ε‑aware **vector far‑field expansion** lands in **Canopy** (not Solverlib), so both rocketrig‑FMM
   and the bubble share it. This is where the "shared asset" argument in Q2's steelman resolves.

## Open questions for Jason

1. **Where does the ε‑aware vector BR kernel live — Canopy or Solverlib?** My design puts it in Canopy
   (near‑total rewrite of `LaplaceKernel.hpp`+`SphericalCoefficients.hpp`, per §0.4). If you'd rather
   prototype it in the solver first and upstream later, the reconciliation boundary changes. This is the
   longest pole (background §"longest‑lead item") — worth deciding early.
2. **Is a device‑resident all‑to‑all acceptable to add** (for the velocity scatter), or should Solverlib
   live with Tessera's current host‑staged `allToAllV` for the prototype? The former avoids a per‑step
   host round‑trip; the latter is less code now.
3. **Confirm the "core is settled" boundary for one specific case:** the post‑refine halo is *not*
   rebuilt except inside `migrate` (Known Issue). Is Solverlib expected to always follow `refine` with a
   `migrate` to get a halo back, or do you want a standalone halo‑rebuild exposed? (This is arguably a
   Tessera gap the milestone forces, even under "design on top.")
4. **Name/scope of the new Solverlib** — standalone repo, or a sibling under this project? Affects the
   dependency and packaging story for publication.

## Questions for the method owner (batch into one email)

1. **State variable:** is the shipped method's per‑vertex unknown a **scalar potential** (as
   background.md §4/§6 states) or the **3‑vector `mu`** both reference codes implement? The whole state
   layout and RK width depend on this.
2. **Local‑operator family:** does the final method use **cotangent 1‑ring** weights (as §2/§6 imply) or
   **RBF‑FD 2‑ring** (as both references implement)? If RBF‑FD, we need the kernel, shape parameter,
   polynomial degree, and stencil radius; if cotangent, the exact weight formula and boundary handling.
3. **Vertex normal convention:** tangent cross‑product (`zmodel-kokkos ZModel.hpp:64-73`) or area‑
   weighted face‑normal average (`zmodel-unstructured Mesh.py:403-410`)? They disagree; the oracle needs
   one.
4. **Vertex area definition:** barycentric / Voronoi / ⅓‑triangle? (Needed for γ = ω·area to 1e‑13.)
5. **Mean‑curvature sign and formula:** `½tr(g⁻¹II)` (reference) vs a cotangent‑Laplacian‑of‑position
   estimator — and the outward sign convention.
6. **ε convention:** is ε added squared (`+ε²`, as `zmodel-kokkos BirkhoffRott.h:146`) or un‑squared
   (`+ε`, as `Beatnik Operators.hpp:134`)? And the ε‑to‑resolution scaling for the unstructured mesh.
7. **ε‑aware far‑field expansion:** the treecode basis/order you use (Cartesian Taylor of
   `(r²+ε²)^{-1/2}`?) and its accuracy target — Canopy's far field must reproduce it exactly.
8. **`mu → omega` mapping** convention on the unstructured surface (metric/basis), to match
   `zmodel-kokkos ZModel.hpp:82-91`.
9. **Quantitative oracle targets** (background §6): the reference run + checkpoints, the enclosed‑volume
   tolerance (~1e‑13), max‑sheet‑strength growth curve, and cross‑sectional profile format.
10. **Relationship between the two zmodel reference codes** and the forthcoming Python — which is the
    binding target, and are the metric/curvature index bugs in `zmodel-kokkos` (`ZModel.hpp:79-84,168`)
    present in the version you'll validate against?
