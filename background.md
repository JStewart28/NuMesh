# Background
This file contains background information about how this unstructured mesh library, Tessera, will be used to interact with other libraries Canopy (An FMM solver for far-field forces) and Beatnik/rocketrig, a fluid interface solver.
Those libraries are located at these paths, for reference:
- Beatnik/rocketrig problem: ~/spack_envs/beatnik/beatnik
- Canopy: ~/spack_envs/beatnik/canopy

## First milestone
A simple unstructured rising bubble, explicit time stepping, use the FMM solve Canopy for far-field forces. It exercises every part of the method that matters for parallel scaling while deferring the genuinely hard topology work. Below contains information about what the closed-surface z-model actually requires, flags, and possible "problems" worth knowing up front, and a note about disjoint-surfaces. There's another note on timing at the end.

A good reference problem is the rocketrig example in the beatnik. Most of the physics carries straight over — you already know the Birkhoff–Rott kernel and the ε-regularization. What's new is almost entirely the setting — closed surfaces in free space, on an unstructured mesh — rather than the math of the kernel. The single longest-lead item is porting the BR evaluation from the periodic/Fourier form to the free-space real-space FMM solver Canopy.

1. Why the rising-bubble / no-surgery target is the right first step

A single rising bubble is one closed surface evolving in time. It loads all four parallel-critical components — the global FMM, the halo-based local mesh operators, dynamic AMR, and a global timestep reduction — but avoids topology changes (pinch-off/merger) and the aggressive remeshing those require. You learn the requirements and find the technical difficulties without fighting connectivity changes on day one.

2. How the method decomposes: local vs. global work

This is the key structural point for parallelization. The update at each interface vertex, at every stage of the time step, splits into two kinds of work:

Local (1-ring stencils, i.e. one-deep halo): vertex normals and areas, mean curvature, the discrete Laplace–Beltrami operator (used for viscous regularization and curvature), and the surface gradient of the Bernoulli forcing. Each depends only on a vertex and its immediate neighbors — exactly one-deep-halo operations.
Global (all-to-all): the Birkhoff–Rott velocity — an N-body sum in which every interface point's velocity depends on every other point. This is the only long-range interaction in the method and is where the overwhelming majority of compute and communication lives. This is what Canopy accelerates.
Mental model: everything except one N-body sum is a local, halo-able stencil; the Birkhoff–Rott sum is the one global interaction. This is the same BR sum thats computed in the rocketrig model — just evaluated differently in the closed-surface/unstructured setting (see 4.a).

3. What Tessera Needs

  1. Unstructured AoSoA mesh built with Cabana AoSoAs. Mesh cells are triangles with vertices, edges, and faces.
  2. One-deep haloing + mesh refinement. All local operators above are 1-ring, so one-deep halos suffice for the first milestone.
  3. I/O that works with MPI. Perhaps the VTK or Silo library, or write HDF5 manually. This is a design decision. Must be able to write mesh connectivity information for visualization in paraview and any data associated with vertices, edges, or faces.

4. Technical difficulties worth knowing now

The BR evaluation — periodic/Fourier transitioning to free-space/FMM (the long pole). The work here is three adaptations to the closed-surface/unstructured setting:

(a) Free-space, not periodic. On the flat RT interface in rocketrig, we sum the BR spectrally (FFT / Fourier-series) on the uniform grid. That approach is tied to both the periodicity and the structured grid. A single bubble — and the general closed-surface methods — live in open space on an unstructured mesh, where neither holds. The real-space hierarchical sum (the FMM) is exactly the tool for that case, which I expect is why you built it. So this confirms the FMM, not the Fourier sum, is the right path here. (If I've misremembered and you're already doing free-space real-space sums, even better — then most of this is done.)

(b) Vector, real-space form of the kernel. In the closed-surface code the velocity is the regularized vector Biot–Savart sum

u(xᵢ) = (1/4π) Σⱼ [ (xᵢ − xⱼ) / (|xᵢ − xⱼ|² + ε²)^{3/2} ] × γⱼ,

where γⱼ is the per-vertex sheet-strength (vorticity) × local area, and sources and targets are the same point set (the interface vertices). On the flat periodic grid the same BR integral is written in a scalar Riesz/Fourier form; that scalar/FFT form is the piece that does not carry over. So the multipole expansion needs to be for this vector kernel. (Canopy-specific information)

(c) ε-aware far-field expansion. Same ε you already use, but in the real-space setting it has to live in the multipole/far-field expansion rather than as a spectral filter. A textbook 1/r Laplace expansion will be inaccurate at the moderate ranges where ε matters. My Python contains a working ε-aware Barnes–Hut treecode for exactly this vector kernel — that's the reference we must port and validate against.

A wider halo will not substitute for the FMM. The one-deep halo covers the local operators only; the BR sum is long-range and is the FMM's job. I mention it only so no effort goes into deepening halos to capture the global interaction — that's not the right method.

Use the "potential" formulation for this demo. The closed-surface method has two equivalent state representations; the potential form keeps the unknown a single per-vertex scalar with a purely local update, and avoids a global integration step the other form needs. (Note: this is about the state variable — the velocity is still the vector BR sum via FMM either way.) It's the simpler target for a first parallel implementation, and it's all the bubble needs.

## Second milestone
AMR: refinement is the easy 80%. Splitting (adding points where curvature grows) is straightforward. The full method also does edge collapse and edge flips; parallel collapse/flip with conflicts across partition boundaries is harder than refinement. For the rising bubble you can lean on split-dominated refinement, but it's worth leaving room in the data-structure design for collapse/flip later.

Dynamic load balancing. AMR concentrates points where curvature grows (the roll-up), so work becomes spatially imbalanced over time and needs dynamic repartitioning. Use the Trilinos library for load balancing. See Canopy_TreePartitinoner for reference. Load balancing should be configured to be external to the library or internal. Assume the external method is also via Trilinos.

Global timestep reduction. We use an adaptive dt, which is a global min-reduction once per step — cheap, but a genuine global synchronization point in the loop.

Operator behavior at irregular vertices is numerical, not a bug. The discrete Laplace–Beltrami operator is first-order and pointwise-inconsistent at irregular (non-valence-6) vertices. If AMR creates such vertices and you see odd local behavior, that's expected numerics, not an MPI race — flagging it to save a debugging goose-chase. (This didn't arise on the uniform square mesh, where every interior vertex is regular; it's a feature of the unstructured setting.)

5. On storing disjoint surfaces

Not needed for the no-surgery bubble — that's a single connected surface. It becomes relevant only after a pinch-off, when one surface becomes two or more components. The good news: the BR/FMM side is unaffected by how many components exist — it's just a global sum over all interface points regardless of component. Only the mesh operators (neighbor traversal, remeshing) need to respect component boundaries. So your instinct that "it's just vertices/edges/faces with connectivity" is right — defer it, but leave the door open in the data-structure design.

6. What I'll provide, and timing

I'm finalizing a piece of the algorithm right now — the topology-change (surgery) step and the remeshing that follows it — and I need a few more weeks to stabilize and clean it up so what I hand you is a tagged, tested and bullet-proof. I do plan to give you the full Python code; that's the plan, just shortly after I perfect the algorithm.

When I send the code it'll come with:

The exact kernel and ε-regularization, plus the reference ε-aware far-field expansion (the treecode), so the FMM port has a concrete target — and a short orientation note on how the closed-surface velocity evaluation differs from the flat/periodic version you already have (real-space vector BR vs. scalar Riesz/Fourier).
The single-rank code as an executable specification and validation oracle: for the rising bubble you can validate the parallel version against the serial interface trajectory, the enclosed-volume relative error (we hold this to ~1e-13), the growth of the maximum sheet strength, and a cross-sectional profile. I'll include a reference run and checkpoints so you have quantitative targets.
The definitions/conventions for the local operators (cotangent weights, normal/area conventions) so those match exactly.
7. Suggested reference configuration for the demo

Potential formulation; explicit RK3; regularized Birkhoff–Rott via your FMM (free-space, ε-aware); Laplace–Beltrami viscous regularization; split-based AMR; surgery off. That's the smallest configuration that still exercises everything that matters for scaling.

This is a great first target and I think it'll teach us a lot about the parallel requirements. I'll follow up in a few weeks with the bullet-proof Python code and the validation material.