############################################################################
# Copyright (c) 2024, JStewart28                                           #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Tessera library. Tessera is distributed under a #
# BSD 3-Clause license. For the licensing terms see the LICENSE file in   #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################
#
# Tessera test suite — tiered by label, selectable by backend.
#
# TIER LABELS
# -----------
#   regression  Full end-to-end tests composing the whole pipeline.
#               These form the ship gate — must pass before merging.
#               Run: ctest -L regression -R SERIAL          (Serial backend)
#                    ctest -L regression -R HIP              (HIP backend)
#                    ctest -L regression -R "SERIAL|HIP"     (both, gate)
#
#   unit        Single-component or utility tests. Diagnostic; not ship-gated.
#               Run: ctest -L unit
#
# BACKEND SUFFIXES
# ----------------
#   Test names follow the convention: <name>_<BACKEND>_np<N>
#     SERIAL  — Kokkos::Serial (CPU, single-threaded)
#     OPENMP  — Kokkos::OpenMP (CPU, multi-threaded)
#     HIP     — Kokkos::HIP (AMD GPU, MI300A on Tuolumne)
#   Select a backend: ctest -R SERIAL  or  ctest -R HIP
#
# RANK PARAMETERIZATION
# ---------------------
#   TESSERA_TEST_MPI_RANKS defaults to "1;2;3;4;5" (the gate rank set).
#   Override: cmake -DTESSERA_TEST_MPI_RANKS="1;2"
#
# GATE DEFINITION (must match CLAUDE.md, run_regression_minset.flux, and CI)
#   Label:    regression
#   Backends: SERIAL, HIP
#   Ranks:    1, 2, 3, 4, 5
#
# NOTE: a failing test is never silently excluded from the gate.
#   If a test is unstable, label it 'unit', record it in README Known Issues,
#   and report it — never promote a failing test to the gate.

set(TESSERA_TEST_MPI_RANKS "1;2;3;4;5" CACHE STRING
    "MPI rank counts over which to parameterize tests (gate uses 1-5)")

# ── Helper macro ──────────────────────────────────────────────────────────────
# tessera_add_test(
#   NAME     <base-name>           e.g. mesh_partition
#   BACKEND  <SERIAL|OPENMP|HIP>
#   TIER     <regression|unit>
#   RANKS    <rank-list>           default: "1" (override per call)
#   SOURCES  <source files>
# )
#
# Creates one executable and one CTest entry per (NAME, BACKEND, rank) triple.
# Test name: <NAME>_<BACKEND>_np<N>
# Labels:    <TIER> and <BACKEND> (multiple labels supported by CTest)
macro(tessera_add_test)
    cmake_parse_arguments(TAT "" "NAME;BACKEND;TIER" "RANKS;SOURCES" ${ARGN})

    if(NOT TAT_RANKS)
        set(TAT_RANKS "1")
    endif()
    if(NOT TAT_TIER)
        message(FATAL_ERROR "tessera_add_test: TIER is required (regression or unit)")
    endif()
    if(NOT TAT_BACKEND)
        message(FATAL_ERROR "tessera_add_test: BACKEND is required (SERIAL, OPENMP, or HIP)")
    endif()

    set(_exe tessera_test_${TAT_NAME}_${TAT_BACKEND})

    add_executable(${_exe} ${TAT_SOURCES})
    target_link_libraries(${_exe} PRIVATE Tessera::Tessera)

    if(Tessera_INSTALL_TEST_EXECUTABLES)
        install(TARGETS ${_exe} DESTINATION ${CMAKE_INSTALL_DATADIR}/Tessera/tests)
    endif()

    foreach(_np IN LISTS TAT_RANKS)
        set(_test_name ${TAT_NAME}_${TAT_BACKEND}_np${_np})
        add_test(
            NAME    ${_test_name}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${_np}
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${_exe}>
        )
        set_tests_properties(${_test_name} PROPERTIES
            LABELS "${TAT_TIER};${TAT_BACKEND}"
        )
    endforeach()
endmacro()

# ── Tests ─────────────────────────────────────────────────────────────────────
#
# Step 2 (core data model) unit tests. Each source internally exercises BOTH the
# host (Kokkos::Serial) and device (Kokkos::DefaultExecutionSpace, = HIP on
# Tuolumne) execution spaces, so the SERIAL- and HIP-labelled registrations run
# the same coverage; the label selects which the gate/ctest filter picks up.
# Single rank (no MPI decomposition exercised until Step 5).

# Structured-key (EdgeKey/FaceKey) determinism.
tessera_add_test(
    NAME     keys
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_keys.cpp
)
tessera_add_test(
    NAME     keys
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_keys.cpp
)

# Core data model: Mesh construction + core/user field-pack slices (double+float,
# Dim=3 and Dim=2).
tessera_add_test(
    NAME     data_model
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_data_model.cpp
)
tessera_add_test(
    NAME     data_model
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_data_model.cpp
)

# Serial icosphere builder + connectivity: Euler, edge/face incidence, vertex
# 1-ring CSR round-trip.
tessera_add_test(
    NAME     connectivity
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_connectivity.cpp
)
tessera_add_test(
    NAME     connectivity
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_connectivity.cpp
)

# Whole-tuple AoSoA migration primitive: round-trip scatter/shift/identity over
# 1-5 ranks (conservation + payload integrity). Multi-rank, so parameterized over
# the full rank set.
tessera_add_test(
    NAME     migrate
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate.cpp
)
tessera_add_test(
    NAME     migrate
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate.cpp
)

# Halo primitives: allToAllV neighbour topology exchange + HaloExchangePlan
# whole-tuple field sync (ring halo) over 1-5 ranks.
tessera_add_test(
    NAME     halo
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo.cpp
)
tessera_add_test(
    NAME     halo
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo.cpp
)

# Local 1->4 red refinement (Step 6a): uniform Euler/counts, partial shared
# midpoints + child incidence, default vs custom per-field interpolation policy.
# Single rank (parallel 2:1 balance is Step 6b).
tessera_add_test(
    NAME     refine
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine.cpp
)
tessera_add_test(
    NAME     refine
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine.cpp
)

# RefinementMode plumbing (conforming refinement, Task 1): a Conforming-typed
# mesh instantiates and its conditional closure face members round-trip; the
# public face user-field indices (FaceField::UserBegin, userFaceField<M>()) are
# bit-identical in both modes, and the Conforming face tuple is exactly two
# members larger. Type-layout property -> single rank.
tessera_add_test(
    NAME     refinement_mode
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_refinement_mode.cpp
)
tessera_add_test(
    NAME     refinement_mode
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_refinement_mode.cpp
)

# Serial closure kernel (conforming refinement, Task 2): the green/blue/red
# closure patterns and their inverse. Pins the hand-computed child triangles for
# each |S| in {0,1,2,3}, CCW winding, the blue lower-gid diagonal tie-break in
# both directions, rotation invariance of the emitted triangle set (why the
# tie-break is partition-independent), unclose o close == identity on a random
# partial mask over buildIcosphere(3), and -- for refineLocal() in Conforming
# mode with a PARTIAL mask -- Euler V-E+F == 2, exactly two faces per edge, no
# T-junction, outward winding, and the same vertex count as HangingNode2to1 mode.
# The same conformity checks are run on the hanging-node result and must fail
# there, so the case cannot go vacuous. Local by construction -> single rank.
tessera_add_test(
    NAME     refine_closure
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine_closure.cpp
)
tessera_add_test(
    NAME     refine_closure
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine_closure.cpp
)

# Stale-handle generation guard: a slice held across a count-changing op and
# then copied (as it would be when captured into a KOKKOS_LAMBDA) must abort;
# a slice held across haloExchange() (topology-preserving) must not. This
# exercises host-side generation bookkeeping, not device kernels, so it is
# registered SERIAL-only (no HIP duplicate, unlike the other unit tests).
tessera_add_test(
    NAME     staleslice_guard
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_staleslice_guard.cpp
)

# Raw geometric primitives (Milestone-1 operators): faceArea, faceNormalRaw,
# edgeVector, cotangentAtCorner on an analytic two-triangle mesh with exact
# hand-computed answers, checked host-side after device evaluation. Single rank.
tessera_add_test(
    NAME     geometry
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_geometry.cpp
)
tessera_add_test(
    NAME     geometry
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_geometry.cpp
)

# k-ring vertex stencil topology (buildVertexStencil): k=1 row equals the
# edge-derived 1-ring; k=2 row is the exact distance-<=2 set and a superset of
# k=1; self excluded. Single rank (topology derivation, gid == local).
tessera_add_test(
    NAME     stencil_topology
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_stencil_topology.cpp
)
tessera_add_test(
    NAME     stencil_topology
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_stencil_topology.cpp
)

# Weighted-stencil apply (applyStencil): analytic field f(p)=p_x, uniform
# weights, reference computed from (haloed) positions independent of the
# in-field halo path -- so a failure is unambiguously an apply/halo bug. Over
# 1-5 ranks to exercise partition boundaries.
tessera_add_test(
    NAME     apply_stencil
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_apply_stencil.cpp
)
tessera_add_test(
    NAME     apply_stencil
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_apply_stencil.cpp
)

# Face -> vertex reduction (reduceVertexFromFaces): a one-third-area op whose
# global owned-vertex sum equals the total surface area (global owned-face
# faceArea sum) -- a partition-independent identity. Over 1-5 ranks.
tessera_add_test(
    NAME     reduce_faces
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_reduce_faces.cpp
)
tessera_add_test(
    NAME     reduce_faces
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_reduce_faces.cpp
)

# Global scalar reduction (globalMin): min placed at different ranks confirms a
# real collective; single-rank identity. Host-side MPI wrapper, SERIAL-only
# (no device kernels), over 1-5 ranks.
tessera_add_test(
    NAME     global_reduce
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_global_reduce.cpp
)

# ── Regression tier (ship gate: label regression x {SERIAL,HIP} x ranks 1-5) ────

# Distributed mesh + deterministic ownership + 1-deep halo (Step 5). First
# end-to-end gate test: partition -> distribute -> ownership/1-ring invariants ->
# halo sync.
tessera_add_test(
    NAME     distribute
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_distribute.cpp
)
tessera_add_test(
    NAME     distribute
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_distribute.cpp
)

# Mesh migration + general (non-replicated) 1-deep halo rebuild (Step 7):
# external face assignment lands entities on requested ranks, ownership is a
# partition, every owned vertex's 1-ring is local, rank-count-independent topology
# checksum preserved, ghost halo sync. The general ghost builder deferred from 6b.
tessera_add_test(
    NAME     migrate_mesh
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate_mesh.cpp
)
tessera_add_test(
    NAME     migrate_mesh
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate_mesh.cpp
)

# Distributed 2:1-balanced refinement (Step 6b): cross-rank midpoint-gid
# agreement, 2:1 level invariant, owned-only Euler (uniform), fixpoint
# termination (adaptive). Owned-entity invariants; halo rebuild is Step 7.
tessera_add_test(
    NAME     refine_parallel
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_parallel.cpp
)
tessera_add_test(
    NAME     refine_parallel
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_parallel.cpp
)

# Distributed split-edge discovery (conforming refinement, Task 3): refine()'s
# Phase 2 advertises EVERY owned face's edges, so RefineResult::midpoints is a
# complete split-edge map -- a rank holding only the KEPT side of an edge a
# neighbour bisected still learns its midpoint gid, which is what the conforming
# closure (Task 4) needs. Pins round-1 exactness against a partition-free
# reference, the midpoint gids forming the contiguous block above the pre-refine
# global vertex count, non-vacuity (kept-side discovery actually occurs at
# ranks >= 2), and coordinator-decided soundness+completeness over three further
# adaptive rounds where the 2:1 fixpoint really propagates.
tessera_add_test(
    NAME     refine_splitedges
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_splitedges.cpp
)
tessera_add_test(
    NAME     refine_splitedges
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_splitedges.cpp
)

# Distributed CONFORMING refinement (conforming refinement, Task 4) -- the
# milestone test for the feature. Over three successive ADAPTIVE rounds: every
# global edge has exactly two incident faces, owned-only Euler V-E+F == 2 (the
# criterion the hanging-node mode fails under an adaptive mask), no vertex
# strictly inside an edge, ownership still a partition, midpoint gids still
# agreed cross-rank, 2:1 balance on the RED layer, and unclose o close reproducing
# the visible and red layers bit-for-bit. The same mask is run on a
# HangingNode2to1 mesh as a non-vacuity control and MUST fail the conformity
# checks there. Plus the degenerate empty-mask (identity) and full-mask (no kept
# faces, so the closure is inert and the counts equal the hanging-node mode's)
# cases. Prints the closure-face fraction and |S| histogram per round.
tessera_add_test(
    NAME     refine_conforming
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_conforming.cpp
)
tessera_add_test(
    NAME     refine_conforming
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_conforming.cpp
)

# Internal Zoltan2 loadBalance() (Step 7b): rebalancing a deliberately
# maximally-imbalanced mesh reduces the max-owned-face imbalance while
# preserving every migrate() distribution invariant and the rank-count-
# independent topology checksum.
tessera_add_test(
    NAME     loadbalance
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_loadbalance.cpp
)
tessera_add_test(
    NAME     loadbalance
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_loadbalance.cpp
)

# Parallel HDF5 + XDMF writer/reader round-trip (Step 8): write -> read ->
# compare topology + user-field checksums, on-disk gid checksum independent
# of rank count, and a post-read corrupt/haloExchange ghost-restoration check.
tessera_add_test(
    NAME     io
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_io.cpp
)
tessera_add_test(
    NAME     io
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_io.cpp
)

# Edge-length quality-based refinement marking (Step 10a): markByQuality's
# marked owned-face gid set matches an independent, partition-free reference
# (rank-count independence), monotone edge-length progress after refine(),
# refine() post-conditions (2:1 balance, midpoint agreement), and the
# empty-mask/full-mask degenerate cases.
tessera_add_test(
    NAME     markquality_edge
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_edge.cpp
)
tessera_add_test(
    NAME     markquality_edge
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_edge.cpp
)

# Curvature/dihedral quality-based refinement marking (Step 10b): CurvatureCriterion
# marks both faces incident to any sharp-fold edge. The dihedral gather is routed
# through edge coordinators (allToAllV), NOT the 1-ring halo, so the marked set is
# rank-count- AND boundary-straddle-independent -- pinned by matching a partition-
# free geometric reference at ranks 1-5. refine() post-conditions still hold.
tessera_add_test(
    NAME     markquality_curv
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_curv.cpp
)
tessera_add_test(
    NAME     markquality_curv
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_curv.cpp
)
