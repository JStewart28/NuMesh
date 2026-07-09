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
