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
# No tests yet — add tessera_add_test() calls here as features are implemented.
#
# Regression example (uncomment and adapt):
#
# tessera_add_test(
#     NAME     mesh_partition
#     BACKEND  SERIAL
#     TIER     regression
#     RANKS    ${TESSERA_TEST_MPI_RANKS}
#     SOURCES  test_mesh_partition.cpp
# )
#
# tessera_add_test(
#     NAME     mesh_partition
#     BACKEND  HIP
#     TIER     regression
#     RANKS    ${TESSERA_TEST_MPI_RANKS}
#     SOURCES  test_mesh_partition.cpp
# )
#
# Unit test example (single rank, no backend suffix requirement):
#
# tessera_add_test(
#     NAME     partition_utils
#     BACKEND  SERIAL
#     TIER     unit
#     RANKS    "1"
#     SOURCES  test_partition_utils.cpp
# )
