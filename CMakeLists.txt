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

cmake_minimum_required(VERSION 3.21)

project(Tessera VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ───────────────────────────────────────────────────────────────────
option(Tessera_ENABLE_TESTING   "Build and register the Tessera test suite"  OFF)
option(Tessera_ENABLE_EXAMPLES  "Build the Tessera example programs"          OFF)
option(Tessera_ENABLE_PROFILING "Enable Tessera internal profiling"           OFF)
set(Tessera_PROFILING_LEVEL 0 CACHE STRING
    "Tessera profiling verbosity level (0–3)")

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Kokkos REQUIRED)
find_package(Cabana REQUIRED)
find_package(MPI    REQUIRED)
# Trilinos (Zoltan2 + Teuchos) — internal loadBalance() (Step 7b). Required: the
# only consumer, src/Tessera_Zoltan2Balancer.hpp, is unconditionally compiled into
# the header-only interface target, so Trilinos must resolve at configure time
# same as Kokkos/Cabana/MPI above.
find_package(Trilinos REQUIRED)

# ── Library target ────────────────────────────────────────────────────────────
# Interface (header-only) target until source files are added under src/.
add_library(Tessera INTERFACE)
add_library(Tessera::Tessera ALIAS Tessera)

target_link_libraries(Tessera INTERFACE
    Kokkos::kokkos
    Cabana::Core
    MPI::MPI_CXX
    ${Trilinos_LIBRARIES}
)

target_include_directories(Tessera INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(Tessera SYSTEM INTERFACE
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
)

if(Tessera_ENABLE_PROFILING)
    target_compile_definitions(Tessera INTERFACE
        TESSERA_ENABLE_PROFILING
        TESSERA_PROFILING_LEVEL=${Tessera_PROFILING_LEVEL}
    )
endif()

# ── cmake module path ─────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ── Format targets (clang-format) ─────────────────────────────────────────────
find_package(CLANG_FORMAT)
if(CLANG_FORMAT_FOUND)
    file(GLOB_RECURSE TESSERA_FORMAT_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    )
    if(TESSERA_FORMAT_SOURCES)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${TESSERA_FORMAT_SOURCES}
            COMMENT "Running clang-format on all source files"
        )
        add_custom_target(format-check
            COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror
                    ${TESSERA_FORMAT_SOURCES}
            COMMENT "Checking clang-format compliance"
        )
    endif()
endif()

# ── Subdirectories ────────────────────────────────────────────────────────────
if(Tessera_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(Tessera_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
