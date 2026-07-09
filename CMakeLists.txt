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

# C is enabled alongside CXX because FindHDF5's compiler-wrapper probe (used
# when no HDF5 CMake config package is installed, e.g. Cray's parallel HDF5
# module) try_compiles a .c test program; without a C compiler configured that
# probe hard-errors instead of falling back gracefully (Step 8, any system).
project(Tessera VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ───────────────────────────────────────────────────────────────────
option(Tessera_ENABLE_TESTING   "Build and register the Tessera test suite"  OFF)
option(Tessera_ENABLE_EXAMPLES  "Build the Tessera example programs"          OFF)
option(Tessera_ENABLE_PROFILING "Enable Tessera internal profiling"           OFF)
set(Tessera_PROFILING_LEVEL "" CACHE STRING
    "Profiling detail level: 0=off, 1=basic phases, 2=detailed sub-phases, 3=verbose")
set_property(CACHE Tessera_PROFILING_LEVEL PROPERTY STRINGS "" 0 1 2 3)

# Resolve the effective profiling level. If Tessera_ENABLE_PROFILING is ON and
# no explicit level was given, default to level 1 (basic phases).
# Tessera_ENABLE_PROFILING=OFF is an authoritative kill switch: it forces the
# level to 0 regardless of any Tessera_PROFILING_LEVEL the user supplied.
if(NOT Tessera_ENABLE_PROFILING AND NOT Tessera_PROFILING_LEVEL STREQUAL "")
  set(_tessera_profiling_level 0)
elseif(Tessera_PROFILING_LEVEL STREQUAL "")
  if(Tessera_ENABLE_PROFILING)
    set(_tessera_profiling_level 1)
  else()
    set(_tessera_profiling_level 0)
  endif()
else()
  set(_tessera_profiling_level ${Tessera_PROFILING_LEVEL})
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Kokkos REQUIRED)
find_package(Cabana REQUIRED)
find_package(MPI    REQUIRED)
# Trilinos (Zoltan2 + Teuchos) — internal loadBalance() (Step 7b). Required: the
# only consumer, src/Tessera_Zoltan2Balancer.hpp, is unconditionally compiled into
# the header-only interface target, so Trilinos must resolve at configure time
# same as Kokkos/Cabana/MPI above.
find_package(Trilinos REQUIRED)
# I/O — parallel HDF5 (Step 8). Step 8 uses collective MPI-IO, so a *parallel*
# HDF5 is required on every system. A bare find_package can silently resolve a
# serial HDF5 (e.g. an OS /usr/lib64 build) when the parallel one is not on
# CMAKE_PREFIX_PATH; the guard below turns that into a loud configure error.
set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR
        "Found HDF5 at ${HDF5_INCLUDE_DIRS} but it is NOT parallel "
        "(HDF5_IS_PARALLEL=FALSE). Tessera I/O (Step 8) requires an "
        "MPI-parallel HDF5. Point CMake at one via -DHDF5_ROOT=<prefix> or "
        "add its prefix to CMAKE_PREFIX_PATH.")
endif()

# ── Library target ────────────────────────────────────────────────────────────
# Interface (header-only) target until source files are added under src/.
add_library(Tessera INTERFACE)
add_library(Tessera::Tessera ALIAS Tessera)

target_link_libraries(Tessera INTERFACE
    Kokkos::kokkos
    Cabana::Core
    MPI::MPI_CXX
    ${Trilinos_LIBRARIES}
    ${HDF5_C_LIBRARIES}
)

target_include_directories(Tessera INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(Tessera SYSTEM INTERFACE
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
)

if(_tessera_profiling_level GREATER 0)
    set(Tessera_ENABLE_PROFILING ON CACHE BOOL
        "Enable Tessera internal profiling" FORCE)
    target_compile_definitions(Tessera INTERFACE
        TESSERA_ENABLE_PROFILING
        TESSERA_PROFILING_LEVEL=${_tessera_profiling_level}
    )
endif()

# ── Install ───────────────────────────────────────────────────────────────────
option(Tessera_INSTALL_TEST_EXECUTABLES "Install built test executables to share/Tessera/tests" OFF)
option(Tessera_INSTALL_EXAMPLES "Install built example executables to share/Tessera/examples" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS Tessera EXPORT TesseraTargets)
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

set(Tessera_CMAKE_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/Tessera)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TesseraConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TesseraConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TesseraConfig.cmake
    INSTALL_DESTINATION ${Tessera_CMAKE_CONFIG_DIR}
)
install(EXPORT TesseraTargets FILE TesseraTargets.cmake NAMESPACE Tessera::
        DESTINATION ${Tessera_CMAKE_CONFIG_DIR})
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TesseraConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TesseraConfigVersion.cmake
    DESTINATION ${Tessera_CMAKE_CONFIG_DIR}
)

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
