#!/usr/bin/env bash
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
# Configure Tessera for a local workstation (Serial + OpenMP; no GPU).
#
# Usage (run from a build directory OR from the repo root):
#   mkdir build-local && cd build-local
#   bash ../run_cmake.sh [-DCMAKE_PREFIX_PATH=<deps>] [extra cmake args]
#
# Dependencies (Kokkos, Cabana, MPI) must be installed and findable via
# CMAKE_PREFIX_PATH or your system package manager.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTessera_ENABLE_TESTING=ON \
    -DTessera_ENABLE_EXAMPLES=ON \
    -DTessera_ENABLE_PROFILING=ON \
    -DTessera_PROFILING_LEVEL=0 \
    "$@" \
    "${SCRIPT_DIR}"
