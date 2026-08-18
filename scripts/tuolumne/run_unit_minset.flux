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
# Unit-test release runner for Tuolumne (Flux scheduler).
#
# Submit: flux batch scripts/tuolumne/run_unit_minset.flux
#
# Runs the full `unit` label (diagnostic, not ship-gated — see
# tests/CMakeLists.txt) from the out-of-tree build directory. ctest launches
# each test via `flux run --ntasks N --nodes=1 --exclusive --cores-per-task=1`
# (configured via MPIEXEC_* cmake args). The outer batch allocation provides
# the node; sub-tasks bind within it.

#FLUX: --job-name=tessera-unit-minset
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --queue=pdebug
#FLUX: --output=tessera-unit-minset.{{id}}.out

set -euo pipefail

# Source the resolver: activates spack env + runtime_env.sh (MPICH_GPU_*,
# HSA_XNACK, OMP_*, etc.) so they reach every flux-run sub-task.
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

echo "=== Tessera unit tests ==="
echo "  System:    ${TESSERA_SYSTEM}"
echo "  Build dir: ${TESSERA_BUILD_DIR}"
echo "  Label:     unit"
echo "  Started:   $(date)"
echo ""

cd "${TESSERA_BUILD_DIR}"
ctest -L unit --output-on-failure

echo ""
echo "=== unit tests complete: $(date) ==="
