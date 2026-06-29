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
# Regression gate wrapper for Tuolumne (Flux scheduler).
#
# Submit: flux batch scripts/tuolumne/run_regression_minset.flux
#
# Gate definition (must match CLAUDE.md, tests/CMakeLists.txt, and CI):
#   Label:    regression
#   Backends: SERIAL, HIP
#   Ranks:    1, 2, 3, 4, 5
#
# ctest launches each test via `flux run --ntasks N --nodes=1 --exclusive
# --cores-per-task=1` (configured via MPIEXEC_* cmake args). The outer
# batch allocation provides the node; sub-tasks bind within it.

#FLUX: --job-name=tessera-regression-gate
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --output=tessera-regression-gate.{{id}}.out

set -euo pipefail

# Source the resolver: activates spack env + runtime_env.sh (MPICH_GPU_*,
# HSA_XNACK, OMP_*, etc.) so they reach every flux-run sub-task.
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

echo "=== Tessera regression gate ==="
echo "  System:    ${TESSERA_SYSTEM}"
echo "  Build dir: ${TESSERA_BUILD_DIR}"
echo "  Backends:  SERIAL HIP"
echo "  Ranks:     1 2 3 4 5"
echo "  Started:   $(date)"
echo ""

cd "${TESSERA_BUILD_DIR}"
ctest -L regression -R "SERIAL|HIP" --output-on-failure

echo ""
echo "=== Gate complete: $(date) ==="
