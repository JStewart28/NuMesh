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
# Full test sweep for Tuolumne (Flux scheduler) — every test (unit +
# regression), both backends.
#
# Submit: flux batch scripts/tuolumne/run_all_tests.flux
#
# Ranks:
#   SERIAL tests run at 1, 2, 3, 4, 5 processes.
#   HIP    tests run at 1, 2, 3, 4 processes only — Tuolumne has 4 GPUs per
#          node, so a 5th rank would oversubscribe the devices.
# The `_np<N>` test-name suffix is the rank parameterization; the regex below
# selects SERIAL np1-5 and HIP np1-4.
#
# ctest launches each test via `flux run --ntasks N --nodes=1 --exclusive
# --cores-per-task=1` (configured via MPIEXEC_* cmake args). The outer batch
# allocation provides the node; sub-tasks bind within it.

#FLUX: --job-name=tessera-all-tests
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --queue=pdebug
#FLUX: -t 20m
#FLUX: --output=tessera-all-tests.{{id}}.out

set -euo pipefail

# Source the resolver: activates spack env + runtime_env.sh (MPICH_GPU_*,
# HSA_XNACK, OMP_*, etc.) so they reach every flux-run sub-task.
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

echo "=== Tessera full test sweep ==="
echo "  System:    ${TESSERA_SYSTEM}"
echo "  Build dir: ${TESSERA_BUILD_DIR}"
echo "  SERIAL:    ranks 1 2 3 4 5"
echo "  HIP:       ranks 1 2 3 4 (4 GPUs/node)"
echo "  Started:   $(date)"
echo ""

cd "${TESSERA_BUILD_DIR}"
ctest -R "SERIAL_np[1-5]$|HIP_np[1-4]$" --output-on-failure

echo ""
echo "=== Sweep complete: $(date) ==="
