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
# DEV test-runner (not a release artifact). Submits a batch job to the pdebug
# queue that runs a ctest label from the out-of-tree build directory. Each test
# is launched by ctest via `flux run` inside this batch allocation.
#
# TESSERA_REPO must be exported in the submitting shell (flux copies this script
# to a temp dir, so it cannot locate the repo from its own path). Submit with:
#   export TESSERA_REPO=$(pwd)
#   flux batch scripts/tuolumne/run_unit_tests.flux [LABEL] [CTEST_ARGS...]
# Examples:
#   flux batch scripts/tuolumne/run_unit_tests.flux              # -> label 'unit'
#   flux batch scripts/tuolumne/run_unit_tests.flux unit -R keys

#FLUX: --job-name=tessera-unit
#FLUX: --queue=pdebug
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: -t 20m
#FLUX: --output=tessera-unit.{{id}}.out

set -euo pipefail

# TESSERA_REPO comes from the submitting environment (flux captures it at submit
# time). Source the resolver (activates the spack env + runtime_env.sh so
# MPICH_GPU_* etc. reach every flux-run sub-task).
: "${TESSERA_REPO:?export TESSERA_REPO=<repo root> before submitting this job}"
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

LABEL="${1:-unit}"
shift || true

echo "=== Tessera ${LABEL} tests ==="
echo "  System:    ${TESSERA_SYSTEM}"
echo "  Build dir: ${TESSERA_BUILD_DIR}"
echo "  Label:     ${LABEL}"
echo "  Extra:     $*"
echo "  Started:   $(date)"
echo ""

ctest --test-dir "${TESSERA_BUILD_DIR}" -L "${LABEL}" --output-on-failure "$@"
_rc=$?

echo ""
echo "=== ${LABEL} tests complete (rc=${_rc}): $(date) ==="
exit ${_rc}
