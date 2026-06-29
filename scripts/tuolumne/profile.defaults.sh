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
# Tuolumne committed defaults for Tessera.
# Sourced automatically by scripts/lib/tessera_env.sh.
#
# To override for your checkout, create scripts/tuolumne/profile.local.sh
# (gitignored). Never edit this file for personal settings.

# Build mode: only manual is active (no spack package for Tessera yet).
TESSERA_BUILD_MODE="${TESSERA_BUILD_MODE:-manual}"

# Spack environment path.
TESSERA_SPACK_ENV="${TESSERA_SPACK_ENV:-${HOME}/spack_envs/tuolumne_trilinos/}"

# Out-of-tree build directory.
TESSERA_BUILD_DIR="${TESSERA_BUILD_DIR:-${TESSERA_REPO}/build-tuolumne}"
