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
# Local workstation committed defaults for Tessera.
# Sourced automatically by scripts/lib/tessera_env.sh.
#
# To override for your checkout, create scripts/local/profile.local.sh
# (gitignored). Never edit this file for personal settings.

# Build mode: manual (hand-compiled out-of-tree).
TESSERA_BUILD_MODE="${TESSERA_BUILD_MODE:-manual}"

# Out-of-tree build directory.
TESSERA_BUILD_DIR="${TESSERA_BUILD_DIR:-${TESSERA_REPO}/build-local}"
