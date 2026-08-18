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
# Tessera examples — ordered subdirectories, one example each.
# Each example's accepted arguments are documented in README.md.
#
# Examples:
#   01_hello_tessera  — minimal build verification (Kokkos + MPI linkage check)
#   02_mesh_pipeline  — end-to-end build/partition/refine/IO pipeline demo

add_subdirectory(01_hello_tessera)
add_subdirectory(02_mesh_pipeline)
