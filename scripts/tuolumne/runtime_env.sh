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
# Tuolumne launch-time environment variables for Tessera.
# Sourced automatically by scripts/lib/tessera_env.sh.
#
# DO NOT inline these in batch scripts — source the resolver instead.
# These vars must reach every process launched by Flux, including MPI ranks.

# Cray-MPICH GPU-aware communication + HIP/HMM for the MI300A APU.
# Required for device pointers in MPI and for the APU to reach host allocations.
# Harmless for CPU/Serial runs.
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE

# NOTE: the glibc static-TLS surplus fix (GLIBC_TUNABLES
# optional_static_tls) that the run binaries need is NOT set here — it must not
# be present in the *build* environment, where it segfaults the Cray linker.
# It is injected per test task via the ctest launcher instead
# (MPIEXEC_PREFLAGS `--env=GLIBC_TUNABLES=...` in run_cmake_toulumne.sh), so it
# reaches the Flux-launched task without polluting the compiler/linker env.

# OpenMP thread placement (also used by the OpenMP backend; harmless for Serial).
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
