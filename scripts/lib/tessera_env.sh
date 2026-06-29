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
# Tessera shared resolver — SOURCE this file (do not execute it) at the
# top of every batch script, before any Tessera commands.
#
# What it does:
#   1. Resolves TESSERA_REPO to the repo root.
#   2. Detects TESSERA_SYSTEM from hostname (overridable via env).
#   3. Sources profile.defaults.sh, then overlays profile.local.sh if present.
#   4. Exports derived variables (TESSERA_BIN_MODE, TESSERA_BUILD_DIR, etc.).
#   5. Activates the spack environment (unless TESSERA_NO_SPACK_ACTIVATE=1).
#   6. Sources scripts/<system>/runtime_env.sh if present (unless dry-run).
#   7. Defines the tessera_exe() helper function.
#
# Usage in a batch script:
#   source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"
#   # -or- (when TESSERA_REPO is not yet set):
#   source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib/tessera_env.sh"

# ── 1. Repo root ──────────────────────────────────────────────────────────────
# Resolve to two levels above this file: scripts/lib/ → scripts/ → repo root.
_TESSERA_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESSERA_REPO="${TESSERA_REPO:-$(cd "${_TESSERA_LIB_DIR}/../.." && pwd)}"
export TESSERA_REPO

# ── 2. System detection ───────────────────────────────────────────────────────
if [[ -z "${TESSERA_SYSTEM:-}" ]]; then
    case "$(hostname)" in
        tuolumne*) TESSERA_SYSTEM=tuolumne ;;
        *)         TESSERA_SYSTEM=local ;;
    esac
fi
export TESSERA_SYSTEM

_TESSERA_SYS_DIR="${TESSERA_REPO}/scripts/${TESSERA_SYSTEM}"

# ── 3. Profile: defaults then local override ──────────────────────────────────
if [[ -f "${_TESSERA_SYS_DIR}/profile.defaults.sh" ]]; then
    # shellcheck source=/dev/null
    source "${_TESSERA_SYS_DIR}/profile.defaults.sh"
fi
if [[ -f "${_TESSERA_SYS_DIR}/profile.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "${_TESSERA_SYS_DIR}/profile.local.sh"
fi

# ── 4. Derived variables ──────────────────────────────────────────────────────
TESSERA_BUILD_MODE="${TESSERA_BUILD_MODE:-manual}"
export TESSERA_BUILD_MODE

TESSERA_BIN_MODE="${TESSERA_BIN_MODE:-${TESSERA_BUILD_MODE}}"
export TESSERA_BIN_MODE

TESSERA_BUILD_DIR="${TESSERA_BUILD_DIR:-${TESSERA_REPO}/build-${TESSERA_SYSTEM}}"
export TESSERA_BUILD_DIR

# ── 5. Spack environment activation ──────────────────────────────────────────
if [[ "${TESSERA_NO_SPACK_ACTIVATE:-0}" != "1" ]]; then
    case "${TESSERA_SYSTEM}" in
        tuolumne)
            TESSERA_SPACK_ENV="${TESSERA_SPACK_ENV:-${HOME}/spack_envs/tuolumne_trilinos/}"
            export TESSERA_SPACK_ENV
            spack env activate "${TESSERA_SPACK_ENV}"
            ;;
        local)
            # No spack activation for local workstation builds.
            ;;
    esac
fi

# ── 6. Runtime environment ────────────────────────────────────────────────────
_TESSERA_RUNTIME_ENV="${_TESSERA_SYS_DIR}/runtime_env.sh"
if [[ -f "${_TESSERA_RUNTIME_ENV}" && "${TESSERA_DRY_RUN:-0}" != "1" ]]; then
    # shellcheck source=/dev/null
    source "${_TESSERA_RUNTIME_ENV}"
fi

# ── 7. tessera_exe helper ─────────────────────────────────────────────────────
# Usage: path=$(tessera_exe <relpath-or-name>)
#   In manual mode: resolves $TESSERA_BUILD_DIR/<relpath>.
#   In installed mode (future): resolves via PATH.
tessera_exe()
{
    local name="$1"
    case "${TESSERA_BIN_MODE}" in
        manual)
            echo "${TESSERA_BUILD_DIR}/${name}"
            ;;
        *)
            command -v "${name##*/}"
            ;;
    esac
}
export -f tessera_exe
