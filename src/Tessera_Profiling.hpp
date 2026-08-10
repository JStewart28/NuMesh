/****************************************************************************
 * Copyright (c) 2024, JStewart28                                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Tessera library. Tessera is distributed under a *
 * BSD 3-Clause license. For the licensing terms see the LICENSE file in   *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef TESSERA_PROFILING_HPP
#define TESSERA_PROFILING_HPP

// ---------------------------------------------------------------------------
// Profiling level hierarchy
//
//   0 — off (no instrumentation)
//   1 — basic: top-level phases (build, partition, distribute, refine,
//       migrate, loadBalance, halo exchange, I/O, quality marking)
//   2 — detailed: major sub-phases within each level-1 phase (refine phases,
//       migrate rounds, distribute steps, writeMesh sections, ...)
//   3 — verbose: fine-grained comm rounds and device kernels
//
// Set via CMake: -DTessera_PROFILING_LEVEL=2 (or the legacy
// -DTessera_ENABLE_PROFILING=ON, which defaults to level 1).
//
// Cadence policy note: Tessera is a library and owns only the *mechanism*. It
// has no timestep loop, so *when* to reset/print is the downstream caller's
// decision. Every scoped region accumulates into two registries: a resettable
// "window" registry (TESSERA_RESET_TIMERS() clears it) and a monotonic
// "lifetime" registry (never auto-cleared). A downstream simulation prints the
// window aggregate every `ts` steps then resets, and prints the lifetime
// aggregate once at shutdown.
// ---------------------------------------------------------------------------
#ifndef TESSERA_PROFILING_LEVEL
#ifdef TESSERA_ENABLE_PROFILING
#define TESSERA_PROFILING_LEVEL 1
#else
#define TESSERA_PROFILING_LEVEL 0
#endif
#endif

#ifdef TESSERA_ENABLE_PROFILING

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace Tessera
{
namespace Profiling
{

// ---------------------------------------------------------------------------
// Phase key constants — used as region names / registry keys. Defined here so
// all instrumented headers share the same strings without risk of typos.
// ---------------------------------------------------------------------------

// Level 1 — top-level phases.
static constexpr const char* TIMER_BUILD_ICOSPHERE = "build_icosphere";
static constexpr const char* TIMER_BUILD_LATLON = "build_latlon_sphere";
//! The distributed initial-construction path (Tessera_DistributedBuilder.hpp):
//! no rank ever materializes the global mesh, so unlike build_icosphere these
//! two are O(global/ranks) rather than O(global) per rank.
static constexpr const char* TIMER_BUILD_SOUP_DIST = "build_soup_distributed";
static constexpr const char* TIMER_BUILD_ICOSPHERE_DIST =
    "build_icosphere_distributed";
static constexpr const char* TIMER_PARTITION = "partition";
static constexpr const char* TIMER_DISTRIBUTE = "distribute";
static constexpr const char* TIMER_HALO_EXCHANGE = "halo_exchange";
static constexpr const char* TIMER_HALO_SCATTER_ADD = "halo_scatter_add";
static constexpr const char* TIMER_REFINE = "refine";
static constexpr const char* TIMER_MIGRATE = "migrate";
static constexpr const char* TIMER_LOAD_BALANCE = "load_balance";
static constexpr const char* TIMER_WRITE_MESH = "write_mesh";
static constexpr const char* TIMER_READ_MESH = "read_mesh";
static constexpr const char* TIMER_MARK_QUALITY = "mark_quality";
static constexpr const char* TIMER_MARK_EDGE_LEN = "mark_edge_length";
static constexpr const char* TIMER_MARK_CURVATURE = "mark_curvature";

// Level 2 — refine() sub-phases.
static constexpr const char* TIMER_REFINE_BALANCE = "refine_2to1_balance";
static constexpr const char* TIMER_REFINE_MIDPOINT = "refine_midpoint_gids";
static constexpr const char* TIMER_REFINE_REBUILD = "refine_local_rebuild";
// RefinementMode::Conforming only: the transient closure layer is discarded
// (step 0) and rebuilt (step 3b) on every refine call. Both are purely local.
static constexpr const char* TIMER_REFINE_UNCLOSE = "refine_unclose";
static constexpr const char* TIMER_REFINE_CLOSE = "refine_close";

// Level 2 — migrate()'s own rounds (the move half).
// RefinementMode::Conforming only: the local pre-move pass that makes every
// closure sibling follow the lowest-gid sibling's destination.
static constexpr const char* TIMER_MIGRATE_SIBLING = "migrate_sibling_cohesion";
static constexpr const char* TIMER_MIGRATE_MOVE = "migrate_round_a_move";

// Level 2 — rebuildHalo() and its rounds (the halo half). Shared by migrate()
// and refine(), so the keys are not migrate-specific: rounds G and B/C/D are
// charged here whichever caller drove them. TIMER_HALO_REBUILD covers only the
// standalone rebuildHalo() entry point, so migrate() does not double-count.
static constexpr const char* TIMER_HALO_REBUILD = "halo_rebuild";
static constexpr const char* TIMER_HALO_GATHER = "halo_round_g_gather";
static constexpr const char* TIMER_HALO_OWNERSHIP = "halo_round_b_ownership";
static constexpr const char* TIMER_HALO_GHOSTFETCH = "halo_round_c_ghostfetch";
static constexpr const char* TIMER_HALO_ASSEMBLE = "halo_round_d_assemble";

// Level 2 — distributed-builder sub-phases (Tessera_DistributedBuilder.hpp).
// The vertex-key round is charged per call, so in buildIcosphereDistributed()
// it accumulates the `subdivisions` per-level rounds plus the final dedup.
static constexpr const char* TIMER_DBUILD_VKEYS = "dbuild_vertex_key_gids";
static constexpr const char* TIMER_DBUILD_FACES = "dbuild_face_gids";
static constexpr const char* TIMER_DBUILD_EDGES = "dbuild_edge_gids";
static constexpr const char* TIMER_DBUILD_ASSEMBLE = "dbuild_assemble";
static constexpr const char* TIMER_DBUILD_GENERATE = "dbuild_generate_subtrees";

// Level 2 — distribute() sub-phases.
static constexpr const char* TIMER_DISTRIBUTE_CSR = "distribute_csr_rebuild";
static constexpr const char* TIMER_DISTRIBUTE_HALOPLAN = "distribute_halo_plan";

// Level 2 — writeMesh() / readMesh() sub-phases.
static constexpr const char* TIMER_WRITE_DENSE_NUMBER = "write_dense_numbering";
static constexpr const char* TIMER_WRITE_GHOST_FETCH = "write_ghost_fetch";
static constexpr const char* TIMER_WRITE_FILE_CREATE = "write_file_create";
static constexpr const char* TIMER_WRITE_DATASETS = "write_datasets";
static constexpr const char* TIMER_READ_BLOCKS = "read_blocks";
static constexpr const char* TIMER_READ_RECONSTRUCT = "read_reconstruct";

// Level 2 — loadBalance() sub-phases.
static constexpr const char* TIMER_LB_GATHER = "lb_gather_to_root";
static constexpr const char* TIMER_LB_SOLVE = "lb_zoltan2_solve";

// Level 3 — fine-grained comm rounds / kernels.
static constexpr const char* TIMER_REFINE_ADVERTISE =
    "refine_advertise_alltoallv";
static constexpr const char* TIMER_REFINE_MARKREQ = "refine_markreq_alltoallv";
//! The green/blue/red pattern application itself, inside TIMER_REFINE_CLOSE.
static constexpr const char* TIMER_REFINE_CLOSURE_PATTERNS =
    "refine_closure_patterns";
static constexpr const char* TIMER_MARK_EDGE_KERNEL = "mark_edge_length_kernel";
static constexpr const char* TIMER_MARK_CURV_KERNEL =
    "mark_curvature_normals_kernel";
static constexpr const char* TIMER_WRITE_HYPERSLAB = "write_hyperslabs";
static constexpr const char* TIMER_READ_HYPERSLAB = "read_hyperslabs";

// ---------------------------------------------------------------------------
// Timer registries — process-local accumulator maps, key -> elapsed seconds.
// Using function-local statics so this is safe in a header-only library:
// exactly one instance per process, initialized on first use.
//
//   window   — cleared by reset_timers(); holds one reporting window.
//   lifetime — never cleared by reset_timers(); holds the whole run.
//
// accumulate() adds elapsed time to BOTH, so a downstream caller can report a
// per-window breakdown (print then reset each interval) and still emit a
// whole-run total at shutdown.
// ---------------------------------------------------------------------------
inline std::unordered_map<std::string, double>& timer_registry()
{
    static std::unordered_map<std::string, double> s_reg;
    return s_reg;
}

inline std::unordered_map<std::string, double>& lifetime_registry()
{
    static std::unordered_map<std::string, double> s_reg;
    return s_reg;
}

inline void reset_timers() { timer_registry().clear(); }

inline void accumulate( const char* key, double elapsed )
{
    timer_registry()[key] += elapsed;
    lifetime_registry()[key] += elapsed;
}

// ---------------------------------------------------------------------------
// ScopedTimer — RAII guard. Records wall time at construction via MPI_Wtime,
// pushes a Kokkos::Profiling region (so external Kokkos-aware tools also see
// the named region), and accumulates elapsed time at destruction. Non-copyable.
//
// Usage:
//   { ScopedTimer t( TIMER_REFINE ); /* work */ }  // accumulates on scope exit
//
// Wall time is charged to the enclosing scope; issue a Kokkos::fence() before
// scope exit where device work must be included in the measured region.
// ---------------------------------------------------------------------------
struct ScopedTimer
{
    const char* key;
    double t0;

    explicit ScopedTimer( const char* phase_key )
        : key( phase_key )
        , t0( MPI_Wtime() )
    {
        Kokkos::Profiling::pushRegion( key );
    }

    ~ScopedTimer()
    {
        Kokkos::Profiling::popRegion();
        accumulate( key, MPI_Wtime() - t0 );
    }

    ScopedTimer( const ScopedTimer& ) = delete;
    ScopedTimer& operator=( const ScopedTimer& ) = delete;
};

// ---------------------------------------------------------------------------
// print_timing_table
//
// Gathers per-rank timing data for a registry via three MPI_Reduce calls
// (MIN, MAX, SUM) to rank 0. Only rank 0 prints the formatted table; the keys
// are sorted alphabetically for a stable ordering. Collective: every rank must
// call this (all ranks participate in the reduces, then non-root ranks return).
//
// Parameters:
//   comm         - MPI communicator
//   section_name - printed in the header line (e.g. "window" / "whole run")
//   reg          - the registry to report (window or lifetime)
// ---------------------------------------------------------------------------
inline void
print_timing_table( MPI_Comm comm, const char* section_name,
                    const std::unordered_map<std::string, double>& reg )
{
    int rank, nprocs;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &nprocs );

    // Rank 0 owns the canonical key ordering and broadcasts it so every rank
    // reduces the same slots in the same order (registries may differ per rank
    // if a phase never ran locally).
    std::vector<std::string> keys;
    if ( rank == 0 )
    {
        keys.reserve( reg.size() );
        for ( const auto& kv : reg )
            keys.push_back( kv.first );
        std::sort( keys.begin(), keys.end() );
    }

    int nkeys = static_cast<int>( keys.size() );
    MPI_Bcast( &nkeys, 1, MPI_INT, 0, comm );

    // Broadcast the key strings (packed with '\n' separators).
    std::string packed;
    if ( rank == 0 )
        for ( const auto& k : keys )
        {
            packed += k;
            packed += '\n';
        }
    int packed_len = static_cast<int>( packed.size() );
    MPI_Bcast( &packed_len, 1, MPI_INT, 0, comm );
    packed.resize( packed_len );
    MPI_Bcast( packed.data(), packed_len, MPI_CHAR, 0, comm );
    if ( rank != 0 )
    {
        keys.clear();
        std::string cur;
        for ( int i = 0; i < packed_len; i++ )
        {
            if ( packed[i] == '\n' )
            {
                keys.push_back( cur );
                cur.clear();
            }
            else
                cur += packed[i];
        }
    }

    std::vector<double> local_vals( nkeys, 0.0 );
    for ( int i = 0; i < nkeys; i++ )
    {
        auto it = reg.find( keys[i] );
        if ( it != reg.end() )
            local_vals[i] = it->second;
    }

    std::vector<double> min_vals( nkeys ), max_vals( nkeys ), sum_vals( nkeys );
    MPI_Reduce( local_vals.data(), min_vals.data(), nkeys, MPI_DOUBLE, MPI_MIN,
                0, comm );
    MPI_Reduce( local_vals.data(), max_vals.data(), nkeys, MPI_DOUBLE, MPI_MAX,
                0, comm );
    MPI_Reduce( local_vals.data(), sum_vals.data(), nkeys, MPI_DOUBLE, MPI_SUM,
                0, comm );

    if ( rank != 0 )
        return;

    static constexpr int COL_LABEL = 36;
    static constexpr int COL_NUM = 9;

    std::printf( "\n[Tessera Diagnostics] %s timing (%d MPI rank%s)\n",
                 section_name, nprocs, nprocs > 1 ? "s" : "" );
    std::printf( "  %-*s  %*s  %*s  %*s  %s\n", COL_LABEL, "Region", COL_NUM,
                 "Min (s)", COL_NUM, "Max (s)", COL_NUM, "Mean (s)",
                 "Imbalance" );

    const int sep_len = COL_LABEL + 3 * ( COL_NUM + 2 ) + 12;
    for ( int i = 0; i < sep_len; i++ )
        std::putchar( '-' );
    std::putchar( '\n' );

    const double inv_nprocs = 1.0 / static_cast<double>( nprocs );
    for ( int i = 0; i < nkeys; i++ )
    {
        const double mn = min_vals[i];
        const double mx = max_vals[i];
        const double mean = sum_vals[i] * inv_nprocs;
        const double imb = ( mean > 0.0 ) ? ( mx - mean ) / mean * 100.0 : 0.0;

        std::printf( "  %-*s  %*.4f  %*.4f  %*.4f  %.1f%%\n", COL_LABEL,
                     keys[i].c_str(), COL_NUM, mn, COL_NUM, mx, COL_NUM, mean,
                     imb );
    }
    std::putchar( '\n' );
    std::fflush( stdout );
}

} // namespace Profiling
} // namespace Tessera

#endif // TESSERA_ENABLE_PROFILING

// ---------------------------------------------------------------------------
// Convenience macros — defined whether or not profiling is enabled so
// instrumentation in other headers compiles in both modes. Level-gated macros
// compile away to no-ops below the requested level so call sites stay in place.
// ---------------------------------------------------------------------------
#ifdef TESSERA_ENABLE_PROFILING
#define TESSERA_SCOPED_TIMER( key )                                            \
    ::Tessera::Profiling::ScopedTimer _tessera_timer_##__LINE__( ( key ) )
#define TESSERA_RESET_TIMERS() ::Tessera::Profiling::reset_timers()
#define TESSERA_WTIME() MPI_Wtime()
#define TESSERA_PRINT_TIMERS( comm )                                           \
    ::Tessera::Profiling::print_timing_table(                                  \
        ( comm ), "window", ::Tessera::Profiling::timer_registry() )
#define TESSERA_PRINT_TIMERS_TOTAL( comm )                                     \
    ::Tessera::Profiling::print_timing_table(                                  \
        ( comm ), "whole run", ::Tessera::Profiling::lifetime_registry() )
#if TESSERA_PROFILING_LEVEL >= 2
#define TESSERA_SCOPED_TIMER_DETAILED( key )                                   \
    ::Tessera::Profiling::ScopedTimer _tessera_timer_d_##__LINE__( ( key ) )
#else
#define TESSERA_SCOPED_TIMER_DETAILED( key )                                   \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#endif
#if TESSERA_PROFILING_LEVEL >= 3
#define TESSERA_SCOPED_TIMER_VERBOSE( key )                                    \
    ::Tessera::Profiling::ScopedTimer _tessera_timer_v_##__LINE__( ( key ) )
#else
#define TESSERA_SCOPED_TIMER_VERBOSE( key )                                    \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#endif
#else
#define TESSERA_SCOPED_TIMER( key )                                            \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#define TESSERA_SCOPED_TIMER_DETAILED( key )                                   \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#define TESSERA_SCOPED_TIMER_VERBOSE( key )                                    \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#define TESSERA_RESET_TIMERS()                                                 \
    do                                                                         \
    {                                                                          \
    } while ( 0 )
#define TESSERA_WTIME() 0.0
#define TESSERA_PRINT_TIMERS( comm )                                           \
    do                                                                         \
    {                                                                          \
        (void)( comm );                                                        \
    } while ( 0 )
#define TESSERA_PRINT_TIMERS_TOTAL( comm )                                     \
    do                                                                         \
    {                                                                          \
        (void)( comm );                                                        \
    } while ( 0 )
#endif

#endif // TESSERA_PROFILING_HPP
