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

#ifndef TESSERA_TYPES_HPP
#define TESSERA_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <cstdint>

namespace Tessera
{

// ============================================================================
// Fundamental identifier / index types
// ============================================================================
//
// GlobalId    Globally-unique 64-bit identifier of an entity
// (vertex/edge/face).
//             64 bits comfortably addresses the 100M+ entities per rank the
//             design targets across all ranks.
// LocalIndex  Dense per-rank array position of an entity in its AoSoA. `int`
//             matches Kokkos::RangePolicy's index type used in device kernels.
// Level       Adaptive-refinement level (0 = coarse). Small signed integer.
// Rank        MPI rank that owns / holds an entity.
//
using GlobalId = std::uint64_t;
using LocalIndex = int;
using Level = std::int16_t;
using Rank = std::int32_t;

//! Sentinel for an unset / invalid global id.
inline constexpr GlobalId invalid_gid = ~static_cast<GlobalId>( 0 );

//! Sentinel for an unset / invalid local index.
inline constexpr LocalIndex invalid_local = -1;

// ============================================================================
// Key<N> — structured canonical entity key
// ============================================================================
//
// The cross-rank *identity* of a composite entity is a structured, order-
// invariant tuple of the GlobalIds it is built from — NOT a hash. Sorting the
// constituent ids on construction makes the key independent of the order they
// are supplied, so every rank that touches a shared entity computes the exact
// same key with no communication and with zero collision risk (a hash of two
// 64-bit ids into 64 bits has a ~27% collision probability at 1e8 entities and
// would silently merge distinct entities across a partition boundary).
//
//   EdgeKey = Key<2>  : the two endpoint vertex gids   (min, max)
//   FaceKey = Key<3>  : the three corner vertex gids   (sorted ascending)
//
// The key is a trivially-copyable POD and every operation is device-callable so
// keys can be built and compared inside Kokkos kernels and shipped over MPI as
// raw bytes.
//
template <int N>
struct Key
{
    static_assert( N >= 1, "Key must have at least one component" );

    GlobalId id[N];

    KOKKOS_DEFAULTED_FUNCTION Key() = default;

    KOKKOS_INLINE_FUNCTION
    bool operator==( const Key& o ) const
    {
        for ( int i = 0; i < N; ++i )
            if ( id[i] != o.id[i] )
                return false;
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    bool operator!=( const Key& o ) const { return !( *this == o ); }

    //! Lexicographic ordering — for sorting and ordered lookup during
    //! halo/ghost matching and deterministic global renumbering.
    KOKKOS_INLINE_FUNCTION
    bool operator<( const Key& o ) const
    {
        for ( int i = 0; i < N; ++i )
        {
            if ( id[i] < o.id[i] )
                return true;
            if ( id[i] > o.id[i] )
                return false;
        }
        return false;
    }
};

using EdgeKey = Key<2>;
using FaceKey = Key<3>;

//! Ascending insertion sort of a key's components (N is small: 2 or 3).
//! Branch-light and device-callable; makes the key order-invariant.
template <int N>
KOKKOS_INLINE_FUNCTION void sortKey( Key<N>& k )
{
    for ( int i = 1; i < N; ++i )
    {
        const GlobalId x = k.id[i];
        int j = i - 1;
        while ( j >= 0 && k.id[j] > x )
        {
            k.id[j + 1] = k.id[j];
            --j;
        }
        k.id[j + 1] = x;
    }
}

//! Canonical edge key from its two endpoint vertex gids (order-invariant).
KOKKOS_INLINE_FUNCTION
EdgeKey makeEdgeKey( GlobalId a, GlobalId b )
{
    EdgeKey k;
    k.id[0] = a;
    k.id[1] = b;
    sortKey( k );
    return k;
}

//! Canonical face key from its three corner vertex gids (order-invariant).
KOKKOS_INLINE_FUNCTION
FaceKey makeFaceKey( GlobalId a, GlobalId b, GlobalId c )
{
    FaceKey k;
    k.id[0] = a;
    k.id[1] = b;
    k.id[2] = c;
    sortKey( k );
    return k;
}

// ----------------------------------------------------------------------------
// Refinement identity note (assignment lives in Step 6, not here)
// ----------------------------------------------------------------------------
//
// A refinement midpoint vertex is *identified* across ranks by the EdgeKey of
// the edge it bisects — identical on both sides of a partition boundary with no
// communication. Its persistent 64-bit GlobalId is assigned later (Step 6): a
// per-rank exclusive scan for interior midpoints, with boundary-shared
// midpoints taking the owner's (lowest-rank's) id during the refinement
// re-halo. Because every vertex therefore always carries a 64-bit gid,
// edge/face keys are always built from 64-bit gids and stay bounded at
// Key<2>/Key<3> regardless of refinement depth. This header provides only the
// key *type and construction*; the id-assignment scheme is implemented with the
// refinement step.

} // namespace Tessera

#endif // TESSERA_TYPES_HPP
