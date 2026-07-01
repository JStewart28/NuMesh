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

#ifndef TESSERA_CSR_ADJACENCY_HPP
#define TESSERA_CSR_ADJACENCY_HPP

#include "Tessera_Types.hpp"

#include <Kokkos_Core.hpp>

namespace Tessera
{

// ============================================================================
// CsrAdjacency
// ============================================================================
//
// Compressed-sparse-row storage for a variable-valence adjacency relation, used
// for the vertex 1-ring (vertex -> incident edges, vertex -> incident faces).
// This is kept OUT of the AoSoA so the AoSoA member slices stay fixed-arity and
// vectorizable; valence varies per vertex (and, at irregular vertices produced
// by AMR, is not 6).
//
// For a source entity with local index i, its neighbors are the local indices
//   neighbors[ offsets(i) .. offsets(i+1) )
// Neighbors are stored as LocalIndex (dense array positions) for direct AoSoA
// indexing in kernels. The relation is (re)built by the mesh builder (Step 3)
// and rebuilt after every topology change (refinement / migration).
//
template <class MemorySpace>
struct CsrAdjacency
{
    using memory_space = MemorySpace;

    //! Row offsets, length (num_sources + 1). offsets(n) ==
    //! neighbors.extent(0).
    Kokkos::View<int*, MemorySpace> offsets;
    //! Flattened neighbor local indices, length offsets(num_sources).
    Kokkos::View<LocalIndex*, MemorySpace> neighbors;

    CsrAdjacency() = default;

    //! Number of source entities (rows). Zero before build.
    KOKKOS_INLINE_FUNCTION
    int numSources() const
    {
        const int n = static_cast<int>( offsets.extent( 0 ) );
        return n > 0 ? n - 1 : 0;
    }

    //! Total number of stored neighbor entries.
    KOKKOS_INLINE_FUNCTION
    int numEntries() const { return static_cast<int>( neighbors.extent( 0 ) ); }

    //! Allocate the CSR structure for `num_sources` rows and `num_entries`
    //! total neighbors. Contents are left uninitialized; the builder fills
    //! them.
    void allocate( int num_sources, int num_entries, const std::string& label )
    {
        offsets = Kokkos::View<int*, MemorySpace>(
            Kokkos::view_alloc( label + "_offsets" ),
            static_cast<std::size_t>( num_sources ) + 1 );
        neighbors = Kokkos::View<LocalIndex*, MemorySpace>(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                label + "_neighbors" ),
            static_cast<std::size_t>( num_entries ) );
    }
};

} // namespace Tessera

#endif // TESSERA_CSR_ADJACENCY_HPP
