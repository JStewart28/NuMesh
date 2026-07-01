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

#ifndef TESSERA_ALL_TO_ALL_V_HPP
#define TESSERA_ALL_TO_ALL_V_HPP

#include <mpi.h>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace Tessera
{

// ============================================================================
// allToAllV — variable-arity neighbor topology exchange (host)
// ============================================================================
//
// The "third" comm primitive (distinct from migrate and the halo field-sync):
// exchange variable-length lists of small POD items where the RECEIVER does not
// yet know how many items it will get or their layout. This is what Step 5 uses
// to BUILD the ghost layer and discover boundary sharing sets — a rank advertises
// to each neighbour the gids/keys of the entities it holds on a shared boundary,
// and each neighbour receives them and matches against its own.
//
// Counts are discovered with one MPI_Alltoall, then the payload is exchanged with
// MPI_Alltoallv. This runs on the host: the data volume is O(boundary entities)
// (small relative to the per-step field sync) and the subsequent matching builds
// host-side maps, so keeping it on the host is both simpler and where the result
// is consumed. `T` must be trivially copyable (e.g. GlobalId, EdgeKey, or a small
// POD struct); it is shipped as raw bytes.
//
// AllToAllVResult groups the received items by SOURCE rank:
//   for source rank r, its items are data[ displs[r] .. displs[r+1] ).
template <class T>
struct AllToAllVResult
{
    std::vector<int> counts; // recv count per source rank (size comm_size)
    std::vector<int> displs; // prefix sum (size comm_size + 1)
    std::vector<T> data;     // received items, grouped by source rank

    //! Number of items received from source rank r.
    int count( int r ) const { return counts[r]; }
    //! Pointer to the first item received from source rank r (or nullptr).
    const T* from( int r ) const
    {
        return counts[r] > 0 ? data.data() + displs[r] : nullptr;
    }
};

//! Exchange variable-length item lists. send[r] holds the items destined for
//! rank r (send.size() must equal comm_size; send[self] is allowed and is simply
//! looped back). Returns the items received from every rank, grouped by source.
template <class T>
AllToAllVResult<T> allToAllV( MPI_Comm comm,
                              const std::vector<std::vector<T>>& send )
{
    static_assert( std::is_trivially_copyable<T>::value,
                   "allToAllV item type must be trivially copyable" );

    int comm_size = 1;
    MPI_Comm_size( comm, &comm_size );

    // Flatten the send lists and record per-destination counts/displacements.
    std::vector<int> send_counts( comm_size, 0 );
    std::vector<int> send_displs( comm_size + 1, 0 );
    for ( int r = 0; r < comm_size; ++r )
        send_counts[r] = static_cast<int>( send[r].size() );
    for ( int r = 0; r < comm_size; ++r )
        send_displs[r + 1] = send_displs[r] + send_counts[r];

    std::vector<T> send_flat;
    send_flat.reserve( send_displs[comm_size] );
    for ( int r = 0; r < comm_size; ++r )
        send_flat.insert( send_flat.end(), send[r].begin(), send[r].end() );

    // Discover incoming counts, then compute recv displacements.
    AllToAllVResult<T> result;
    result.counts.assign( comm_size, 0 );
    MPI_Alltoall( send_counts.data(), 1, MPI_INT, result.counts.data(), 1,
                  MPI_INT, comm );
    result.displs.assign( comm_size + 1, 0 );
    for ( int r = 0; r < comm_size; ++r )
        result.displs[r + 1] = result.displs[r] + result.counts[r];
    result.data.resize( result.displs[comm_size] );

    // Exchange the payload as raw bytes (one MPI element = one item).
    MPI_Datatype item_dtype;
    MPI_Type_contiguous( static_cast<int>( sizeof( T ) ), MPI_BYTE,
                         &item_dtype );
    MPI_Type_commit( &item_dtype );
    MPI_Alltoallv( send_flat.data(), send_counts.data(), send_displs.data(),
                   item_dtype, result.data.data(), result.counts.data(),
                   result.displs.data(), item_dtype, comm );
    MPI_Type_free( &item_dtype );

    return result;
}

} // namespace Tessera

#endif // TESSERA_ALL_TO_ALL_V_HPP
