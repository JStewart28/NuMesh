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

#ifndef TESSERA_MIGRATE_HPP
#define TESSERA_MIGRATE_HPP

#include "Tessera_RegisteredBufferPool.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <unordered_map>
#include <vector>

namespace Tessera
{

// ============================================================================
// MigrateBuffers
// ============================================================================
//
// Persistent, grow-only staging buffers for migrate(). Hold ONE instance and
// pass it into every migrate() call so the CXI NIC registration footprint stays
// bounded across a run (see RegisteredBufferPool). The data pools are byte-typed
// because the AoSoA tuple type is only known inside the migrate() template; per
// call they are reinterpreted to an unmanaged View of the tuple type. The int
// pool holds the packed send-index list.
template <class MemorySpace>
struct MigrateBuffers
{
    detail::RegisteredBufferPool<char, MemorySpace> send_pool;
    detail::RegisteredBufferPool<char, MemorySpace> recv_pool;
    detail::RegisteredBufferPool<int, MemorySpace> send_idx_pool;
};

// ============================================================================
// migrate
// ============================================================================
//
// Coalesced, registration-bounded whole-tuple migration of an AoSoA. Ported and
// generalized from Canopy's TreePartitioner::migrate_particles: instead of
// deriving a destination from a key -> owner map, migrate() takes an explicit
// per-element destination-rank array `dest` (dest(i) == the rank that should own
// element i after migration; dest(i) == this rank keeps it local). This is the
// substrate the mesh migration / load-balancing step (Step 7) drives after it
// computes face destinations.
//
// Mechanics (all preserved from the reference):
//   - Outgoing tuples are packed on the execution space into per-peer subviews
//     of ONE persistent registered send region; one MPI_Isend per peer; matching
//     MPI_Irecv land in ONE persistent registered recv region. Peak concurrent
//     registrations are O(1) per direction regardless of peer count.
//   - The MPI element is one whole tuple (MPI_Type_contiguous over
//     sizeof(tuple) bytes, count == tuple count), so a single peer's payload can
//     exceed 2 GiB without overflowing MPI's signed int count.
//   - Self-peer traffic is never posted to MPI: elements with dest(i) == rank
//     stay in place. At one rank (or when nothing moves) migrate() takes a fast
//     path and does no MPI at all — this is the guard that avoids the MI300A
//     GPU-aware self-send fault.
//
// The AoSoA is rebuilt as (kept elements) ++ (received elements); within-AoSoA
// order afterward is unspecified (callers rebuild keys / re-sort as needed).
// Returns the number of elements sent to other ranks.
//
// `dest` may live in any memory space (it is mirrored to host for the counting
// pass); its extent must equal aosoa.size().
template <class AoSoAType, class DestView>
int migrate( MPI_Comm comm, AoSoAType& aosoa, const DestView& dest,
             MigrateBuffers<typename AoSoAType::memory_space>& bufs )
{
    using memory_space = typename AoSoAType::memory_space;
    using execution_space = typename AoSoAType::execution_space;
    using tuple_type = typename AoSoAType::tuple_type;
    using umtuple_view = Kokkos::View<tuple_type*, memory_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    int rank = 0, comm_size = 1;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &comm_size );

    const int n = static_cast<int>( aosoa.size() );

    // Destination rank per element, on host, for the counting/index passes.
    auto h_dest =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), dest );

    std::vector<int> send_counts( comm_size, 0 );
    int num_sent = 0;
    for ( int i = 0; i < n; ++i )
    {
        const int d = h_dest( i );
        if ( d != rank )
        {
            ++send_counts[d];
            ++num_sent;
        }
    }

    // Discover incoming counts (one Alltoall of comm_size ints).
    std::vector<int> recv_counts( comm_size, 0 );
    MPI_Alltoall( send_counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                  MPI_INT, comm );

    // Ordered peer lists (ascending rank) with tuple offsets into the pools.
    std::vector<int> send_peers, send_peer_off, send_peer_n;
    std::vector<int> recv_peers, recv_peer_off, recv_peer_n;
    std::size_t total_send = 0, total_recv = 0;
    for ( int r = 0; r < comm_size; ++r )
    {
        if ( r != rank && send_counts[r] > 0 )
        {
            send_peers.push_back( r );
            send_peer_off.push_back( static_cast<int>( total_send ) );
            send_peer_n.push_back( send_counts[r] );
            total_send += static_cast<std::size_t>( send_counts[r] );
        }
        if ( r != rank && recv_counts[r] > 0 )
        {
            recv_peers.push_back( r );
            recv_peer_off.push_back( static_cast<int>( total_recv ) );
            recv_peer_n.push_back( recv_counts[r] );
            total_recv += static_cast<std::size_t>( recv_counts[r] );
        }
    }

    // Fast path: nothing leaves and nothing arrives (includes the single-rank
    // case). The AoSoA is unchanged; no MPI on (possibly device) buffers.
    if ( total_send == 0 && total_recv == 0 )
        return num_sent;

    // ---- Packed send-index list (host), grouped by peer in ascending order --
    bufs.send_idx_pool.reserve( total_send );
    auto send_idx = bufs.send_idx_pool.subview( 0, total_send );
    if ( total_send > 0 )
    {
        std::unordered_map<int, int> peer_slot; // rank -> index in send_peers
        for ( int q = 0; q < static_cast<int>( send_peers.size() ); ++q )
            peer_slot[send_peers[q]] = q;
        std::vector<int> cursor = send_peer_off; // running write pos per peer
        auto h_send_idx = Kokkos::create_mirror_view( send_idx );
        for ( int i = 0; i < n; ++i )
        {
            const int d = h_dest( i );
            if ( d != rank )
                h_send_idx( cursor[peer_slot[d]]++ ) = i;
        }
        Kokkos::deep_copy( send_idx, h_send_idx );
    }

    // ---- Size the registered tuple regions up front for stable addresses. ---
    bufs.send_pool.reserve( total_send * sizeof( tuple_type ) );
    bufs.recv_pool.reserve( total_recv * sizeof( tuple_type ) );
    umtuple_view send_buf(
        reinterpret_cast<tuple_type*>( bufs.send_pool.data() ), total_send );
    umtuple_view recv_buf(
        reinterpret_cast<tuple_type*>( bufs.recv_pool.data() ), total_recv );

    // ---- Pack outgoing tuples on device into the one send region. ----
    if ( total_send > 0 )
    {
        auto src = aosoa;
        auto idx = send_idx;
        auto out = send_buf;
        Kokkos::parallel_for(
            "tessera_migrate_pack",
            Kokkos::RangePolicy<execution_space>(
                0, static_cast<int>( total_send ) ),
            KOKKOS_LAMBDA( const int i ) {
                out( i ) = src.getTuple( idx( i ) );
            } );
        Kokkos::fence();
    }

    // ---- Exchange: one MPI element = one whole tuple. ----
    MPI_Datatype tuple_dtype;
    MPI_Type_contiguous( static_cast<int>( sizeof( tuple_type ) ), MPI_BYTE,
                         &tuple_dtype );
    MPI_Type_commit( &tuple_dtype );

    std::vector<MPI_Request> recv_reqs;
    recv_reqs.reserve( recv_peers.size() );
    for ( std::size_t q = 0; q < recv_peers.size(); ++q )
    {
        MPI_Request req;
        MPI_Irecv( recv_buf.data() + recv_peer_off[q], recv_peer_n[q],
                   tuple_dtype, recv_peers[q], /*tag=*/0, comm, &req );
        recv_reqs.push_back( req );
    }
    std::vector<MPI_Request> send_reqs;
    send_reqs.reserve( send_peers.size() );
    for ( std::size_t q = 0; q < send_peers.size(); ++q )
    {
        MPI_Request req;
        MPI_Isend( send_buf.data() + send_peer_off[q], send_peer_n[q],
                   tuple_dtype, send_peers[q], /*tag=*/0, comm, &req );
        send_reqs.push_back( req );
    }
    if ( !recv_reqs.empty() )
        MPI_Waitall( static_cast<int>( recv_reqs.size() ), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( static_cast<int>( send_reqs.size() ), send_reqs.data(),
                     MPI_STATUSES_IGNORE );
    MPI_Type_free( &tuple_dtype );

    // ---- Rebuild the AoSoA as (kept elements) ++ (received elements). ----
    int num_kept = 0;
    for ( int i = 0; i < n; ++i )
        if ( h_dest( i ) == rank )
            ++num_kept;

    Kokkos::View<int*, memory_space> keep_idx(
        Kokkos::view_alloc( Kokkos::WithoutInitializing,
                            "tessera_migrate_keep" ),
        static_cast<std::size_t>( num_kept ) );
    {
        auto h_keep = Kokkos::create_mirror_view( keep_idx );
        int k = 0;
        for ( int i = 0; i < n; ++i )
            if ( h_dest( i ) == rank )
                h_keep( k++ ) = i;
        Kokkos::deep_copy( keep_idx, h_keep );
    }

    const std::size_t new_size =
        static_cast<std::size_t>( num_kept ) + total_recv;
    AoSoAType migrated( "tessera_migrated", new_size );

    if ( num_kept > 0 )
    {
        auto in = aosoa;
        auto out = migrated;
        auto idx = keep_idx;
        Kokkos::parallel_for(
            "tessera_migrate_keep",
            Kokkos::RangePolicy<execution_space>( 0, num_kept ),
            KOKKOS_LAMBDA( const int i ) {
                out.setTuple( i, in.getTuple( idx( i ) ) );
            } );
    }
    if ( total_recv > 0 )
    {
        auto out = migrated;
        auto buf = recv_buf;
        const int base = num_kept;
        Kokkos::parallel_for(
            "tessera_migrate_unpack",
            Kokkos::RangePolicy<execution_space>(
                0, static_cast<int>( total_recv ) ),
            KOKKOS_LAMBDA( const int j ) {
                out.setTuple( base + j, buf( j ) );
            } );
    }
    Kokkos::fence();

    aosoa = migrated;
    return num_sent;
}

} // namespace Tessera

#endif // TESSERA_MIGRATE_HPP
