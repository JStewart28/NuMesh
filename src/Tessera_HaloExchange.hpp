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

#ifndef TESSERA_HALO_EXCHANGE_HPP
#define TESSERA_HALO_EXCHANGE_HPP

#include "Tessera_RegisteredBufferPool.hpp"
#include "Tessera_Types.hpp"

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <map>
#include <string>
#include <vector>

namespace Tessera
{

// ============================================================================
// HaloExchangePlan
// ============================================================================
//
// A first-class description of a 1-deep halo exchange: which owned entities this
// rank SENDS to each neighbour (because that neighbour ghosts them), and which
// local ghost slots RECEIVE from each neighbour. It owns the persistent
// registered staging pools so repeated syncs over a stable topology reuse one
// registered region per direction.
//
// Layout convention (matches the mesh): an AoSoA holds owned entities first, then
// ghost entities; `send_idx` are owned local indices, `recv_idx` are ghost local
// indices. Both index lists are flat and grouped by peer in the peer-list order;
// `send_off`/`recv_off` are the per-peer prefix offsets (size peers+1).
//
// ALIGNMENT CONTRACT: for a peer pair (A,B), the order in which A packs the
// entities it sends to B must equal the order in which B lays out the ghost slots
// it receives from A. The plan is a passive container — the BUILDER (Step 5)
// guarantees this by ordering shared entities by their canonical gid/key on both
// sides. Self-peer entries must not appear in the plan (the builder drops them);
// a single rank therefore has an empty plan and haloExchange() is a no-op.
//
// INVALIDATION: any change to the local entity count or the ghost set (refinement,
// migration) invalidates the plan. Call clear() and rebuild before the next sync.
template <class MemorySpace>
struct HaloExchangePlan
{
    using memory_space = MemorySpace;

    std::vector<int> send_peers; // ranks we send owned data to
    std::vector<int> send_off;   // prefix offsets into send_idx (peers+1)
    std::vector<int> recv_peers; // ranks we receive ghost data from
    std::vector<int> recv_off;   // prefix offsets into recv_idx (peers+1)

    Kokkos::View<LocalIndex*, MemorySpace> send_idx; // owned local indices
    Kokkos::View<LocalIndex*, MemorySpace> recv_idx; // ghost local indices

    detail::RegisteredBufferPool<char, MemorySpace> send_pool;
    detail::RegisteredBufferPool<char, MemorySpace> recv_pool;

    std::size_t totalSend() const { return send_idx.extent( 0 ); }
    std::size_t totalRecv() const { return recv_idx.extent( 0 ); }

    //! Reset to an empty (no-op) plan. Pools are retained (grow-only) so their
    //! registered regions are reused after a rebuild.
    void clear()
    {
        send_peers.clear();
        send_off.clear();
        recv_peers.clear();
        recv_off.clear();
        send_idx = Kokkos::View<LocalIndex*, MemorySpace>();
        recv_idx = Kokkos::View<LocalIndex*, MemorySpace>();
    }

    //! Build the plan from per-peer host index lists. Self-peer entries (key ==
    //! self_rank) are dropped. `send_by_peer[p]` = owned local indices sent to p;
    //! `recv_by_peer[p]` = ghost local indices filled from p. Maps are ordered,
    //! so peers are visited in ascending rank on both sides.
    void
    setFromHost( int self_rank,
                 const std::map<int, std::vector<LocalIndex>>& send_by_peer,
                 const std::map<int, std::vector<LocalIndex>>& recv_by_peer )
    {
        clear();
        auto pack = [&]( const std::map<int, std::vector<LocalIndex>>& by_peer,
                         std::vector<int>& peers, std::vector<int>& off,
                         Kokkos::View<LocalIndex*, MemorySpace>& idx,
                         const char* label )
        {
            off.push_back( 0 );
            std::vector<LocalIndex> flat;
            for ( const auto& kv : by_peer )
            {
                if ( kv.first == self_rank || kv.second.empty() )
                    continue;
                peers.push_back( kv.first );
                flat.insert( flat.end(), kv.second.begin(), kv.second.end() );
                off.push_back( static_cast<int>( flat.size() ) );
            }
            idx = Kokkos::View<LocalIndex*, MemorySpace>(
                Kokkos::view_alloc( std::string( label ),
                                    Kokkos::WithoutInitializing ),
                flat.size() );
            auto h = Kokkos::create_mirror_view( idx );
            for ( std::size_t i = 0; i < flat.size(); ++i )
                h( i ) = flat[i];
            Kokkos::deep_copy( idx, h );
        };
        pack( send_by_peer, send_peers, send_off, send_idx, "halo_send_idx" );
        pack( recv_by_peer, recv_peers, recv_off, recv_idx, "halo_recv_idx" );
    }
};

// ============================================================================
// haloExchange — whole-tuple field sync over a stable plan
// ============================================================================
//
// Packs every owned entity named in the plan's send_idx and ships it, in place,
// into the ghost slots named in recv_idx. The MPI element is one whole AoSoA
// tuple (all core + user fields), so a single call syncs the entire field pack of
// every ghost from its owner — this is mesh.haloExchange(). Same GPU-resident,
// registration-bounded machinery as migrate(): device pack/unpack, one registered
// region per direction, whole-tuple MPI_Type_contiguous, self-peer never posted.
// The AoSoA is NOT resized (ghost slots already exist); only ghost tuples change.
template <class AoSoAType, class MemorySpace>
void haloExchange( MPI_Comm comm, AoSoAType& aosoa,
                   HaloExchangePlan<MemorySpace>& plan )
{
    using execution_space = typename AoSoAType::execution_space;
    using tuple_type = typename AoSoAType::tuple_type;
    using umtuple_view = Kokkos::View<tuple_type*, MemorySpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    const std::size_t total_send = plan.totalSend();
    const std::size_t total_recv = plan.totalRecv();
    if ( total_send == 0 && total_recv == 0 )
        return; // single rank / no ghosts: no MPI on device buffers

    plan.send_pool.reserve( total_send * sizeof( tuple_type ) );
    plan.recv_pool.reserve( total_recv * sizeof( tuple_type ) );
    umtuple_view send_buf(
        reinterpret_cast<tuple_type*>( plan.send_pool.data() ), total_send );
    umtuple_view recv_buf(
        reinterpret_cast<tuple_type*>( plan.recv_pool.data() ), total_recv );

    // Pack owned tuples (at send_idx) into the one send region.
    if ( total_send > 0 )
    {
        auto src = aosoa;
        auto idx = plan.send_idx;
        auto out = send_buf;
        Kokkos::parallel_for(
            "tessera_halo_pack",
            Kokkos::RangePolicy<execution_space>(
                0, static_cast<int>( total_send ) ),
            KOKKOS_LAMBDA( const int i ) {
                out( i ) = src.getTuple( idx( i ) );
            } );
        Kokkos::fence();
    }

    // Exchange: one MPI element = one whole tuple.
    MPI_Datatype tuple_dtype;
    MPI_Type_contiguous( static_cast<int>( sizeof( tuple_type ) ), MPI_BYTE,
                         &tuple_dtype );
    MPI_Type_commit( &tuple_dtype );

    std::vector<MPI_Request> recv_reqs;
    for ( std::size_t q = 0; q < plan.recv_peers.size(); ++q )
    {
        const int n = plan.recv_off[q + 1] - plan.recv_off[q];
        MPI_Request req;
        MPI_Irecv( recv_buf.data() + plan.recv_off[q], n, tuple_dtype,
                   plan.recv_peers[q], /*tag=*/0, comm, &req );
        recv_reqs.push_back( req );
    }
    std::vector<MPI_Request> send_reqs;
    for ( std::size_t q = 0; q < plan.send_peers.size(); ++q )
    {
        const int n = plan.send_off[q + 1] - plan.send_off[q];
        MPI_Request req;
        MPI_Isend( send_buf.data() + plan.send_off[q], n, tuple_dtype,
                   plan.send_peers[q], /*tag=*/0, comm, &req );
        send_reqs.push_back( req );
    }
    if ( !recv_reqs.empty() )
        MPI_Waitall( static_cast<int>( recv_reqs.size() ), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( static_cast<int>( send_reqs.size() ), send_reqs.data(),
                     MPI_STATUSES_IGNORE );
    MPI_Type_free( &tuple_dtype );

    // Unpack received tuples into their ghost slots (at recv_idx).
    if ( total_recv > 0 )
    {
        auto dst = aosoa;
        auto idx = plan.recv_idx;
        auto in = recv_buf;
        Kokkos::parallel_for(
            "tessera_halo_unpack",
            Kokkos::RangePolicy<execution_space>(
                0, static_cast<int>( total_recv ) ),
            KOKKOS_LAMBDA( const int i ) {
                dst.setTuple( idx( i ), in( i ) );
            } );
        Kokkos::fence();
    }
}

} // namespace Tessera

#endif // TESSERA_HALO_EXCHANGE_HPP
