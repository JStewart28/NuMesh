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

#ifndef TESSERA_HALO_SCATTER_ADD_HPP
#define TESSERA_HALO_SCATTER_ADD_HPP

#include "Tessera_Distribute.hpp" // MeshHalo
#include "Tessera_HaloExchange.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_Slice.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace Tessera
{

// ============================================================================
// haloScatterAdd — the REVERSE halo: ghost -> owner, accumulating
// ============================================================================
//
// haloExchange() is a pure GATHER: every ghost slot named in plan.recv_idx is
// overwritten with its owner's value. This is the missing other half of the
// standard distributed-assembly pattern. Whenever a per-vertex quantity is
// assembled by iterating FACES -- vertex areas, a face-to-vertex gradient
// scatter, a mass-matrix diagonal, a per-element residual -- each rank computes
// only the contribution of the faces it owns. A vertex on a partition boundary
// is incident on faces owned by several ranks, so its owner ends up with a
// PARTIAL sum and every ghost copy holds a different partial sum. Correct
// assembly requires each rank to push its ghost partials back to the owner and
// add them there; that is this function.
//
// The plan needed for the reverse direction already exists and is already
// correct: HaloExchangePlan is symmetric by construction (`send_idx` are owned
// local indices, `recv_idx` are ghost local indices, and the builder guarantees
// the per-peer alignment on both sides). The only new machinery is a reverse
// pack/unpack with `+=` on the unpack side.
//
// WHY THIS IS FIELD-TEMPLATED WHILE haloExchange() IS WHOLE-TUPLE. haloExchange()
// ships the entire AoSoA tuple as one opaque MPI_Type_contiguous of
// sizeof(tuple_type) bytes, which is right for a gather: overwriting a ghost with
// its owner's tuple is correct for every member at once, Gid/Owner/Level and the
// connectivity gids included. It is WRONG for an accumulate -- summing Gid or
// Owner is meaningless and summing connectivity gids is corrupting. So the
// reverse operation must name the one field it accumulates.
//
// THE CONTRACT (three properties a caller will otherwise get wrong):
//
//   1. GHOST SLOTS ARE LEFT UNTOUCHED. After the call an owned entry holds the
//      complete global sum and every ghost copy still holds that rank's local
//      partial, so the mesh is NOT halo-consistent for that field. Follow with
//      haloExchange() if downstream kernels read ghosts. The ghosts are
//      deliberately not zeroed here: a caller wanting assemble-then-broadcast
//      calls haloExchange(), and a caller that only reads owned values must not
//      pay for a second collective.
//
//   2. CALLING IT TWICE DOUBLE-COUNTS. It is not idempotent, precisely because
//      of (1) -- the second call re-sends the same ghost partials. This is the
//      standard scatter-add contract.
//
//   3. THE SUM ORDER IS FIXED BY PEER ORDER, NOT BY RANK COUNT. Peers are
//      visited in ascending rank on both sides (HaloExchangePlan::setFromHost
//      packs from an ordered std::map) and the unpack is serialized per peer, so
//      within one run the floating-point result is deterministic and bitwise
//      reproducible. It is NOT bitwise identical across rank counts, because the
//      partition into partial sums differs. A consumer comparing an assembled
//      field across rank counts must not expect bitwise equality.
//
// WHY THE UNPACK NEEDS NO ATOMICS. plan.send_idx may name the same owned entity
// once per peer, so a single flat parallel_for over the whole receive buffer
// WOULD race. Rather than pay for atomics on the GPU, the peer loop is on the
// host and one kernel is launched per peer over that peer's contiguous slice.
// Within one peer's slice an owned index appears at most once (the builder emits
// one entry per shared entity per peer), so each kernel is race-free -- and the
// serialization across peers is exactly what fixes the summation order in
// property (3).
//
// NON-GOALS: a generic reverse reduction with a caller-supplied operator (min,
// max, custom) -- `+=` is what assembly needs and a template-on-op API is harder
// to keep race-free; and multi-field/whole-pack scatter-add -- one field per
// call, so a caller with three fields makes three calls.

namespace detail
{

//! Component count of an AoSoA member type: 1 for a scalar member (`double`),
//! the array extent for a rank-1 member (`double[3]`). Rank-2+ members are not
//! supported by the scatter-add (they would need a second index in the access
//! helper below and no consumer has one).
template <class MemberType>
constexpr int scatterAddComponents()
{
    static_assert( std::rank<MemberType>::value <= 1,
                   "Tessera::haloScatterAdd supports scalar (rank-0) and "
                   "array (rank-1) AoSoA members only" );
    return ( std::rank<MemberType>::value == 0 )
               ? 1
               : static_cast<int>( std::extent<MemberType, 0>::value );
}

//! One component of one element of a slice, for a rank-0 or rank-1 member. Lets
//! the pack/unpack kernels share a single code path across both shapes.
//! `SliceType::rank` is std::rank<member>+1, so 1 == scalar, 2 == array.
template <class SliceType>
KOKKOS_INLINE_FUNCTION typename SliceType::reference_type
scatterAddRef( const SliceType& s, const int i, const int c )
{
    if constexpr ( static_cast<int>( SliceType::rank ) == 1 )
    {
        (void)c;
        return s( i );
    }
    else
    {
        return s( i, c );
    }
}

} // namespace detail

//! Accumulate one field from ghost slots into their owners (ghost -> owner, +=).
//! FieldIndex is the Cabana member index within the AoSoA's member type list --
//! a core field (e.g. VertexField::Position) or a user field
//! (userVertexField<M>()).
//!
//! Direction is the exact reverse of haloExchange(): pack from plan.recv_idx and
//! send along plan.recv_peers; receive along plan.send_peers and accumulate into
//! plan.send_idx. Multi-component members are accumulated componentwise. The
//! AoSoA is never resized and no other member is touched.
//!
//! Self-peer entries never appear in the plan, so a single rank has an empty plan
//! and the call returns immediately, as haloExchange() does.
template <std::size_t FieldIndex, class AoSoAType, class MemorySpace>
void haloScatterAdd( MPI_Comm comm, AoSoAType& aosoa,
                     HaloExchangePlan<MemorySpace>& plan )
{
    using execution_space = typename AoSoAType::execution_space;
    using member_type = typename Cabana::MemberTypeAtIndex<
        FieldIndex, typename AoSoAType::member_types>::type;
    using value_type = std::remove_all_extents_t<member_type>;
    constexpr int ncomp = detail::scatterAddComponents<member_type>();
    using buf_view = Kokkos::View<value_type*, MemorySpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Roles swap relative to haloExchange(): we PACK the ghost slots (totalRecv
    // of them) and RECEIVE into the owned slots (totalSend of them).
    const std::size_t n_pack = plan.totalRecv();
    const std::size_t n_recv = plan.totalSend();
    if ( n_pack == 0 && n_recv == 0 )
        return; // single rank / no ghosts: no MPI on device buffers

    const std::size_t elem = sizeof( value_type ) * ncomp;
    plan.send_pool.reserve( n_pack * elem );
    plan.recv_pool.reserve( n_recv * elem );
    buf_view send_buf( reinterpret_cast<value_type*>( plan.send_pool.data() ),
                       n_pack * ncomp );
    buf_view recv_buf( reinterpret_cast<value_type*>( plan.recv_pool.data() ),
                       n_recv * ncomp );

    auto field = Cabana::slice<FieldIndex>( aosoa );

    // ---- pack the ghost partials -------------------------------------------
    if ( n_pack > 0 )
    {
        auto idx = plan.recv_idx;
        auto out = send_buf;
        auto src = field;
        Kokkos::parallel_for(
            "tessera_halo_scatter_pack",
            Kokkos::RangePolicy<execution_space>( 0,
                                                  static_cast<int>( n_pack ) ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int c = 0; c < ncomp; ++c )
                    out( i * ncomp + c ) =
                        detail::scatterAddRef( src, idx( i ), c );
            } );
        Kokkos::fence();
    }

    // ---- exchange: one MPI element = one field entry (all components) ------
    MPI_Datatype elem_dtype;
    MPI_Type_contiguous( static_cast<int>( elem ), MPI_BYTE, &elem_dtype );
    MPI_Type_commit( &elem_dtype );

    std::vector<MPI_Request> recv_reqs;
    for ( std::size_t q = 0; q < plan.send_peers.size(); ++q )
    {
        const int n = plan.send_off[q + 1] - plan.send_off[q];
        MPI_Request req;
        MPI_Irecv( recv_buf.data() + plan.send_off[q] * ncomp, n, elem_dtype,
                   plan.send_peers[q], /*tag=*/0, comm, &req );
        recv_reqs.push_back( req );
    }
    std::vector<MPI_Request> send_reqs;
    for ( std::size_t q = 0; q < plan.recv_peers.size(); ++q )
    {
        const int n = plan.recv_off[q + 1] - plan.recv_off[q];
        MPI_Request req;
        MPI_Isend( send_buf.data() + plan.recv_off[q] * ncomp, n, elem_dtype,
                   plan.recv_peers[q], /*tag=*/0, comm, &req );
        send_reqs.push_back( req );
    }
    if ( !recv_reqs.empty() )
        MPI_Waitall( static_cast<int>( recv_reqs.size() ), recv_reqs.data(),
                     MPI_STATUSES_IGNORE );
    if ( !send_reqs.empty() )
        MPI_Waitall( static_cast<int>( send_reqs.size() ), send_reqs.data(),
                     MPI_STATUSES_IGNORE );
    MPI_Type_free( &elem_dtype );

    // ---- accumulate into the owned slots, ONE KERNEL PER PEER --------------
    // See "WHY THE UNPACK NEEDS NO ATOMICS" above: an owned index is unique
    // within a peer's slice but not across peers, so the peer loop stays on the
    // host and the per-peer kernels serialize on the execution space's default
    // instance -- which is also what makes the summation order deterministic.
    for ( std::size_t q = 0; q < plan.send_peers.size(); ++q )
    {
        const int begin = plan.send_off[q];
        const int end = plan.send_off[q + 1];
        if ( end <= begin )
            continue;
        auto idx = plan.send_idx;
        auto in = recv_buf;
        auto dst = field;
        Kokkos::parallel_for(
            "tessera_halo_scatter_unpack",
            Kokkos::RangePolicy<execution_space>( begin, end ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int c = 0; c < ncomp; ++c )
                    detail::scatterAddRef( dst, idx( i ), c ) +=
                        in( i * ncomp + c );
            } );
    }
    Kokkos::fence();
}

// ============================================================================
// Kind-named conveniences over a MeshHalo
// ============================================================================
//
// One field of one entity kind per call, mirroring haloExchange(mesh, halo)'s
// spelling but NOT its all-three-kinds behaviour -- an accumulate names a field,
// and a field belongs to one kind.

template <std::size_t FieldIndex, class MeshT, class MemorySpace>
void haloScatterAddVertices( MeshT& mesh, MeshHalo<MemorySpace>& halo )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_HALO_SCATTER_ADD );
    haloScatterAdd<FieldIndex>( mesh.comm(), mesh.vertices(), halo.vplan );
}

template <std::size_t FieldIndex, class MeshT, class MemorySpace>
void haloScatterAddEdges( MeshT& mesh, MeshHalo<MemorySpace>& halo )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_HALO_SCATTER_ADD );
    haloScatterAdd<FieldIndex>( mesh.comm(), mesh.edges(), halo.eplan );
}

template <std::size_t FieldIndex, class MeshT, class MemorySpace>
void haloScatterAddFaces( MeshT& mesh, MeshHalo<MemorySpace>& halo )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_HALO_SCATTER_ADD );
    haloScatterAdd<FieldIndex>( mesh.comm(), mesh.faces(), halo.fplan );
}

} // namespace Tessera

#endif // TESSERA_HALO_SCATTER_ADD_HPP
