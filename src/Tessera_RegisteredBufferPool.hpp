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

#ifndef TESSERA_REGISTERED_BUFFER_POOL_HPP
#define TESSERA_REGISTERED_BUFFER_POOL_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace Tessera
{
namespace detail
{

// ============================================================================
// RegisteredBufferPool
// ============================================================================
//
// A persistent, grow-only device buffer reused across communication calls for
// the send/recv staging of GPU-aware MPI exchanges. Ported from Canopy
// (Canopy_RegisteredBufferPool.hpp) — the rationale is identical:
//
// On Slingshot/CXI with GPU-aware Cray-MPICH, every fresh device allocation
// handed to MPI_Isend/MPI_Irecv triggers a fresh NIC memory registration. A
// mesh that refines and rebalances repeatedly would allocate a new staging
// buffer per peer per call, churning the registration cache until the NIC runs
// out of registration resources and aborts. A pool keeps ONE allocation with a
// stable base address that only grows (1.5x headroom, never shrinks); all peers
// for a direction are packed into non-overlapping [offset, offset+n) sub-ranges
// of that single region, so the registration footprint stays bounded.
//
// Usage:
//   pool.reserve( total_elems );                 // once, before any subview()
//   auto v = pool.subview( peer_off, peer_n );   // unmanaged view per peer
//   MPI_Isend( v.data(), ... );                  // points into the one region
//
// Caller contract: reserve() the full total for the call BEFORE taking any
// subview(), so the base address is stable for the lifetime of every subview
// handed out that call. Distinct peers must use non-overlapping ranges.
//
template <class T, class MemorySpace>
class RegisteredBufferPool
{
  public:
    using memory_space = MemorySpace;
    using view_type = Kokkos::View<T*, memory_space>;
    using unmanaged_view_type =
        Kokkos::View<T*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    // Ensure capacity for at least n elements of T. Grow-only with 1.5x
    // headroom; never shrinks. A grow reallocates (and re-registers) once, but
    // capacity converges to the working-set size so grows are log-many over a
    // run, not per-call.
    void reserve( std::size_t n )
    {
        if ( n > _capacity )
        {
            const std::size_t new_cap = n + n / 2; // 1.5x headroom
            _buf =
                view_type( Kokkos::view_alloc( "Tessera_RegisteredBufferPool",
                                               Kokkos::WithoutInitializing ),
                           new_cap );
            _capacity = new_cap;
        }
    }

    // Unmanaged view of the [offset, offset+n) element range of the pool. Valid
    // only while no intervening reserve() has grown the pool. The caller
    // guarantees offset + n <= the reserved capacity.
    unmanaged_view_type subview( std::size_t offset, std::size_t n ) const
    {
        return unmanaged_view_type( _buf.data() + offset, n );
    }

    T* data() const { return _buf.data(); }
    std::size_t capacity() const { return _capacity; }

  private:
    view_type _buf;
    std::size_t _capacity = 0;
};

} // namespace detail
} // namespace Tessera

#endif // TESSERA_REGISTERED_BUFFER_POOL_HPP
