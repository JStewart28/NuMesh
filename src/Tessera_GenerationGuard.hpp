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

#ifndef TESSERA_GENERATION_GUARD_HPP
#define TESSERA_GENERATION_GUARD_HPP

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace Tessera
{

// ============================================================================
// GenerationHandle
// ============================================================================
//
// INVALIDATION: any change to the local entity count or the ghost set
// (distribution, migration, refinement) invalidates every slice/CSR/key-View
// handed out before the change. `Mesh::vertexSlice()`/`edgeSlice()`/
// `faceSlice()` and the CSR/key-View "handle" accessors wrap their returned
// handle in a GenerationHandle stamped with the mesh's generation at creation
// time. `haloExchange()` does NOT change the local count or ghost set, so it
// never bumps the generation and handles taken before it remain valid across
// it.
//
// Validation is host-side only, and runs when a handle is COPIED (its copy
// constructor / copy-assignment), not on every element access: capturing a
// handle by value into a KOKKOS_LAMBDA invokes this copy on the host, right
// before kernel dispatch, which is exactly the point a stale-handle bug should
// be caught. `operator()` forwards to the underlying handle unconditionally
// (device-callable, no branch), so per-element access on the device costs
// nothing extra.
//
// The check itself is gated on TESSERA_ENABLE_DEBUG_CHECKS (on by default; see
// the Tessera_ENABLE_DEBUG_CHECKS CMake option) and is additionally elided
// under device compilation so it never touches the host-side generation
// pointer from device code.
template <class Underlying>
class GenerationHandle
{
  public:
    GenerationHandle() = default;

    KOKKOS_INLINE_FUNCTION
    GenerationHandle( Underlying u, std::size_t gen,
                      const std::size_t* mesh_generation )
        : _u( std::move( u ) )
        , _gen( gen )
        , _mesh_generation( mesh_generation )
    {
    }

    KOKKOS_INLINE_FUNCTION
    GenerationHandle( const GenerationHandle& other )
        : _u( other._u )
        , _gen( other._gen )
        , _mesh_generation( other._mesh_generation )
    {
#if defined( TESSERA_ENABLE_DEBUG_CHECKS ) && !defined( __CUDA_ARCH__ ) &&     \
    !defined( __HIP_DEVICE_COMPILE__ )
        validate();
#endif
    }

    KOKKOS_INLINE_FUNCTION
    GenerationHandle& operator=( const GenerationHandle& other )
    {
        _u = other._u;
        _gen = other._gen;
        _mesh_generation = other._mesh_generation;
#if defined( TESSERA_ENABLE_DEBUG_CHECKS ) && !defined( __CUDA_ARCH__ ) &&     \
    !defined( __HIP_DEVICE_COMPILE__ )
        validate();
#endif
        return *this;
    }

    //! Forward element access to the underlying slice/view. No validation here
    //! by design -- see the class comment; this keeps device-side access as
    //! cheap as using the underlying handle directly.
    template <class... Args>
    KOKKOS_INLINE_FUNCTION decltype( auto ) operator()( Args&&... args ) const
    {
        return _u( std::forward<Args>( args )... );
    }

    //! Host-side accessor for handle types with no operator() (e.g. CSR
    //! snapshots, key-Views). Validates before returning.
    const Underlying& get() const
    {
        validate();
        return _u;
    }

    //! Explicit host-side check, e.g. immediately before a kernel launch that
    //! doesn't otherwise copy the handle.
    void validate() const
    {
#ifdef TESSERA_ENABLE_DEBUG_CHECKS
        if ( _mesh_generation && *_mesh_generation != _gen )
        {
            std::fprintf(
                stderr,
                "Tessera: stale handle used after a topology change "
                "(captured generation %zu, mesh is now generation %zu). "
                "INVALIDATION: this handle was taken from the mesh before a "
                "distribute()/migrate()/refine() call changed the local "
                "entity count or ghost set; re-slice from the mesh after the "
                "topology-changing call. haloExchange() does not trigger "
                "this -- it is topology-preserving.\n",
                _gen, *_mesh_generation );
            std::abort();
        }
#endif
    }

  private:
    Underlying _u{};
    std::size_t _gen = 0;
    const std::size_t* _mesh_generation = nullptr;
};

} // namespace Tessera

#endif // TESSERA_GENERATION_GUARD_HPP
