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

#ifndef TESSERA_REDUCTION_HPP
#define TESSERA_REDUCTION_HPP

#include <mpi.h>

#include <type_traits>

namespace Tessera
{

namespace detail
{
//! Dependent false, so the primary-template static_assert below fires only when
//! the template is actually instantiated (a bare `false` would fire always).
template <class T>
struct AlwaysFalse : std::false_type
{
};

//! Map a scalar C++ type to its MPI datatype.
//!
//! This is a specialization set rather than an `if` chain over `std::is_same`:
//! the chain was resolved at *runtime* (every call ran through every branch),
//! and — the actual bug — it could not express "reject an unmapped type". Its
//! trailing `static_assert( std::is_arithmetic<T>::value )` passes for e.g.
//! `long long`, so the `MPI_DATATYPE_NULL` fallthrough escaped into
//! MPI_Allreduce and failed at runtime with an MPI error. An unmapped
//! arithmetic type must be a *compile* error, and only a specialization set
//! makes that expressible.
//!
//! MPI_Datatype handles are not guaranteed to be constant expressions, so the
//! mapping is exposed as a static function, not a static member.
template <class T>
struct MpiType
{
    static_assert( AlwaysFalse<T>::value,
                   "Tessera: no MPI_Datatype mapping for this scalar type. "
                   "Add a detail::MpiType specialization in "
                   "Tessera_Reduction.hpp if the type is genuinely needed "
                   "(note that long double has no portable MPI mapping)." );
};

#define TESSERA_MPI_TYPE_MAP( CXX_TYPE, MPI_HANDLE )                           \
    template <>                                                                \
    struct MpiType<CXX_TYPE>                                                   \
    {                                                                          \
        static MPI_Datatype value() { return MPI_HANDLE; }                     \
    }

TESSERA_MPI_TYPE_MAP( double, MPI_DOUBLE );
TESSERA_MPI_TYPE_MAP( float, MPI_FLOAT );
TESSERA_MPI_TYPE_MAP( int, MPI_INT );
TESSERA_MPI_TYPE_MAP( long, MPI_LONG );
TESSERA_MPI_TYPE_MAP( long long, MPI_LONG_LONG );
TESSERA_MPI_TYPE_MAP( unsigned int, MPI_UNSIGNED );
TESSERA_MPI_TYPE_MAP( unsigned long, MPI_UNSIGNED_LONG );
TESSERA_MPI_TYPE_MAP( unsigned long long, MPI_UNSIGNED_LONG_LONG );
TESSERA_MPI_TYPE_MAP( short, MPI_SHORT );
TESSERA_MPI_TYPE_MAP( char, MPI_CHAR );

#undef TESSERA_MPI_TYPE_MAP

//! The MPI datatype for `T`. Hard compile error for any type with no mapping.
template <class T>
MPI_Datatype mpiTypeOf()
{
    return MpiType<T>::value();
}

//! The one place a scalar Allreduce over the mesh comm is written.
template <class MeshT, class Scalar>
Scalar meshAllreduce( const MeshT& mesh, Scalar local, MPI_Op op )
{
    Scalar global = local;
    MPI_Allreduce( &local, &global, 1, mpiTypeOf<Scalar>(), op, mesh.comm() );
    return global;
}
} // namespace detail

// ============================================================================
// Scalar collectives over the mesh comm
// ============================================================================
//
// Thin, single-sourced wrappers over one MPI_Allreduce on mesh.comm(). Every
// rank passes its local value and receives the global result. **Tessera owns
// the collective; the caller owns the local value** — Tessera cannot know which
// fields a given consumer cares about, so it never computes the local scalar.
//
// All are collective: every rank on mesh.comm() must call them.

//! Collective minimum of a per-rank scalar. This is the primitive behind an
//! adaptive-timestep min-reduce; the driver owns the local dt estimate,
//! Tessera owns only the one collective.
template <class MeshT, class Scalar>
Scalar globalMin( const MeshT& mesh, Scalar local )
{
    return detail::meshAllreduce( mesh, local, MPI_MIN );
}

//! Collective maximum of a per-rank scalar. Mirror of globalMin; the primitive
//! behind a CFL-style upper bound.
template <class MeshT, class Scalar>
Scalar globalMax( const MeshT& mesh, Scalar local )
{
    return detail::meshAllreduce( mesh, local, MPI_MAX );
}

//! Collective sum of a per-rank scalar over the mesh comm.
//!
//! REPRODUCIBILITY: MPI_SUM is not associative in floating point, so a `double`
//! result is NOT bitwise reproducible across rank counts, nor across runs on a
//! GPU partial-sum path. A quantity carried for the whole run — an enclosed
//! volume, a reduced minimum edge length — will therefore differ in its low bits
//! between a 4-rank and a 5-rank run, and anything scaling off it (an adaptive
//! timestep) diverges from there. Integer sums ARE exact and reproducible.
//! Callers comparing results across rank counts must account for this; do not
//! write a cross-rank bitwise comparison over a floating-point globalSum.
template <class MeshT, class Scalar>
Scalar globalSum( const MeshT& mesh, Scalar local )
{
    return detail::meshAllreduce( mesh, local, MPI_SUM );
}

//! Collective logical AND of a per-rank "everything I hold is finite" verdict.
//! Returns true iff every rank passed true. Implemented as
//! MPI_Allreduce(MPI_LAND) on an int, because MPI_CXX_BOOL is not universally
//! available.
//!
//! This is the NaN/Inf tripwire: the caller checks its own owned data (a Kokkos
//! reduction over whichever fields it cares about) and this call turns that
//! local verdict into a global one, so every rank aborts on the step that
//! produced the NaN rather than one rank diverging silently for fifty steps.
//!
//! NOTE: this takes a *verdict*, not data. There is deliberately no device-side
//! "is this slice all finite" helper — the local sweep is the caller's, because
//! only the caller knows which fields matter. Do not look for the missing half.
template <class MeshT>
bool globalAllFinite( const MeshT& mesh, bool local_all_finite )
{
    const int local = local_all_finite ? 1 : 0;
    int global = local;
    MPI_Allreduce( &local, &global, 1, MPI_INT, MPI_LAND, mesh.comm() );
    return global != 0;
}

// ============================================================================
// Owned-entity count reductions
// ============================================================================
//
// Owned entities partition the global mesh, so a SUM of the owned counts is the
// global count. Reduced as `long long` — the natural type for a production-scale
// entity count — and exact, since integer.

//! Global vertex count of a distributed mesh.
template <class MeshT>
long long globalOwnedVertices( const MeshT& mesh )
{
    return globalSum( mesh, static_cast<long long>( mesh.numOwnedVertices() ) );
}

//! Global edge count of a distributed mesh.
template <class MeshT>
long long globalOwnedEdges( const MeshT& mesh )
{
    return globalSum( mesh, static_cast<long long>( mesh.numOwnedEdges() ) );
}

//! Global face count of a distributed mesh.
template <class MeshT>
long long globalOwnedFaces( const MeshT& mesh )
{
    return globalSum( mesh, static_cast<long long>( mesh.numOwnedFaces() ) );
}

//! V - E + F over owned entities. 2 for a closed conforming genus-0 surface;
//! adaptive (hanging-node) refinement introduces bounded non-conformity and
//! does not preserve it.
template <class MeshT>
long long globalOwnedEuler( const MeshT& mesh )
{
    const long long local = static_cast<long long>( mesh.numOwnedVertices() ) -
                            static_cast<long long>( mesh.numOwnedEdges() ) +
                            static_cast<long long>( mesh.numOwnedFaces() );
    return globalSum( mesh, local );
}

} // namespace Tessera

#endif // TESSERA_REDUCTION_HPP
