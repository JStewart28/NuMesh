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
//! Map a scalar C++ type to its MPI datatype (the arithmetic types the mesh
//! coordinate / user-field scalars use).
template <class T>
MPI_Datatype mpiTypeOf()
{
    if ( std::is_same<T, double>::value )
        return MPI_DOUBLE;
    if ( std::is_same<T, float>::value )
        return MPI_FLOAT;
    if ( std::is_same<T, int>::value )
        return MPI_INT;
    if ( std::is_same<T, long>::value )
        return MPI_LONG;
    static_assert( std::is_arithmetic<T>::value,
                   "globalMin requires an arithmetic scalar type" );
    return MPI_DATATYPE_NULL;
}
} // namespace detail

// ============================================================================
// globalMin — collective minimum of a per-rank scalar over the mesh comm
// ============================================================================
//
// A thin, single-sourced wrapper over MPI_Allreduce(MPI_MIN) on mesh.comm().
// Every rank passes its local value and receives the global minimum. This is
// the primitive behind an adaptive-timestep min-reduce; the driver owns the
// local dt estimate, Tessera owns only the one collective.
template <class MeshT, class Scalar>
Scalar globalMin( const MeshT& mesh, Scalar local )
{
    Scalar global = local;
    MPI_Allreduce( &local, &global, 1, detail::mpiTypeOf<Scalar>(), MPI_MIN,
                   mesh.comm() );
    return global;
}

} // namespace Tessera

#endif // TESSERA_REDUCTION_HPP
