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

#ifndef TESSERA_IO_COMMON_HPP
#define TESSERA_IO_COMMON_HPP

#include "Tessera_Fields.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>

#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

namespace Tessera
{
namespace detail
{

// ============================================================================
// h5_type<T>() — H5 native predefined datatype for a C++ scalar type used by
// the mesh I/O datasets (Step 8).
// ============================================================================
template <class T>
hid_t h5_type();

template <>
inline hid_t h5_type<double>()
{
    return H5T_NATIVE_DOUBLE;
}
template <>
inline hid_t h5_type<float>()
{
    return H5T_NATIVE_FLOAT;
}
template <>
inline hid_t h5_type<std::uint64_t>()
{
    return H5T_NATIVE_UINT64;
}
template <>
inline hid_t h5_type<std::int16_t>()
{
    return H5T_NATIVE_INT16;
}
template <>
inline hid_t h5_type<std::int32_t>()
{
    return H5T_NATIVE_INT32;
}

// ============================================================================
// Dense global numbering — the MPI_Exscan core (spec 8.2 step 1).
// ============================================================================

//! Global count (Allreduce SUM) + this rank's exclusive-scan offset (Exscan
//! SUM, rank 0 -> 0) for an owned-only per-rank count.
struct GlobalCount
{
    long long N = 0;
    long long off = 0;
};

inline GlobalCount exscanCount( MPI_Comm comm, long long nOwned )
{
    GlobalCount gc;
    MPI_Allreduce( &nOwned, &gc.N, 1, MPI_LONG_LONG, MPI_SUM, comm );
    long long off = 0;
    MPI_Exscan( &nOwned, &off, 1, MPI_LONG_LONG, MPI_SUM, comm );
    int rank = 0;
    MPI_Comm_rank( comm, &rank );
    gc.off = ( rank == 0 ) ? 0 : off;
    return gc;
}

// ============================================================================
// Block-partition arithmetic (reader's fresh dense-index partition, spec 8.5).
// ============================================================================

//! Half-open block [begin,end) owned by rank R out of `size` under a
//! contiguous N-element block partition (R*N/size .. (R+1)*N/size).
inline void blockRange( long long N, int R, int size, long long& begin,
                        long long& end )
{
    begin = ( size <= 0 ) ? 0 : ( static_cast<long long>( R ) * N ) / size;
    end = ( size <= 0 ) ? 0 : ( static_cast<long long>( R + 1 ) * N ) / size;
}

//! Which block-rank owns dense index `idx` under the same partition as
//! blockRange(). Handles the floor-division rounding directly rather than
//! assuming a closed form is exactly invertible.
inline int blockOwner( long long idx, long long N, int size )
{
    if ( size <= 1 || N == 0 )
        return 0;
    int r = static_cast<int>( ( idx * static_cast<long long>( size ) ) / N );
    if ( r >= size )
        r = size - 1;
    if ( r < 0 )
        r = 0;
    long long b, e;
    blockRange( N, r, size, b, e );
    while ( r > 0 && idx < b )
    {
        --r;
        blockRange( N, r, size, b, e );
    }
    while ( r < size - 1 && idx >= e )
    {
        ++r;
        blockRange( N, r, size, b, e );
    }
    return r;
}

// ============================================================================
// Compile-time user-field iteration (the Step 6a empty-pack guard pattern).
// ============================================================================
//
// Calls fn(std::integral_constant<std::size_t, UserBegin+J>{}) for every user
// field J of the entity kind's AoSoA type; a no-op when the user pack is
// empty (Cabana::slice<UserBegin> is ill-formed for an empty pack, so the
// whole loop body is behind `if constexpr`).
template <std::size_t UserBegin, class AoSoAType, class Fn, std::size_t... Js>
void forEachUserFieldImpl( Fn&& fn, std::index_sequence<Js...> )
{
    ( fn( std::integral_constant<std::size_t, UserBegin + Js>{} ), ... );
}

//! Iterate exactly `N` user fields starting at `UserBegin`. Use this whenever
//! the user pack is NOT the tuple's suffix -- which is the case for FACES in
//! RefinementMode::Conforming, where the two closure bookkeeping members are
//! appended after the user pack (see numFaceUserFields<>()). Iterating to the
//! end of the tuple there would treat ClosureParent/ClosureParentVerts as user
//! fields "u<n>"/"u<n+1>" on disk.
template <std::size_t UserBegin, std::size_t N, class AoSoAType, class Fn>
void forEachUserFieldN( Fn&& fn )
{
    if constexpr ( N > 0 )
        forEachUserFieldImpl<UserBegin, AoSoAType>(
            std::forward<Fn>( fn ), std::make_index_sequence<N>{} );
}

template <std::size_t UserBegin, class AoSoAType, class Fn>
void forEachUserField( Fn&& fn )
{
    constexpr std::size_t total = AoSoAType::member_types::size;
    constexpr std::size_t N = total > UserBegin ? total - UserBegin : 0;
    forEachUserFieldN<UserBegin, N, AoSoAType>( std::forward<Fn>( fn ) );
}

//! Number of user fields (0 for an empty pack) of an entity kind's AoSoA.
template <class AoSoAType, std::size_t UserBegin>
constexpr std::size_t userFieldCount()
{
    constexpr std::size_t total = AoSoAType::member_types::size;
    return total > UserBegin ? total - UserBegin : 0;
}

//! Compile-time extent (1 for a scalar member, N for Scalar[N]) and element
//! scalar type of member `Mabs` of an AoSoA's member-type list.
template <class AoSoAType, std::size_t Mabs>
struct FieldInfo
{
    using member_types = typename AoSoAType::member_types;
    using field_type =
        typename Cabana::MemberTypeAtIndex<Mabs, member_types>::type;
    using scalar_type = typename std::remove_all_extents<field_type>::type;
    static constexpr int extent =
        ( std::rank<field_type>::value == 0 )
            ? 1
            : static_cast<int>( std::extent<field_type, 0>::value );
};

// ============================================================================
// Collective hyperslab read/write of an owned-block dataset (spec 8.3).
// ============================================================================
//
// Zero-owned-rank gotcha: every rank must call H5Dcreate2/H5Dwrite (or
// H5Dopen2/H5Dread) collectively even when it has nothing to select
// (H5Sselect_none on both the file- and mem-space) -- skipping the call
// deadlocks the collective operation.

//! Create (collectively, identical on every rank) and write a dataset `name`
//! of global shape {Nrows,width} (width==1 -> a 1-D dataset). This rank's
//! owned rows are the hyperslab [rowOff,rowOff+nRows); `data` holds
//! nRows*width row-major values (ignored/may be null when nRows==0).
template <class T>
void writeHyperslab( hid_t loc, const std::string& name, long long Nrows,
                     int width, long long rowOff, long long nRows,
                     const T* data )
{
    const int ndims = ( width > 1 ) ? 2 : 1;
    hsize_t gdims[2] = { static_cast<hsize_t>( Nrows ),
                         static_cast<hsize_t>( width ) };
    hid_t filespace = H5Screate_simple( ndims, gdims, nullptr );
    hid_t dset = H5Dcreate2( loc, name.c_str(), h5_type<T>(), filespace,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    H5Sclose( filespace );

    filespace = H5Dget_space( dset );
    hid_t memspace;
    if ( nRows > 0 )
    {
        hsize_t start[2] = { static_cast<hsize_t>( rowOff ), 0 };
        hsize_t count[2] = { static_cast<hsize_t>( nRows ),
                             static_cast<hsize_t>( width ) };
        H5Sselect_hyperslab( filespace, H5S_SELECT_SET, start, nullptr, count,
                             nullptr );
        memspace = H5Screate_simple( ndims, count, nullptr );
    }
    else
    {
        H5Sselect_none( filespace );
        hsize_t zero[2] = { 0, static_cast<hsize_t>( width ) };
        memspace = H5Screate_simple( ndims, zero, nullptr );
        H5Sselect_none( memspace );
    }

    hid_t dxpl = H5Pcreate( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio( dxpl, H5FD_MPIO_COLLECTIVE );
    H5Dwrite( dset, h5_type<T>(), memspace, filespace, dxpl, data );

    H5Pclose( dxpl );
    H5Sclose( memspace );
    H5Sclose( filespace );
    H5Dclose( dset );
}

//! Collectively read a hyperslab [rowOff,rowOff+nRows) of dataset `name`
//! (global shape {Nrows,width}) into `data` (nRows*width row-major values;
//! `data` may be null when nRows==0, per the zero-owned guard above).
template <class T>
void readHyperslab( hid_t loc, const std::string& name, int width,
                    long long rowOff, long long nRows, T* data )
{
    hid_t dset = H5Dopen2( loc, name.c_str(), H5P_DEFAULT );
    hid_t filespace = H5Dget_space( dset );
    const int ndims = ( width > 1 ) ? 2 : 1;
    hid_t memspace;
    if ( nRows > 0 )
    {
        hsize_t start[2] = { static_cast<hsize_t>( rowOff ), 0 };
        hsize_t count[2] = { static_cast<hsize_t>( nRows ),
                             static_cast<hsize_t>( width ) };
        H5Sselect_hyperslab( filespace, H5S_SELECT_SET, start, nullptr, count,
                             nullptr );
        memspace = H5Screate_simple( ndims, count, nullptr );
    }
    else
    {
        H5Sselect_none( filespace );
        hsize_t zero[2] = { 0, static_cast<hsize_t>( width ) };
        memspace = H5Screate_simple( ndims, zero, nullptr );
        H5Sselect_none( memspace );
    }

    hid_t dxpl = H5Pcreate( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio( dxpl, H5FD_MPIO_COLLECTIVE );
    H5Dread( dset, h5_type<T>(), memspace, filespace, dxpl, data );

    H5Pclose( dxpl );
    H5Sclose( memspace );
    H5Sclose( filespace );
    H5Dclose( dset );
}

// ============================================================================
// Root scalar attributes (written identically by all ranks; safe under MPIO).
// A file identifier doubles as the root group's location for attribute I/O.
// ============================================================================
template <class T>
void writeAttr( hid_t loc, const std::string& name, T value, hid_t h5t )
{
    hid_t space = H5Screate( H5S_SCALAR );
    hid_t attr =
        H5Acreate2( loc, name.c_str(), h5t, space, H5P_DEFAULT, H5P_DEFAULT );
    H5Awrite( attr, h5t, &value );
    H5Aclose( attr );
    H5Sclose( space );
}

inline void writeIntAttr( hid_t loc, const std::string& name, int value )
{
    writeAttr( loc, name, value, H5T_NATIVE_INT );
}
inline void writeU64Attr( hid_t loc, const std::string& name,
                          std::uint64_t value )
{
    writeAttr( loc, name, value, H5T_NATIVE_UINT64 );
}

inline int readIntAttr( hid_t loc, const std::string& name )
{
    int value = 0;
    hid_t attr = H5Aopen( loc, name.c_str(), H5P_DEFAULT );
    H5Aread( attr, H5T_NATIVE_INT, &value );
    H5Aclose( attr );
    return value;
}
inline std::uint64_t readU64Attr( hid_t loc, const std::string& name )
{
    std::uint64_t value = 0;
    hid_t attr = H5Aopen( loc, name.c_str(), H5P_DEFAULT );
    H5Aread( attr, H5T_NATIVE_UINT64, &value );
    H5Aclose( attr );
    return value;
}

//! Collectively abort with a clear message if any rank found a validated root
//! attribute mismatched against its compile-time template (spec 8.5 step 1).
inline void abortOnMismatch( MPI_Comm comm, bool ok, const char* what )
{
    int local_bad = ok ? 0 : 1, global_bad = 0;
    MPI_Allreduce( &local_bad, &global_bad, 1, MPI_INT, MPI_SUM, comm );
    if ( global_bad != 0 )
    {
        if ( !ok )
            std::fprintf( stderr,
                          "Tessera readMesh: root attribute mismatch: %s\n",
                          what );
        MPI_Abort( comm, 1 );
    }
}

} // namespace detail
} // namespace Tessera

#endif // TESSERA_IO_COMMON_HPP
