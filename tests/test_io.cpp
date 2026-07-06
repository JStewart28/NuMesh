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

// Regression test: parallel HDF5 + XDMF writer/reader round-trip (Step 8).
// Builds a distributed icosphere with a non-trivial user field per vertex and
// per face, writes it, reads it back into a fresh mesh, and checks:
//   - ownership partition, owned-1-ring-local, owned Euler == 2 (uniform
//     coarse icosphere) on the round-tripped mesh,
//   - the rank-count-independent topology checksum is unchanged,
//   - the user-field checksums (vertex + face) are unchanged,
//   - the on-disk /*/gid BXOR (read serially by rank 0) equals the in-memory
//     checksum, demonstrating rank-count independence (the gate running
//     np1-5 confirms all five agree),
//   - a corrupt -> haloExchange -> restored-ghost check on the round-tripped
//     mesh (same as the distribute/migrate regression tests).
// Runs on host (Serial) and device (default, HIP), np1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <hdf5.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace Tessera;

// Reinterpret a Scalar's raw bytes as an (up to 8-byte) integer for BXOR
// checksums -- exact for the small deterministic values this test writes.
template <class Scalar>
static inline unsigned long long bitsOf( Scalar v )
{
    unsigned long long u = 0;
    std::memcpy( &u, &v, sizeof( Scalar ) );
    return u;
}

template <class MeshT>
static unsigned long long vertexFieldChecksum( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    const std::size_t no = mesh.numOwnedVertices();
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> h(
        "h", mesh.numVertices() );
    Cabana::deep_copy( h, mesh.vertices() );
    auto f = Cabana::slice<Tessera::userVertexField<0>()>( h );
    unsigned long long local = 0;
    for ( std::size_t i = 0; i < no; ++i )
        local ^= bitsOf<Scalar>( f( i ) );
    unsigned long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                   mesh.comm() );
    return global;
}

template <class MeshT>
static unsigned long long faceFieldChecksum( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    const std::size_t no = mesh.numOwnedFaces();
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> h(
        "h", mesh.numFaces() );
    Cabana::deep_copy( h, mesh.faces() );
    auto f = Cabana::slice<Tessera::userFaceField<0>()>( h );
    unsigned long long local = 0;
    for ( std::size_t i = 0; i < no; ++i )
        local ^= bitsOf<Scalar>( f( i ) );
    unsigned long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                   mesh.comm() );
    return global;
}

// Corrupt every ghost tuple, halo-sync, verify restoration (as in
// test_distribute.cpp / test_migrate_mesh.cpp).
template <class AoSoAType, class Mem>
static int corrupt_sync_verify( MPI_Comm comm, int rank, AoSoAType& a,
                                std::size_t n_owned,
                                HaloExchangePlan<Mem>& plan )
{
    using exec = typename AoSoAType::execution_space;
    const int n = static_cast<int>( a.size() );
    const int no = static_cast<int>( n_owned );

    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h0( "h0",
                                                                           n );
    Cabana::deep_copy( h0, a );
    auto g0 = Cabana::slice<0>( h0 );
    auto o0 = Cabana::slice<1>( h0 );
    std::vector<GlobalId> expg;
    std::vector<Rank> expo;
    for ( int i = no; i < n; ++i )
    {
        expg.push_back( g0( i ) );
        expo.push_back( o0( i ) );
    }

    auto g = Cabana::slice<0>( a );
    auto o = Cabana::slice<1>( a );
    Kokkos::parallel_for(
        "corrupt", Kokkos::RangePolicy<exec>( no, n ),
        KOKKOS_LAMBDA( const int i ) {
            g( i ) = ~static_cast<GlobalId>( 0 );
            o( i ) = -1;
        } );
    Kokkos::fence();

    haloExchange( comm, a, plan );

    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h1( "h1",
                                                                           n );
    Cabana::deep_copy( h1, a );
    auto g1 = Cabana::slice<0>( h1 );
    auto o1 = Cabana::slice<1>( h1 );
    int fails = 0;
    for ( int i = 0; i < no; ++i )
        if ( o1( i ) != rank )
            ++fails;
    for ( int i = no; i < n; ++i )
    {
        if ( g1( i ) != expg[i - no] || o1( i ) != expo[i - no] )
            ++fails;
        if ( o1( i ) == -1 || g1( i ) == ~static_cast<GlobalId>( 0 ) )
            ++fails;
    }
    return fails;
}

// Serial (rank 0 only) BXOR of a whole /group/gid dataset, for the on-disk
// rank-count-independence check.
static unsigned long long readWholeGidChecksum( const std::string& file,
                                                const char* dsetPath )
{
    hid_t f = H5Fopen( file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
    hid_t dset = H5Dopen2( f, dsetPath, H5P_DEFAULT );
    hid_t space = H5Dget_space( dset );
    hsize_t dims[1] = { 0 };
    H5Sget_simple_extent_dims( space, dims, nullptr );
    std::vector<std::uint64_t> buf( dims[0] );
    if ( dims[0] > 0 )
        H5Dread( dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 buf.data() );
    H5Sclose( space );
    H5Dclose( dset );
    H5Fclose( f );
    unsigned long long c = 0;
    for ( auto v : buf )
        c ^= v;
    return c;
}

template <class Exec>
int run( int rank, int size, const char* tag, const std::string& stem )
{
    using mem = typename Exec::memory_space;
    using Scalar = double;
    using MeshT = Mesh<Scalar, 3, VertexFields<Scalar>, EdgeFields<>,
                       FaceFields<Scalar>, mem, Exec>;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );
    const long long NvG = static_cast<long long>( mesh.numVertices() );
    const long long NeG = static_cast<long long>( mesh.numEdges() );
    const long long NfG = static_cast<long long>( mesh.numFaces() );

    // Deterministic user fields (function of gid) set on the replicated mesh
    // so they carry through distribute() automatically.
    {
        auto gid = mesh.template vertexSlice<VertexField::Gid>();
        auto vf = mesh.template vertexSlice<userVertexField<0>()>();
        const int nv = static_cast<int>( mesh.numVertices() );
        Kokkos::parallel_for(
            "set_vfield", Kokkos::RangePolicy<Exec>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                vf( i ) = static_cast<Scalar>( ( gid( i ) * 2654435761ULL ) %
                                               1009ULL );
            } );
        auto fgid = mesh.template faceSlice<FaceField::Gid>();
        auto ff = mesh.template faceSlice<userFaceField<0>()>();
        const int nf = static_cast<int>( mesh.numFaces() );
        Kokkos::parallel_for(
            "set_ffield", Kokkos::RangePolicy<Exec>( 0, nf ),
            KOKKOS_LAMBDA( const int i ) {
                ff( i ) = static_cast<Scalar>( ( fgid( i ) * 40503ULL + 7 ) %
                                               997ULL );
            } );
        Kokkos::fence();
    }

    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );

    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    const unsigned long long fcv = vertexFieldChecksum( mesh );
    const unsigned long long fcf = faceFieldChecksum( mesh );

    writeMesh( mesh, stem );
    MPI_Barrier( MPI_COMM_WORLD );

    MeshT mesh2( MPI_COMM_WORLD );
    MeshHalo<mem> halo2;
    readMesh( mesh2, halo2, stem );

    int fails = 0;
    {
        int f2 = TesseraTest::checkOwnershipPartition( mesh2, NvG, NeG, NfG );
        f2 += TesseraTest::owned1RingLocal( mesh2 );
        int g2 = 0;
        MPI_Allreduce( &f2, &g2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( g2 != 0 )
            ++fails;

        if ( TesseraTest::ownedEulerGlobal( mesh2 ) != 2 )
            ++fails;

        unsigned long long cv2, ce2, cf2;
        TesseraTest::topologyChecksum( mesh2, cv2, ce2, cf2 );
        if ( cv2 != cv || ce2 != ce || cf2 != cf )
            ++fails;

        if ( vertexFieldChecksum( mesh2 ) != fcv ||
             faceFieldChecksum( mesh2 ) != fcf )
            ++fails;
    }

    // On-disk rank-count independence: rank 0 reads the whole /*/gid
    // datasets serially and BXORs them; must equal the in-memory checksum
    // computed identically at every np.
    int diskFails = 0;
    if ( rank == 0 )
    {
        const std::string h5file = stem + ".h5";
        if ( readWholeGidChecksum( h5file, "/vertices/gid" ) != cv )
            ++diskFails;
        if ( readWholeGidChecksum( h5file, "/edges/gid" ) != ce )
            ++diskFails;
        if ( readWholeGidChecksum( h5file, "/faces/gid" ) != cf )
            ++diskFails;
    }
    int diskGlobal = 0;
    MPI_Allreduce( &diskFails, &diskGlobal, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    fails += diskGlobal;

    int haloFails = 0;
    haloFails += corrupt_sync_verify( mesh2.comm(), rank, mesh2.vertices(),
                                      mesh2.numOwnedVertices(), halo2.vplan );
    haloFails += corrupt_sync_verify( mesh2.comm(), rank, mesh2.edges(),
                                      mesh2.numOwnedEdges(), halo2.eplan );
    haloFails += corrupt_sync_verify( mesh2.comm(), rank, mesh2.faces(),
                                      mesh2.numOwnedFaces(), halo2.fplan );
    int haloGlobal = 0;
    MPI_Allreduce( &haloFails, &haloGlobal, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    fails += haloGlobal;

    if ( rank == 0 )
        std::printf( "  [%s] io round-trip %s (diskFails=%d haloFails=%d)\n",
                     tag, fails == 0 ? "ok" : "FAIL", diskGlobal, haloGlobal );

    // Clean up the written files.
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0 )
    {
        std::remove( ( stem + ".h5" ).c_str() );
        std::remove( ( stem + ".xmf" ).c_str() );
    }

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return global;
}

static std::string basenameOf( const std::string& path )
{
    const auto pos = path.find_last_of( "/\\" );
    return pos == std::string::npos ? path : path.substr( pos + 1 );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0, size = 1;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        if ( rank == 0 )
            std::printf( "test_io: parallel HDF5 + XDMF round-trip (size %d)\n",
                         size );

        // Unique stem per executable+rankcount (SERIAL/HIP exes differ,
        // np differs => unique), avoiding concurrent-run collisions.
        const std::string exeName =
            argc > 0 ? basenameOf( argv[0] ) : "test_io";
        std::string stem = exeName + "_np" + std::to_string( size );
        if ( const char* tmpdir = std::getenv( "TESSERA_IO_TMPDIR" ) )
            stem = std::string( tmpdir ) + "/" + stem;

        fails += run<Kokkos::Serial>( rank, size, "Serial", stem + "_serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( rank, size, "Default",
                                                         stem + "_default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
