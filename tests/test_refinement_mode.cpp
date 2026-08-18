/****************************************************************************
 * Copyright (c) 2024, JStewart28                                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Tessera library. Tessera is distributed under a *
 * BSD 3-Clause license. For the licensing terms see the LICENSE file in   *
 * the top-level directory.                                                 *
 ****************************************************************************/

// Unit test: RefinementMode plumbing (Task 1 of the conforming-refinement plan).
//
// The option must exist and be INERT. What this pins:
//
//   1. A RefinementMode::Conforming-typed Mesh instantiates, constructs, and
//      resizes -- the conditional face member list is a valid Cabana AoSoA.
//
//   2. FaceField::UserBegin does NOT move between modes, and userFaceField<M>()
//      is bit-identical in both. This is the single most consequential property
//      of the layout choice: the closure members are appended AFTER the user
//      pack precisely so that no existing consumer's user-field index shifts. A
//      violation here is silent (garbage face user data, not a crash), so it is
//      asserted at compile time as well as at run time.
//
//   3. The Conforming face tuple is strictly larger than the HangingNode2to1
//      one -- exactly two extra members, ClosureParent (GlobalId) and
//      ClosureParentVerts (GlobalId[3]) -- and those two are the tuple's LAST
//      members, at the indices closureParentField<>() /
//      closureParentVertsField<>() report. The vertex and edge tuples are
//      unaffected by the mode.
//
//   4. numFaceUserFields<>() counts the user pack only, whereas
//      `face_member_types::size - UserBegin` over-counts by two in Conforming
//      mode. Code iterating face user fields must use the former.
//
//   5. The closure members are real, writable storage: written on device and
//      read back on host.
//
// Runs on both the host (Serial) and the default execution space (HIP on
// Tuolumne), single rank -- this is a type-layout property, with no MPI
// decomposition or device kernel beyond a fill.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

using namespace Tessera;

// A deliberately non-trivial face user pack: a scalar and an array member, so
// the index arithmetic is exercised over more than one user field.
template <class Scalar>
using TestFaceFields = FaceFields<Scalar, Scalar[2]>;

template <class Scalar, class Exec>
int run( const char* label )
{
    using mem = typename Exec::memory_space;
    using VF = VertexFields<Scalar[2]>;
    using EF = EdgeFields<>;
    using FF = TestFaceFields<Scalar>;

    using MeshH =
        Mesh<Scalar, 3, VF, EF, FF, mem, Exec, RefinementMode::HangingNode2to1>;
    using MeshC =
        Mesh<Scalar, 3, VF, EF, FF, mem, Exec, RefinementMode::Conforming>;
    // The seven-argument spelling every existing call site uses must still name
    // a valid type after the parameter was appended. Which mode it defaults to
    // is deliberately NOT asserted here: Task 7 flips that default, and this
    // test must survive the flip.
    using MeshDefault = Mesh<Scalar, 3, VF, EF, FF, mem, Exec>;

    using tupleH = typename MeshH::face_member_types;
    using tupleC = typename MeshC::face_member_types;

    // ---- (2) UserBegin and the user-field indices do not move ---------------
    static_assert( FaceField::UserBegin == 5,
                   "FaceField::UserBegin is public API and must stay 5" );
    static_assert( userFaceField<0>() == 5 && userFaceField<1>() == 6, "" );
    static_assert(
        std::is_same<typename Cabana::MemberTypeAtIndex<userFaceField<0>(),
                                                        tupleH>::type,
                     typename Cabana::MemberTypeAtIndex<userFaceField<0>(),
                                                        tupleC>::type>::value,
        "user face field 0 must have the same type in both refinement modes" );
    static_assert(
        std::is_same<typename Cabana::MemberTypeAtIndex<userFaceField<1>(),
                                                        tupleH>::type,
                     typename Cabana::MemberTypeAtIndex<userFaceField<1>(),
                                                        tupleC>::type>::value,
        "user face field 1 must have the same type in both refinement modes" );

    // ---- (3) tuple sizes, closure member indices, and their types -----------
    static_assert(
        tupleC::size == tupleH::size + 2,
        "Conforming adds exactly ClosureParent + ClosureParentVerts" );
    static_assert( closureParentField<FF>() == tupleC::size - 2, "" );
    static_assert( closureParentVertsField<FF>() == tupleC::size - 1, "" );
    static_assert( closureParentField<FF>() == MeshC::closure_parent_field,
                   "" );
    static_assert( closureParentVertsField<FF>() ==
                       MeshC::closure_parent_verts_field,
                   "" );
    static_assert( std::is_same<typename Cabana::MemberTypeAtIndex<
                                    closureParentField<FF>(), tupleC>::type,
                                GlobalId>::value,
                   "ClosureParent is a GlobalId" );
    static_assert(
        std::is_same<typename Cabana::MemberTypeAtIndex<
                         closureParentVertsField<FF>(), tupleC>::type,
                     GlobalId[3]>::value,
        "ClosureParentVerts is a GlobalId[3]" );

    // The mode must not touch the vertex or edge member lists.
    static_assert( std::is_same<typename MeshH::vertex_member_types,
                                typename MeshC::vertex_member_types>::value,
                   "vertex layout is mode-independent" );
    static_assert( std::is_same<typename MeshH::edge_member_types,
                                typename MeshC::edge_member_types>::value,
                   "edge layout is mode-independent" );

    // ---- (4) user-field count vs tuple suffix -------------------------------
    static_assert( numFaceUserFields<FF>() == 2, "" );
    static_assert( tupleH::size - FaceField::UserBegin ==
                       numFaceUserFields<FF>(),
                   "in HangingNode2to1 the user pack IS the tuple suffix" );
    static_assert( tupleC::size - FaceField::UserBegin ==
                       numFaceUserFields<FF>() + 2,
                   "in Conforming the tuple suffix over-counts by the two "
                   "closure members -- iterate with numFaceUserFields<>()" );

    // ---- refinement_mode is exposed and self-consistent ---------------------
    static_assert( MeshH::refinement_mode == RefinementMode::HangingNode2to1,
                   "" );
    static_assert( MeshC::refinement_mode == RefinementMode::Conforming, "" );
    static_assert(
        std::is_same<typename MeshDefault::face_member_types,
                     FaceMemberTypes<FF, MeshDefault::refinement_mode>>::value,
        "the default-mode mesh's face tuple must match its own mode" );

    int fails = 0;

    // Mirror the compile-time layout facts as run-time checks so a failure is
    // reported by name rather than only as a build break.
    if ( userFaceField<0>() != 5 || userFaceField<1>() != 6 )
        ++fails;
    if ( tupleC::size != tupleH::size + 2 )
        ++fails;
    if ( numFaceUserFields<FF>() != 2 )
        ++fails;

    // ---- (1)+(5) a Conforming mesh constructs; closure storage round-trips --
    {
        MeshC mesh( MPI_COMM_WORLD );
        const std::size_t nf = 16;
        mesh.resizeFaces( nf );
        if ( mesh.numFaces() != nf )
            ++fails;

        auto fgid = mesh.template faceSlice<FaceField::Gid>();
        auto fuser0 = mesh.template faceSlice<userFaceField<0>()>();
        auto cpar = mesh.template faceSlice<MeshC::closure_parent_field>();
        auto cverts =
            mesh.template faceSlice<MeshC::closure_parent_verts_field>();
        Kokkos::parallel_for(
            "fill_closure", Kokkos::RangePolicy<Exec>( 0, nf ),
            KOKKOS_LAMBDA( const int f ) {
                fgid( f ) = static_cast<GlobalId>( 100 + f );
                fuser0( f ) = static_cast<Scalar>( f );
                // Even faces are red (no closure parent); odd faces stand in for
                // closure children of the preceding face.
                cpar( f ) = ( f % 2 == 0 )
                                ? invalid_gid
                                : static_cast<GlobalId>( 100 + f - 1 );
                for ( int k = 0; k < 3; ++k )
                    cverts( f, k ) = static_cast<GlobalId>( 10 * f + k );
            } );
        Kokkos::fence();

        Cabana::AoSoA<tupleC, Kokkos::HostSpace> hf( "hf", nf );
        Cabana::deep_copy( hf, mesh.faces() );
        auto hgid = Cabana::slice<FaceField::Gid>( hf );
        auto huser0 = Cabana::slice<userFaceField<0>()>( hf );
        auto hpar = Cabana::slice<closureParentField<FF>()>( hf );
        auto hverts = Cabana::slice<closureParentVertsField<FF>()>( hf );
        for ( std::size_t f = 0; f < nf; ++f )
        {
            if ( hgid( f ) != static_cast<GlobalId>( 100 + f ) )
                ++fails;
            if ( huser0( f ) != static_cast<Scalar>( f ) )
                ++fails;
            const GlobalId expect = ( f % 2 == 0 )
                                        ? invalid_gid
                                        : static_cast<GlobalId>( 100 + f - 1 );
            if ( hpar( f ) != expect )
                ++fails;
            for ( int k = 0; k < 3; ++k )
                if ( hverts( f, k ) != static_cast<GlobalId>( 10 * f + k ) )
                    ++fails;
        }
    }

    // ---- a HangingNode2to1 mesh is unchanged --------------------------------
    {
        MeshH mesh( MPI_COMM_WORLD );
        mesh.resizeFaces( 4 );
        auto fuser1 = mesh.template faceSlice<userFaceField<1>()>();
        Kokkos::parallel_for(
            "fill_hn", Kokkos::RangePolicy<Exec>( 0, 4 ),
            KOKKOS_LAMBDA( const int f ) {
                for ( int c = 0; c < 2; ++c )
                    fuser1( f, c ) = static_cast<Scalar>( f + c );
            } );
        Kokkos::fence();
        Cabana::AoSoA<tupleH, Kokkos::HostSpace> hf( "hf", 4 );
        Cabana::deep_copy( hf, mesh.faces() );
        auto huser1 = Cabana::slice<userFaceField<1>()>( hf );
        for ( int f = 0; f < 4; ++f )
            for ( int c = 0; c < 2; ++c )
                if ( huser1( f, c ) != static_cast<Scalar>( f + c ) )
                    ++fails;
    }

    std::printf( "  [%s] face tuple: hanging=%zu conforming=%zu "
                 "(ClosureParent@%zu, ClosureParentVerts@%zu) %s\n",
                 label, static_cast<std::size_t>( tupleH::size ),
                 static_cast<std::size_t>( tupleC::size ),
                 closureParentField<FF>(), closureParentVertsField<FF>(),
                 fails == 0 ? "ok" : "FAIL" );
    return fails;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        if ( rank == 0 )
            std::printf( "test_refinement_mode: RefinementMode plumbing "
                         "(conditional closure face members)\n" );

        fails += run<double, Kokkos::Serial>( "Serial/double" );
        fails += run<float, Kokkos::Serial>( "Serial/float" );

        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<double, Kokkos::DefaultExecutionSpace>( "Default/double" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
