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

#ifndef TESSERA_HDF5_READER_HPP
#define TESSERA_HDF5_READER_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_IoCommon.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_MeshMigrate.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_RefineClosure.hpp"
#include "Tessera_RefinementMode.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace Tessera
{

// ============================================================================
// readMesh — round-trip reader (Step 8), reconstructing via migrate()
// ============================================================================
//
// Reconstructs a valid distributed Mesh + MeshHalo from <stem>.h5 without
// re-deriving ownership/halo logic (spec 8.5): each rank takes a FRESH
// contiguous block of dense FACE indices (deliberately unrelated to the
// writer's partition -- this exercises rank-count independence), fetches the
// vertex/edge records its block's faces reference (translating dense indices
// back to persistent gids), assembles an intermediate ALL-OWNED mesh from that
// covering, then hands it to the tested `migrate()` with a self-destination
// `dest` -- migrate() recomputes lowest-rank ownership, discovers/fetches the
// 1-deep ghost layer, and rebuilds the CSR + key tables + halo plans. The
// postcondition is a valid distributed mesh identical in structure to the
// Step-5/7 output, so every Step-5 invariant passes by construction.
//
// CONFORMING MODE. Two things the hanging-node path does not need:
//
//   * the closure bookkeeping members are read from /faces/closure_parent and
//     /faces/closure_parent_verts (persistent gids, no dense translation) after
//     initClosureFaceMembers() has stamped the whole freshly-materialized block
//     — a Cabana AoSoA's backing View is zero-initialized, so an untouched
//     ClosureParent reads as face gid 0 and unclose() would restore a bogus red
//     parent for every face;
//   * the fresh dense-index block partition cuts wherever it likes, so it
//     routinely SPLITS a closure sibling group across ranks. migrate()'s round
//     S repairs only groups that are already co-resident, which is exactly the
//     assumption a read-back breaks, so the collective
//     repairClosureCohesion() runs first and builds a `dest` that puts each
//     group back on one rank.
template <class MeshT>
void readMesh( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
               const std::string& stem )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_READ_MESH );
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;
    using HostV = Cabana::AoSoA<VMT, Kokkos::HostSpace>;
    using HostE = Cabana::AoSoA<EMT, Kokkos::HostSpace>;
    using HostF = Cabana::AoSoA<FMT, Kokkos::HostSpace>;
    using VTuple = typename HostV::tuple_type;
    using ETuple = typename HostE::tuple_type;

    MPI_Comm comm = mesh.comm();
    const int R = mesh.rank();
    const int size = mesh.commSize();

    hid_t fapl = H5Pcreate( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio( fapl, comm, MPI_INFO_NULL );
    const std::string filename = stem + ".h5";
    hid_t file = H5Fopen( filename.c_str(), H5F_ACC_RDONLY, fapl );
    H5Pclose( fapl );

    // ---- validate root attrs against the compile-time template ------------
    constexpr bool kConforming =
        ( MeshT::refinement_mode == RefinementMode::Conforming );

    detail::abortOnMismatch( comm,
                             detail::readIntAttr( file, "format_version" ) == 2,
                             "format_version" );
    // Which mode wrote the file is a hard mismatch, not a fallback: a
    // conforming mesh read from a hanging-node file would have no closure
    // bookkeeping to un-close, and a hanging-node mesh cannot carry a
    // conforming file's closure layer at all.
    detail::abortOnMismatch( comm,
                             detail::readIntAttr( file, "refinement_mode" ) ==
                                 ( kConforming ? 1 : 0 ),
                             "refinement_mode" );
    detail::abortOnMismatch( comm, detail::readIntAttr( file, "dim" ) == Dim,
                             "dim" );
    detail::abortOnMismatch( comm,
                             detail::readIntAttr( file, "scalar_bytes" ) ==
                                 static_cast<int>( sizeof( Scalar ) ),
                             "scalar_bytes" );

    constexpr std::size_t nUserV =
        detail::userFieldCount<HostV, VertexField::UserBegin>();
    constexpr std::size_t nUserE =
        detail::userFieldCount<HostE, EdgeField::UserBegin>();
    // NOT the face tuple's suffix in Conforming mode — see the writer.
    constexpr std::size_t nUserF =
        numFaceUserFields<typename MeshT::face_user_fields>();
    detail::abortOnMismatch( comm,
                             detail::readIntAttr( file, "n_user_v_fields" ) ==
                                 static_cast<int>( nUserV ),
                             "n_user_v_fields" );
    detail::abortOnMismatch( comm,
                             detail::readIntAttr( file, "n_user_e_fields" ) ==
                                 static_cast<int>( nUserE ),
                             "n_user_e_fields" );
    detail::abortOnMismatch( comm,
                             detail::readIntAttr( file, "n_user_f_fields" ) ==
                                 static_cast<int>( nUserF ),
                             "n_user_f_fields" );

    bool extentsOk = true;
    detail::forEachUserField<VertexField::UserBegin, HostV>(
        [&]( auto MabsIc )
        {
            constexpr std::size_t Mabs = decltype( MabsIc )::value;
            constexpr std::size_t j = Mabs - VertexField::UserBegin;
            constexpr int E = detail::FieldInfo<HostV, Mabs>::extent;
            if ( detail::readIntAttr( file, "uv_ext_" + std::to_string( j ) ) !=
                 E )
                extentsOk = false;
        } );
    detail::forEachUserField<EdgeField::UserBegin, HostE>(
        [&]( auto MabsIc )
        {
            constexpr std::size_t Mabs = decltype( MabsIc )::value;
            constexpr std::size_t j = Mabs - EdgeField::UserBegin;
            constexpr int E = detail::FieldInfo<HostE, Mabs>::extent;
            if ( detail::readIntAttr( file, "ue_ext_" + std::to_string( j ) ) !=
                 E )
                extentsOk = false;
        } );
    detail::forEachUserFieldN<FaceField::UserBegin, nUserF, HostF>(
        [&]( auto MabsIc )
        {
            constexpr std::size_t Mabs = decltype( MabsIc )::value;
            constexpr std::size_t j = Mabs - FaceField::UserBegin;
            constexpr int E = detail::FieldInfo<HostF, Mabs>::extent;
            if ( detail::readIntAttr( file, "uf_ext_" + std::to_string( j ) ) !=
                 E )
                extentsOk = false;
        } );
    detail::abortOnMismatch( comm, extentsOk, "user field extent" );

    const long long Nv =
        static_cast<long long>( detail::readU64Attr( file, "Nv" ) );
    const long long Ne =
        static_cast<long long>( detail::readU64Attr( file, "Ne" ) );
    const long long Nf =
        static_cast<long long>( detail::readU64Attr( file, "Nf" ) );

    hid_t gVerts = H5Gopen2( file, "/vertices", H5P_DEFAULT );
    hid_t gEdges = H5Gopen2( file, "/edges", H5P_DEFAULT );
    hid_t gFaces = H5Gopen2( file, "/faces", H5P_DEFAULT );

    // ---- this rank's fresh dense FACE block (spec 8.5 step 2) --------------
    long long fs, fe;
    detail::blockRange( Nf, R, size, fs, fe );
    const long long blockFaces = fe - fs;

    std::vector<std::uint64_t> fVertsDense(
        static_cast<std::size_t>( blockFaces ) * 3 ),
        fEdgesDense( static_cast<std::size_t>( blockFaces ) * 3 );
    HostF hf( "hf_io", static_cast<std::size_t>( blockFaces ) );
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_READ_BLOCKS );
        std::vector<std::uint64_t> gidBuf( blockFaces );
        std::vector<Level> levelBuf( blockFaces );
        {
            TESSERA_SCOPED_TIMER_VERBOSE(
                ::Tessera::Profiling::TIMER_READ_HYPERSLAB );
            detail::readHyperslab( gFaces, "gid", 1, fs, blockFaces,
                                   gidBuf.data() );
            detail::readHyperslab( gFaces, "verts", 3, fs, blockFaces,
                                   fVertsDense.data() );
            detail::readHyperslab( gFaces, "edges", 3, fs, blockFaces,
                                   fEdgesDense.data() );
            detail::readHyperslab( gFaces, "level", 1, fs, blockFaces,
                                   levelBuf.data() );
        }
        auto gid = Cabana::slice<FaceField::Gid>( hf );
        auto own = Cabana::slice<FaceField::Owner>( hf );
        auto lev = Cabana::slice<FaceField::Level>( hf );
        for ( long long i = 0; i < blockFaces; ++i )
        {
            const int li = static_cast<int>( i );
            gid( li ) = gidBuf[i];
            own( li ) = static_cast<Rank>( R );
            lev( li ) = levelBuf[i];
        }
        detail::forEachUserFieldN<FaceField::UserBegin, nUserF, HostF>(
            [&]( auto MabsIc )
            {
                constexpr std::size_t Mabs = decltype( MabsIc )::value;
                constexpr std::size_t j = Mabs - FaceField::UserBegin;
                using FI = detail::FieldInfo<HostF, Mabs>;
                using ST = typename FI::scalar_type;
                constexpr int E = FI::extent;
                std::vector<ST> buf( static_cast<std::size_t>( blockFaces ) *
                                     E );
                detail::readHyperslab( gFaces, "u" + std::to_string( j ), E, fs,
                                       blockFaces, buf.data() );
                auto s = Cabana::slice<Mabs>( hf );
                for ( long long i = 0; i < blockFaces; ++i )
                {
                    const int li = static_cast<int>( i );
                    if constexpr ( E == 1 )
                        s( li ) = buf[i];
                    else
                        for ( int c = 0; c < E; ++c )
                            s( li, c ) = buf[i * E + c];
                }
            } );

        // ---- closure bookkeeping (Conforming mode only) --------------------
        // Stamp the whole block first: this is a freshly-materialized AoSoA, so
        // without it an unwritten ClosureParent reads as the perfectly valid
        // face gid 0 (see initClosureFaceMembers()). The datasets then overwrite
        // every row, but the stamp is what keeps a future partial read honest.
        initClosureFaceMembers<MeshT>( hf, 0,
                                       static_cast<std::size_t>( blockFaces ) );
        if constexpr ( kConforming )
        {
            std::vector<std::uint64_t> parentBuf( blockFaces ),
                parentVertsBuf( static_cast<std::size_t>( blockFaces ) * 3 );
            detail::readHyperslab( gFaces, "closure_parent", 1, fs, blockFaces,
                                   parentBuf.data() );
            detail::readHyperslab( gFaces, "closure_parent_verts", 3, fs,
                                   blockFaces, parentVertsBuf.data() );
            auto cp = Cabana::slice<MeshT::closure_parent_field>( hf );
            auto cv = Cabana::slice<MeshT::closure_parent_verts_field>( hf );
            for ( long long i = 0; i < blockFaces; ++i )
            {
                const int li = static_cast<int>( i );
                cp( li ) = parentBuf[i];
                for ( int k = 0; k < 3; ++k )
                    cv( li, k ) = parentVertsBuf[i * 3 + k];
            }
        }
    }

    // Dense vertex/edge indices this rank's face block references.
    std::set<std::uint64_t> refV( fVertsDense.begin(), fVertsDense.end() );
    std::set<std::uint64_t> refE( fEdgesDense.begin(), fEdgesDense.end() );

    // ---- own vertex/edge block, read up front (spec 8.5 step 3) -----------
    long long vbs, vbe;
    detail::blockRange( Nv, R, size, vbs, vbe );
    const long long vBlockN = vbe - vbs;
    HostV blockV( "blockV_io", static_cast<std::size_t>( vBlockN ) );
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_READ_BLOCKS );
        std::vector<std::uint64_t> gidBuf( vBlockN );
        std::vector<Scalar> posBuf( static_cast<std::size_t>( vBlockN ) * Dim );
        detail::readHyperslab( gVerts, "gid", 1, vbs, vBlockN, gidBuf.data() );
        detail::readHyperslab( gVerts, "position", Dim, vbs, vBlockN,
                               posBuf.data() );
        auto gid = Cabana::slice<VertexField::Gid>( blockV );
        auto own = Cabana::slice<VertexField::Owner>( blockV );
        auto flg = Cabana::slice<VertexField::Flags>( blockV );
        auto pos = Cabana::slice<VertexField::Position>( blockV );
        for ( long long i = 0; i < vBlockN; ++i )
        {
            const int li = static_cast<int>( i );
            gid( li ) = gidBuf[i];
            own( li ) = static_cast<Rank>( R );
            flg( li ) = 0;
            for ( int d = 0; d < Dim; ++d )
                pos( li, d ) = posBuf[i * Dim + d];
        }
        detail::forEachUserField<VertexField::UserBegin, HostV>(
            [&]( auto MabsIc )
            {
                constexpr std::size_t Mabs = decltype( MabsIc )::value;
                constexpr std::size_t j = Mabs - VertexField::UserBegin;
                using FI = detail::FieldInfo<HostV, Mabs>;
                using ST = typename FI::scalar_type;
                constexpr int E = FI::extent;
                std::vector<ST> buf( static_cast<std::size_t>( vBlockN ) * E );
                detail::readHyperslab( gVerts, "u" + std::to_string( j ), E,
                                       vbs, vBlockN, buf.data() );
                auto s = Cabana::slice<Mabs>( blockV );
                for ( long long i = 0; i < vBlockN; ++i )
                {
                    const int li = static_cast<int>( i );
                    if constexpr ( E == 1 )
                        s( li ) = buf[i];
                    else
                        for ( int c = 0; c < E; ++c )
                            s( li, c ) = buf[i * E + c];
                }
            } );
    }

    long long ebs, ebe;
    detail::blockRange( Ne, R, size, ebs, ebe );
    const long long eBlockN = ebe - ebs;
    HostE blockE( "blockE_io", static_cast<std::size_t>( eBlockN ) );
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_READ_BLOCKS );
        std::vector<std::uint64_t> gidBuf( eBlockN ), vertsBuf( eBlockN * 2 );
        std::vector<Level> levelBuf( eBlockN );
        detail::readHyperslab( gEdges, "gid", 1, ebs, eBlockN, gidBuf.data() );
        detail::readHyperslab( gEdges, "verts", 2, ebs, eBlockN,
                               vertsBuf.data() );
        detail::readHyperslab( gEdges, "level", 1, ebs, eBlockN,
                               levelBuf.data() );
        auto gid = Cabana::slice<EdgeField::Gid>( blockE );
        auto verts = Cabana::slice<EdgeField::Verts>( blockE );
        auto lev = Cabana::slice<EdgeField::Level>( blockE );
        for ( long long i = 0; i < eBlockN; ++i )
        {
            const int li = static_cast<int>( i );
            gid( li ) = gidBuf[i];
            // Verts temporarily holds DENSE vertex indices (translated to
            // persistent gids once denseV->gid is known, below); Faces/Owner
            // are set during that same translation pass.
            verts( li, 0 ) = vertsBuf[i * 2 + 0];
            verts( li, 1 ) = vertsBuf[i * 2 + 1];
            lev( li ) = levelBuf[i];
        }
        detail::forEachUserField<EdgeField::UserBegin, HostE>(
            [&]( auto MabsIc )
            {
                constexpr std::size_t Mabs = decltype( MabsIc )::value;
                constexpr std::size_t j = Mabs - EdgeField::UserBegin;
                using FI = detail::FieldInfo<HostE, Mabs>;
                using ST = typename FI::scalar_type;
                constexpr int E = FI::extent;
                std::vector<ST> buf( static_cast<std::size_t>( eBlockN ) * E );
                detail::readHyperslab( gEdges, "u" + std::to_string( j ), E,
                                       ebs, eBlockN, buf.data() );
                auto s = Cabana::slice<Mabs>( blockE );
                for ( long long i = 0; i < eBlockN; ++i )
                {
                    const int li = static_cast<int>( i );
                    if constexpr ( E == 1 )
                        s( li ) = buf[i];
                    else
                        for ( int c = 0; c < E; ++c )
                            s( li, c ) = buf[i * E + c];
                }
            } );
    }

    // ---- fetch referenced records outside this rank's own block -----------
    auto fetchRecords = [&]( const std::set<std::uint64_t>& refSet,
                             long long blockBegin, long long blockEnd,
                             long long N, auto& blockAoSoA )
    {
        using Tup = typename std::decay_t<decltype( blockAoSoA )>::tuple_type;
        std::vector<std::vector<std::uint64_t>> req( size );
        std::vector<std::uint64_t> localIdx;
        for ( auto idx : refSet )
        {
            if ( idx >= static_cast<std::uint64_t>( blockBegin ) &&
                 idx < static_cast<std::uint64_t>( blockEnd ) )
                localIdx.push_back( idx );
            else
                req[detail::blockOwner( static_cast<long long>( idx ), N,
                                        size )]
                    .push_back( idx );
        }
        auto got = allToAllV( comm, req );
        std::vector<std::vector<detail::TupleBlob<Tup>>> rep( size );
        for ( int s = 0; s < size; ++s )
        {
            const std::uint64_t* p = got.from( s );
            const int c = got.count( s );
            for ( int k = 0; k < c; ++k )
            {
                const int li = static_cast<int>( p[k] - blockBegin );
                rep[s].push_back( detail::toBlob( blockAoSoA.getTuple( li ) ) );
            }
        }
        auto repGot = allToAllV( comm, rep );

        std::map<std::uint64_t, Tup> recs;
        for ( auto idx : localIdx )
            recs[idx] =
                blockAoSoA.getTuple( static_cast<int>( idx - blockBegin ) );
        for ( int s = 0; s < size; ++s )
        {
            const int c = repGot.count( s );
            const auto* p = repGot.from( s );
            for ( int k = 0; k < c; ++k )
                recs[req[s][k]] = detail::fromBlob( p[k] );
        }
        return recs;
    };

    std::map<std::uint64_t, VTuple> vRec =
        fetchRecords( refV, vbs, vbe, Nv, blockV );
    std::map<std::uint64_t, ETuple> eRec =
        fetchRecords( refE, ebs, ebe, Ne, blockE );

    // denseV/denseE -> persistent gid, from the fetched/owned records.
    std::map<std::uint64_t, GlobalId> denseVGid, denseEGid;
    for ( const auto& kv : vRec )
        denseVGid[kv.first] = Cabana::get<VertexField::Gid>( kv.second );
    for ( const auto& kv : eRec )
        denseEGid[kv.first] = Cabana::get<EdgeField::Gid>( kv.second );

    // ---- assemble the intermediate ALL-OWNED mesh (spec 8.5 step 4) -------
    const int nHeldV = static_cast<int>( vRec.size() );
    HostV finalV( "finalV_io", nHeldV );
    {
        int li = 0;
        for ( const auto& kv : vRec )
            finalV.setTuple( li++, kv.second );
    }
    mesh.resizeVertices( nHeldV );
    Cabana::deep_copy( mesh.vertices(), finalV );

    const int nHeldE = static_cast<int>( eRec.size() );
    HostE finalE( "finalE_io", nHeldE );
    {
        int li = 0;
        for ( const auto& kv : eRec )
        {
            ETuple t = kv.second;
            const std::uint64_t dv0 = Cabana::get<EdgeField::Verts>( t, 0 );
            const std::uint64_t dv1 = Cabana::get<EdgeField::Verts>( t, 1 );
            Cabana::get<EdgeField::Verts>( t, 0 ) = denseVGid.at( dv0 );
            Cabana::get<EdgeField::Verts>( t, 1 ) = denseVGid.at( dv1 );
            Cabana::get<EdgeField::Faces>( t, 0 ) = invalid_gid;
            Cabana::get<EdgeField::Faces>( t, 1 ) = invalid_gid;
            Cabana::get<EdgeField::Owner>( t ) = static_cast<Rank>( R );
            finalE.setTuple( li++, t );
        }
    }
    mesh.resizeEdges( nHeldE );
    Cabana::deep_copy( mesh.edges(), finalE );

    {
        auto verts = Cabana::slice<FaceField::Verts>( hf );
        auto edges = Cabana::slice<FaceField::Edges>( hf );
        for ( long long i = 0; i < blockFaces; ++i )
        {
            const int li = static_cast<int>( i );
            for ( int k = 0; k < 3; ++k )
            {
                verts( li, k ) = denseVGid.at( fVertsDense[i * 3 + k] );
                edges( li, k ) = denseEGid.at( fEdgesDense[i * 3 + k] );
            }
        }
    }
    mesh.resizeFaces( static_cast<int>( blockFaces ) );
    Cabana::deep_copy( mesh.faces(), hf );
    mesh.setOwnedCounts( static_cast<std::size_t>( nHeldV ),
                         static_cast<std::size_t>( nHeldE ),
                         static_cast<std::size_t>( blockFaces ) );

    H5Gclose( gVerts );
    H5Gclose( gEdges );
    H5Gclose( gFaces );
    H5Fclose( file );

    // ---- hand off to migrate(): recomputes ownership, ghosts, halo --------
    // `dest` is self everywhere (the dense block partition IS the target
    // partition), except where repairClosureCohesion() has to pull a sibling
    // group the block boundary split back onto one rank.
    std::vector<Rank> dest( static_cast<std::size_t>( blockFaces ),
                            static_cast<Rank>( R ) );
    repairClosureCohesion( mesh, dest );
    migrate( mesh, halo, dest );
}

} // namespace Tessera

#endif // TESSERA_HDF5_READER_HPP
