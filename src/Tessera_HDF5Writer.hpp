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

#ifndef TESSERA_HDF5_WRITER_HPP
#define TESSERA_HDF5_WRITER_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_IoCommon.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_RefinementMode.hpp"
#include "Tessera_Types.hpp"
#include "Tessera_Xdmf.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace Tessera
{

// ============================================================================
// writeMesh — collective parallel HDF5 + XDMF sidecar writer (Step 8).
// ============================================================================
//
// Writes the OWNED entities of every rank, each exactly once, into a clean
// partition of <stem>.h5 (spec 8.0-8.4): dense global vertex/edge indices are
// assigned via MPI_Exscan over owned-only counts (8.2), connectivity
// (edge/face verts, face edges) is translated to those dense indices, and a
// persistent 64-bit gid is carried verbatim alongside for every entity. Ghost
// entities referenced by an owned face/edge but owned by another rank have
// their dense index fetched from that owner via two allToAllV rounds (the
// halo plan is not needed -- each entity's `Owner` field is enough).
// Collective under MPI-IO (spec 8.3); needs only the mesh (no halo argument).
//
// FORMAT VERSION 2 — refinement mode on disk.
// A RefinementMode::Conforming mesh's visible faces carry two extra closure
// bookkeeping members, and a file that dropped them could not be un-closed
// after a read-back, so the red layer — and with it every subsequent refine()
// — would be lost. They are written as their own datasets
// /faces/closure_parent (u64) and /faces/closure_parent_verts (u64 x 3), NOT as
// user fields:
//
//   * the two members are appended AFTER the face user pack, so the pack is no
//     longer the face tuple's suffix. The face user-field loop is therefore
//     bounded by numFaceUserFields<>() rather than by the tuple size — writing
//     them as "u<n>"/"u<n+1>" would round-trip by accident but would also put
//     them in `n_user_f_fields` and in the XDMF attribute list;
//   * they hold PERSISTENT gids (a retired red face gid and three vertex gids),
//     not dense indices, so unlike /edges/verts and /faces/verts they need no
//     dense translation and no ghost fetch. A parent's corners are always
//     corners of the parent's own children, hence of faces this rank holds.
//
// The root attribute `refinement_mode` (0 = HangingNode2to1, 1 = Conforming)
// records which shape the file has; the reader validates it against the mesh
// type it was handed, so reading a hanging-node file into a conforming mesh (or
// vice versa) aborts with a named mismatch instead of silently producing a mesh
// with no closure bookkeeping. It is written in BOTH modes — the version bump
// from 1 to 2 is what makes that safe, since a v1 file carries no such
// attribute and the reader would fail to open it.
template <class MeshT>
void writeMesh( const MeshT& mesh, const std::string& stem )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_WRITE_MESH );
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    using VMT = typename MeshT::vertex_member_types;
    using EMT = typename MeshT::edge_member_types;
    using FMT = typename MeshT::face_member_types;
    using HostV = Cabana::AoSoA<VMT, Kokkos::HostSpace>;
    using HostE = Cabana::AoSoA<EMT, Kokkos::HostSpace>;
    using HostF = Cabana::AoSoA<FMT, Kokkos::HostSpace>;

    constexpr bool kConforming =
        ( MeshT::refinement_mode == RefinementMode::Conforming );
    // The face user pack is NOT the face tuple's suffix in Conforming mode, so
    // every face user-field loop below is bounded by this, not by the tuple.
    constexpr std::size_t nUserF =
        numFaceUserFields<typename MeshT::face_user_fields>();

    MPI_Comm comm = mesh.comm();
    const int R = mesh.rank();
    const int size = mesh.commSize();

    const long long nOwnedV = static_cast<long long>( mesh.numOwnedVertices() );
    const long long nOwnedE = static_cast<long long>( mesh.numOwnedEdges() );
    const long long nOwnedF = static_cast<long long>( mesh.numOwnedFaces() );

    const auto gcV = detail::exscanCount( comm, nOwnedV );
    const auto gcE = detail::exscanCount( comm, nOwnedE );
    const auto gcF = detail::exscanCount( comm, nOwnedF );

    // ---- host copies of everything currently held (owned + ghost) ---------
    const int nv = static_cast<int>( mesh.numVertices() );
    const int ne = static_cast<int>( mesh.numEdges() );
    const int nf = static_cast<int>( mesh.numFaces() );
    HostV hv( "hv_io", nv );
    HostE he( "he_io", ne );
    HostF hf( "hf_io", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto v_owner = Cabana::slice<VertexField::Owner>( hv );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );
    auto e_gid = Cabana::slice<EdgeField::Gid>( he );
    auto e_owner = Cabana::slice<EdgeField::Owner>( he );
    auto e_verts = Cabana::slice<EdgeField::Verts>( he );
    auto e_level = Cabana::slice<EdgeField::Level>( he );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_edges = Cabana::slice<FaceField::Edges>( hf );
    auto f_level = Cabana::slice<FaceField::Level>( hf );

    // ---- dense numbering: owned entries are dense directly (8.2 step 2) ---
    std::map<GlobalId, std::uint64_t> denseV, denseE;
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_WRITE_DENSE_NUMBER );
        for ( long long i = 0; i < nOwnedV; ++i )
            denseV[v_gid( static_cast<int>( i ) )] =
                static_cast<std::uint64_t>( gcV.off + i );
        for ( long long i = 0; i < nOwnedE; ++i )
            denseE[e_gid( static_cast<int>( i ) )] =
                static_cast<std::uint64_t>( gcE.off + i );
    }

    // ---- ghost dense-index fetch (8.2 step 3) ------------------------------
    auto fetchGhostDense = [&]( int n_total, int n_owned, auto gidSlice,
                                auto ownerSlice,
                                std::map<GlobalId, std::uint64_t>& dense )
    {
        std::vector<std::vector<GlobalId>> req( size );
        for ( int i = n_owned; i < n_total; ++i )
            req[ownerSlice( i )].push_back( gidSlice( i ) );
        auto got = allToAllV( comm, req );

        // Each owner replies the dense index for every gid it received, in
        // received order (matches the requester's per-source ordering).
        std::vector<std::vector<std::uint64_t>> reply( size );
        for ( int s = 0; s < size; ++s )
        {
            const GlobalId* p = got.from( s );
            const int c = got.count( s );
            for ( int k = 0; k < c; ++k )
                reply[s].push_back( dense.at( p[k] ) );
        }
        auto repGot = allToAllV( comm, reply );
        for ( int s = 0; s < size; ++s )
        {
            const std::uint64_t* p = repGot.from( s );
            const int c = repGot.count( s );
            for ( int k = 0; k < c; ++k )
                dense[req[s][k]] = p[k];
        }
    };
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_WRITE_GHOST_FETCH );
        fetchGhostDense( nv, static_cast<int>( nOwnedV ), v_gid, v_owner,
                         denseV );
        fetchGhostDense( ne, static_cast<int>( nOwnedE ), e_gid, e_owner,
                         denseE );
    }

    // ---- open file (collective, MPI-IO) ------------------------------------
    hid_t fapl = H5Pcreate( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio( fapl, comm, MPI_INFO_NULL );
    const std::string filename = stem + ".h5";
    hid_t file =
        H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl );
    H5Pclose( fapl );

    hid_t gVerts =
        H5Gcreate2( file, "/vertices", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hid_t gEdges =
        H5Gcreate2( file, "/edges", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    hid_t gFaces =
        H5Gcreate2( file, "/faces", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    std::vector<detail::XdmfField> vXdmf, fXdmf;

    // ---- /vertices ----------------------------------------------------------
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_WRITE_DATASETS );
        std::vector<std::uint64_t> gidBuf( nOwnedV );
        std::vector<Scalar> posBuf( static_cast<std::size_t>( nOwnedV ) * Dim );
        for ( long long i = 0; i < nOwnedV; ++i )
        {
            const int li = static_cast<int>( i );
            gidBuf[i] = v_gid( li );
            for ( int d = 0; d < Dim; ++d )
                posBuf[i * Dim + d] = v_pos( li, d );
        }
        {
            TESSERA_SCOPED_TIMER_VERBOSE(
                ::Tessera::Profiling::TIMER_WRITE_HYPERSLAB );
            detail::writeHyperslab( gVerts, "gid", gcV.N, 1, gcV.off, nOwnedV,
                                    gidBuf.data() );
            detail::writeHyperslab( gVerts, "position", gcV.N, Dim, gcV.off,
                                    nOwnedV, posBuf.data() );
        }

        detail::forEachUserField<VertexField::UserBegin, HostV>(
            [&]( auto MabsIc )
            {
                constexpr std::size_t Mabs = decltype( MabsIc )::value;
                constexpr std::size_t j = Mabs - VertexField::UserBegin;
                using FI = detail::FieldInfo<HostV, Mabs>;
                using ST = typename FI::scalar_type;
                constexpr int E = FI::extent;
                auto s = Cabana::slice<Mabs>( hv );
                std::vector<ST> buf( static_cast<std::size_t>( nOwnedV ) * E );
                for ( long long i = 0; i < nOwnedV; ++i )
                {
                    const int li = static_cast<int>( i );
                    if constexpr ( E == 1 )
                        buf[i] = s( li );
                    else
                        for ( int c = 0; c < E; ++c )
                            buf[i * E + c] = s( li, c );
                }
                const std::string name = "u" + std::to_string( j );
                detail::writeHyperslab( gVerts, name, gcV.N, E, gcV.off,
                                        nOwnedV, buf.data() );
                detail::writeIntAttr( file, "uv_ext_" + std::to_string( j ),
                                      E );
                vXdmf.push_back( { name, "v" + name, E } );
            } );
    }

    // ---- /edges ---------------------------------------------------------
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_WRITE_DATASETS );
        std::vector<std::uint64_t> gidBuf( nOwnedE ), vertsBuf( nOwnedE * 2 );
        std::vector<Level> levelBuf( nOwnedE );
        for ( long long i = 0; i < nOwnedE; ++i )
        {
            const int li = static_cast<int>( i );
            gidBuf[i] = e_gid( li );
            vertsBuf[i * 2 + 0] = denseV.at( e_verts( li, 0 ) );
            vertsBuf[i * 2 + 1] = denseV.at( e_verts( li, 1 ) );
            levelBuf[i] = e_level( li );
        }
        {
            TESSERA_SCOPED_TIMER_VERBOSE(
                ::Tessera::Profiling::TIMER_WRITE_HYPERSLAB );
            detail::writeHyperslab( gEdges, "gid", gcE.N, 1, gcE.off, nOwnedE,
                                    gidBuf.data() );
            detail::writeHyperslab( gEdges, "verts", gcE.N, 2, gcE.off, nOwnedE,
                                    vertsBuf.data() );
            detail::writeHyperslab( gEdges, "level", gcE.N, 1, gcE.off, nOwnedE,
                                    levelBuf.data() );
        }

        detail::forEachUserField<EdgeField::UserBegin, HostE>(
            [&]( auto MabsIc )
            {
                constexpr std::size_t Mabs = decltype( MabsIc )::value;
                constexpr std::size_t j = Mabs - EdgeField::UserBegin;
                using FI = detail::FieldInfo<HostE, Mabs>;
                using ST = typename FI::scalar_type;
                constexpr int E = FI::extent;
                auto s = Cabana::slice<Mabs>( he );
                std::vector<ST> buf( static_cast<std::size_t>( nOwnedE ) * E );
                for ( long long i = 0; i < nOwnedE; ++i )
                {
                    const int li = static_cast<int>( i );
                    if constexpr ( E == 1 )
                        buf[i] = s( li );
                    else
                        for ( int c = 0; c < E; ++c )
                            buf[i * E + c] = s( li, c );
                }
                detail::writeHyperslab( gEdges, "u" + std::to_string( j ),
                                        gcE.N, E, gcE.off, nOwnedE,
                                        buf.data() );
                detail::writeIntAttr( file, "ue_ext_" + std::to_string( j ),
                                      E );
            } );
    }

    // ---- /faces ---------------------------------------------------------
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_WRITE_DATASETS );
        std::vector<std::uint64_t> gidBuf( nOwnedF ), vertsBuf( nOwnedF * 3 ),
            edgesBuf( nOwnedF * 3 );
        std::vector<Level> levelBuf( nOwnedF );
        for ( long long i = 0; i < nOwnedF; ++i )
        {
            const int li = static_cast<int>( i );
            gidBuf[i] = f_gid( li );
            for ( int k = 0; k < 3; ++k )
            {
                vertsBuf[i * 3 + k] = denseV.at( f_verts( li, k ) );
                edgesBuf[i * 3 + k] = denseE.at( f_edges( li, k ) );
            }
            levelBuf[i] = f_level( li );
        }
        {
            TESSERA_SCOPED_TIMER_VERBOSE(
                ::Tessera::Profiling::TIMER_WRITE_HYPERSLAB );
            detail::writeHyperslab( gFaces, "gid", gcF.N, 1, gcF.off, nOwnedF,
                                    gidBuf.data() );
            detail::writeHyperslab( gFaces, "verts", gcF.N, 3, gcF.off, nOwnedF,
                                    vertsBuf.data() );
            detail::writeHyperslab( gFaces, "edges", gcF.N, 3, gcF.off, nOwnedF,
                                    edgesBuf.data() );
            detail::writeHyperslab( gFaces, "level", gcF.N, 1, gcF.off, nOwnedF,
                                    levelBuf.data() );
        }

        detail::forEachUserFieldN<FaceField::UserBegin, nUserF, HostF>(
            [&]( auto MabsIc )
            {
                constexpr std::size_t Mabs = decltype( MabsIc )::value;
                constexpr std::size_t j = Mabs - FaceField::UserBegin;
                using FI = detail::FieldInfo<HostF, Mabs>;
                using ST = typename FI::scalar_type;
                constexpr int E = FI::extent;
                auto s = Cabana::slice<Mabs>( hf );
                std::vector<ST> buf( static_cast<std::size_t>( nOwnedF ) * E );
                for ( long long i = 0; i < nOwnedF; ++i )
                {
                    const int li = static_cast<int>( i );
                    if constexpr ( E == 1 )
                        buf[i] = s( li );
                    else
                        for ( int c = 0; c < E; ++c )
                            buf[i * E + c] = s( li, c );
                }
                const std::string name = "u" + std::to_string( j );
                detail::writeHyperslab( gFaces, name, gcF.N, E, gcF.off,
                                        nOwnedF, buf.data() );
                detail::writeIntAttr( file, "uf_ext_" + std::to_string( j ),
                                      E );
                fXdmf.push_back( { name, "f" + name, E } );
            } );

        // ---- closure bookkeeping (Conforming mode only) -------------------
        // Persistent gids throughout, so no dense translation: ClosureParent is
        // a retired red FACE gid (it names no live entity in the file at all —
        // that is the point of retiring it) and ClosureParentVerts holds three
        // VERTEX gids. invalid_gid marks a passed-through red face.
        if constexpr ( kConforming )
        {
            auto cp = Cabana::slice<MeshT::closure_parent_field>( hf );
            auto cv = Cabana::slice<MeshT::closure_parent_verts_field>( hf );
            std::vector<std::uint64_t> parentBuf( nOwnedF ),
                parentVertsBuf( static_cast<std::size_t>( nOwnedF ) * 3 );
            for ( long long i = 0; i < nOwnedF; ++i )
            {
                const int li = static_cast<int>( i );
                parentBuf[i] = cp( li );
                for ( int k = 0; k < 3; ++k )
                    parentVertsBuf[i * 3 + k] = cv( li, k );
            }
            detail::writeHyperslab( gFaces, "closure_parent", gcF.N, 1, gcF.off,
                                    nOwnedF, parentBuf.data() );
            detail::writeHyperslab( gFaces, "closure_parent_verts", gcF.N, 3,
                                    gcF.off, nOwnedF, parentVertsBuf.data() );
        }
    }

    // ---- root attributes (identical on every rank; safe under MPIO) -------
    detail::writeIntAttr( file, "format_version", 2 );
    detail::writeIntAttr( file, "refinement_mode", kConforming ? 1 : 0 );
    detail::writeIntAttr( file, "dim", Dim );
    detail::writeIntAttr( file, "scalar_bytes",
                          static_cast<int>( sizeof( Scalar ) ) );
    detail::writeU64Attr( file, "Nv", static_cast<std::uint64_t>( gcV.N ) );
    detail::writeU64Attr( file, "Ne", static_cast<std::uint64_t>( gcE.N ) );
    detail::writeU64Attr( file, "Nf", static_cast<std::uint64_t>( gcF.N ) );
    detail::writeIntAttr(
        file, "n_user_v_fields",
        static_cast<int>(
            detail::userFieldCount<HostV, VertexField::UserBegin>() ) );
    detail::writeIntAttr(
        file, "n_user_e_fields",
        static_cast<int>(
            detail::userFieldCount<HostE, EdgeField::UserBegin>() ) );
    detail::writeIntAttr( file, "n_user_f_fields", static_cast<int>( nUserF ) );

    H5Gclose( gVerts );
    H5Gclose( gEdges );
    H5Gclose( gFaces );
    H5Fclose( file );

    // ---- XDMF sidecar (rank 0, after the collective write completes) ------
    MPI_Barrier( comm );
    if ( R == 0 )
        writeXdmf( stem, Dim, static_cast<int>( sizeof( Scalar ) ),
                   static_cast<unsigned long long>( gcV.N ),
                   static_cast<unsigned long long>( gcF.N ), vXdmf, fXdmf );
}

} // namespace Tessera

#endif // TESSERA_HDF5_WRITER_HPP
