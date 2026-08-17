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

#ifndef TESSERA_XDMF_HPP
#define TESSERA_XDMF_HPP

// XDMF sidecar (<stem>.xmf) for the parallel HDF5 mesh writer (Step 8). Plain
// text XML with no HDF5 dependency: it references the dense HDF5 datasets
// writeMesh() already wrote, so it is byte-identical regardless of the writer
// rank count. Written once by rank 0, after the collective HDF5 write
// completes (caller barriers first). Edges are round-trip data, not a
// standard surface cell type, so they are intentionally left out of the XDMF
// grid (out of scope for the gate; see the Step-8 spec).

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace Tessera
{

//! One user-field attribute to add to the XDMF grid.
struct XdmfField
{
    std::string dataset; // HDF5 dataset name within its group, e.g. "u0"
    std::string name;    // Attribute display name shown in Paraview
    int extent;          // 1 (scalar) or N (Scalar[N])
};

//! Everything the XML needs to describe one written mesh frame: the metadata
//! writeMesh() already knows once its collective HDF5 write has completed.
//! Returned by writeMesh() so a caller can re-emit the same grid later (e.g.
//! as one child of a temporal collection) without reopening the HDF5 file.
struct XdmfFrame
{
    //! BASENAME of the HDF5 file this grid references, e.g. "run_frame3.h5" --
    //! never a path. An .xmf can only reference .h5 files in its own
    //! directory, because XDMF resolves the part before the ':' relative to
    //! the .xmf itself.
    std::string h5name;
    int dim;                        // 2 or 3 emits a surface grid; see below
    int scalarBytes;                // sizeof(Scalar): XDMF Precision=
    unsigned long long Nv;          // global owned-vertex count
    unsigned long long Nf;          // global owned-face count
    std::vector<XdmfField> vFields; // node-centered user fields
    std::vector<XdmfField> fFields; // cell-centered user fields
};

namespace detail
{

inline std::string xdmfBasename( const std::string& path )
{
    const auto pos = path.find_last_of( "/\\" );
    return pos == std::string::npos ? path : path.substr( pos + 1 );
}

//! `indent` is the column the enclosing `<Grid>` starts at, in spaces; every
//! line below is written relative to it. See writeXdmfGrid() for why the
//! parameter exists at all.
inline void writeXdmfAttribute( std::FILE* fp, const XdmfField& f,
                                const char* center, const char* groupPath,
                                const std::string& h5name,
                                unsigned long long count, int dim,
                                int scalarBytes, int indent )
{
    const std::string pad( static_cast<std::size_t>( indent ), ' ' );
    const char* atype =
        ( f.extent == dim && f.extent > 1 ) ? "Vector" : "Scalar";
    if ( f.extent > 1 )
        std::fprintf(
            fp,
            "%s  <Attribute Name=\"%s\" Center=\"%s\" AttributeType=\"%s\">\n"
            "%s    <DataItem Dimensions=\"%llu %d\" NumberType=\"Float\" "
            "Precision=\"%d\" Format=\"HDF\">\n"
            "%s      %s:%s/%s</DataItem></Attribute>\n",
            pad.c_str(), f.name.c_str(), center, atype, pad.c_str(), count,
            f.extent, scalarBytes, pad.c_str(), h5name.c_str(), groupPath,
            f.dataset.c_str() );
    else
        std::fprintf(
            fp,
            "%s  <Attribute Name=\"%s\" Center=\"%s\" "
            "AttributeType=\"Scalar\">\n"
            "%s    <DataItem Dimensions=\"%llu\" NumberType=\"Float\" "
            "Precision=\"%d\" Format=\"HDF\">\n"
            "%s      %s:%s/%s</DataItem></Attribute>\n",
            pad.c_str(), f.name.c_str(), center, pad.c_str(), count,
            scalarBytes, pad.c_str(), h5name.c_str(), groupPath,
            f.dataset.c_str() );
}

//! The whole `<Grid>` element for one frame, `<Time>` included when `time` is
//! non-null. Both public writeXdmfGrid() overloads below funnel here so the
//! timed and untimed spellings can never drift apart -- the API is an overload
//! pair, not a pointer, but the emitter is written once.
inline void writeXdmfGridImpl( std::FILE* fp, const XdmfFrame& frame,
                               int indent, const double* time )
{
    const std::string pad( static_cast<std::size_t>( indent ), ' ' );
    const char* h5 = frame.h5name.c_str();

    std::fprintf( fp, "%s<Grid Name=\"Tessera\" GridType=\"Uniform\">\n",
                  pad.c_str() );
    // %.17g so a double round-trips exactly through the XML text.
    if ( time )
        std::fprintf( fp, "%s  <Time Value=\"%.17g\"/>\n", pad.c_str(),
                      *time );

    if ( frame.dim == 2 || frame.dim == 3 )
    {
        std::fprintf(
            fp,
            "%s  <Topology TopologyType=\"Triangle\" "
            "NumberOfElements=\"%llu\">\n"
            "%s    <DataItem Dimensions=\"%llu 3\" NumberType=\"UInt\" "
            "Precision=\"8\" Format=\"HDF\">\n"
            "%s      %s:/faces/verts</DataItem>\n"
            "%s  </Topology>\n",
            pad.c_str(), frame.Nf, pad.c_str(), frame.Nf, pad.c_str(), h5,
            pad.c_str() );
        std::fprintf(
            fp,
            "%s  <Geometry GeometryType=\"%s\">\n"
            "%s    <DataItem Dimensions=\"%llu %d\" NumberType=\"Float\" "
            "Precision=\"%d\" Format=\"HDF\">\n"
            "%s      %s:/vertices/position</DataItem>\n"
            "%s  </Geometry>\n",
            pad.c_str(), frame.dim == 3 ? "XYZ" : "XY", pad.c_str(), frame.Nv,
            frame.dim, frame.scalarBytes, pad.c_str(), h5, pad.c_str() );
    }
    else
    {
        std::fprintf( fp,
                      "%s  <!-- Dim=%d is not a visualizable surface grid; "
                      "HDF5 data only. -->\n",
                      pad.c_str(), frame.dim );
    }

    std::fprintf( fp,
                  "%s  <Attribute Name=\"v_gid\" Center=\"Node\" "
                  "AttributeType=\"Scalar\">\n"
                  "%s    <DataItem Dimensions=\"%llu\" NumberType=\"UInt\" "
                  "Precision=\"8\" "
                  "Format=\"HDF\">\n"
                  "%s      %s:/vertices/gid</DataItem></Attribute>\n",
                  pad.c_str(), pad.c_str(), frame.Nv, pad.c_str(), h5 );
    std::fprintf( fp,
                  "%s  <Attribute Name=\"f_level\" Center=\"Cell\" "
                  "AttributeType=\"Scalar\">\n"
                  "%s    <DataItem Dimensions=\"%llu\" NumberType=\"Int\" "
                  "Precision=\"2\" "
                  "Format=\"HDF\">\n"
                  "%s      %s:/faces/level</DataItem></Attribute>\n",
                  pad.c_str(), pad.c_str(), frame.Nf, pad.c_str(), h5 );

    for ( const auto& f : frame.vFields )
        writeXdmfAttribute( fp, f, "Node", "/vertices", frame.h5name, frame.Nv,
                            frame.dim, frame.scalarBytes, indent );
    for ( const auto& f : frame.fFields )
        writeXdmfAttribute( fp, f, "Cell", "/faces", frame.h5name, frame.Nf,
                            frame.dim, frame.scalarBytes, indent );

    std::fprintf( fp, "%s</Grid>\n", pad.c_str() );
}

//! Emit one frame's `<Grid>` element, with no `<Time>`.
//!
//! `indent` is the column the `<Grid>` line starts at, in spaces: a standalone
//! sidecar nests the grid directly in `<Domain>` (2), while a temporal
//! collection master nests it one level deeper inside the collection grid (4).
//! That is the only reason the parameter exists.
//!
//! `frame.dim` must be 2 or 3 to emit a Topology/Geometry surface grid; any
//! other value emits attribute-only metadata with a one-line warning
//! (Milestone 1 always uses Dim=3).
inline void writeXdmfGrid( std::FILE* fp, const XdmfFrame& frame, int indent )
{
    writeXdmfGridImpl( fp, frame, indent, nullptr );
}

//! Emit one frame's `<Grid>` element with a `<Time Value=>` child. `time` is
//! whatever the caller means by time -- physical time or a step index -- in
//! the caller's own units; nothing here interprets it.
inline void writeXdmfGrid( std::FILE* fp, const XdmfFrame& frame, int indent,
                           double time )
{
    writeXdmfGridImpl( fp, frame, indent, &time );
}

//! `<stem>.xmf` holding one frame, `<Time>` included when `time` is non-null.
//! Both public writeXdmf() overloads funnel here.
inline void writeXdmfFile( const std::string& stem, const XdmfFrame& frame,
                           const double* time )
{
    const std::string xmfname = stem + ".xmf";
    std::FILE* fp = std::fopen( xmfname.c_str(), "w" );
    if ( !fp )
        return;

    std::fprintf( fp, "<?xml version=\"1.0\" ?>\n" );
    std::fprintf( fp, "<Xdmf Version=\"3.0\"><Domain>\n" );
    writeXdmfGridImpl( fp, frame, 2, time );
    std::fprintf( fp, "</Domain></Xdmf>\n" );
    std::fclose( fp );
}

} // namespace detail

//! Write the XDMF sidecar describing `frame.h5name`, which must sit in the
//! same directory as `<stem>.xmf`. No `<Time>` element: the grid is timeless,
//! exactly as it has always been for a single-shot writeMesh().
inline void writeXdmf( const std::string& stem, const XdmfFrame& frame )
{
    detail::writeXdmfFile( stem, frame, nullptr );
}

//! As above, plus a `<Time Value=>` child on the grid. `time` is the caller's
//! own quantity (physical time or a step index) in the caller's own units.
inline void writeXdmf( const std::string& stem, const XdmfFrame& frame,
                       double time )
{
    detail::writeXdmfFile( stem, frame, &time );
}

} // namespace Tessera

#endif // TESSERA_XDMF_HPP
