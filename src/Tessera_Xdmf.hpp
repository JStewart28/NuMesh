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

#include <cstdio>
#include <string>
#include <vector>

namespace Tessera
{
namespace detail
{

//! One user-field attribute to add to the XDMF grid.
struct XdmfField
{
    std::string dataset; // HDF5 dataset name within its group, e.g. "u0"
    std::string name;    // Attribute display name shown in Paraview
    int extent;          // 1 (scalar) or N (Scalar[N])
};

inline std::string xdmfBasename( const std::string& path )
{
    const auto pos = path.find_last_of( "/\\" );
    return pos == std::string::npos ? path : path.substr( pos + 1 );
}

inline void writeXdmfAttribute( std::FILE* fp, const XdmfField& f,
                                const char* center, const char* groupPath,
                                const std::string& h5name,
                                unsigned long long count, int dim,
                                int scalarBytes )
{
    const char* atype =
        ( f.extent == dim && f.extent > 1 ) ? "Vector" : "Scalar";
    if ( f.extent > 1 )
        std::fprintf(
            fp,
            "    <Attribute Name=\"%s\" Center=\"%s\" AttributeType=\"%s\">\n"
            "      <DataItem Dimensions=\"%llu %d\" NumberType=\"Float\" "
            "Precision=\"%d\" Format=\"HDF\">\n"
            "        %s:%s/%s</DataItem></Attribute>\n",
            f.name.c_str(), center, atype, count, f.extent, scalarBytes,
            h5name.c_str(), groupPath, f.dataset.c_str() );
    else
        std::fprintf(
            fp,
            "    <Attribute Name=\"%s\" Center=\"%s\" "
            "AttributeType=\"Scalar\">\n"
            "      <DataItem Dimensions=\"%llu\" NumberType=\"Float\" "
            "Precision=\"%d\" Format=\"HDF\">\n"
            "        %s:%s/%s</DataItem></Attribute>\n",
            f.name.c_str(), center, count, scalarBytes, h5name.c_str(),
            groupPath, f.dataset.c_str() );
}

} // namespace detail

//! Write the XDMF sidecar describing `<stem>.h5`. `dim` must be 2 or 3 to
//! emit a Topology/Geometry surface grid; any other value emits attribute-only
//! metadata with a one-line warning (Milestone 1 always uses Dim=3).
inline void writeXdmf( const std::string& stem, int dim, int scalarBytes,
                       unsigned long long Nv, unsigned long long Nf,
                       const std::vector<detail::XdmfField>& vFields,
                       const std::vector<detail::XdmfField>& fFields )
{
    const std::string h5name = detail::xdmfBasename( stem ) + ".h5";
    const std::string xmfname = stem + ".xmf";
    std::FILE* fp = std::fopen( xmfname.c_str(), "w" );
    if ( !fp )
        return;

    std::fprintf( fp, "<?xml version=\"1.0\" ?>\n" );
    std::fprintf( fp, "<Xdmf Version=\"3.0\"><Domain>\n" );
    std::fprintf( fp, "  <Grid Name=\"Tessera\" GridType=\"Uniform\">\n" );

    if ( dim == 2 || dim == 3 )
    {
        std::fprintf(
            fp,
            "    <Topology TopologyType=\"Triangle\" "
            "NumberOfElements=\"%llu\">\n"
            "      <DataItem Dimensions=\"%llu 3\" NumberType=\"UInt\" "
            "Precision=\"8\" Format=\"HDF\">\n"
            "        %s:/faces/verts</DataItem>\n"
            "    </Topology>\n",
            Nf, Nf, h5name.c_str() );
        std::fprintf(
            fp,
            "    <Geometry GeometryType=\"%s\">\n"
            "      <DataItem Dimensions=\"%llu %d\" NumberType=\"Float\" "
            "Precision=\"%d\" Format=\"HDF\">\n"
            "        %s:/vertices/position</DataItem>\n"
            "    </Geometry>\n",
            dim == 3 ? "XYZ" : "XY", Nv, dim, scalarBytes, h5name.c_str() );
    }
    else
    {
        std::fprintf( fp,
                      "    <!-- Dim=%d is not a visualizable surface grid; "
                      "HDF5 data only. -->\n",
                      dim );
    }

    std::fprintf( fp,
                  "    <Attribute Name=\"v_gid\" Center=\"Node\" "
                  "AttributeType=\"Scalar\">\n"
                  "      <DataItem Dimensions=\"%llu\" NumberType=\"UInt\" "
                  "Precision=\"8\" "
                  "Format=\"HDF\">\n"
                  "        %s:/vertices/gid</DataItem></Attribute>\n",
                  Nv, h5name.c_str() );
    std::fprintf( fp,
                  "    <Attribute Name=\"f_level\" Center=\"Cell\" "
                  "AttributeType=\"Scalar\">\n"
                  "      <DataItem Dimensions=\"%llu\" NumberType=\"Int\" "
                  "Precision=\"2\" "
                  "Format=\"HDF\">\n"
                  "        %s:/faces/level</DataItem></Attribute>\n",
                  Nf, h5name.c_str() );

    for ( const auto& f : vFields )
        detail::writeXdmfAttribute( fp, f, "Node", "/vertices", h5name, Nv, dim,
                                    scalarBytes );
    for ( const auto& f : fFields )
        detail::writeXdmfAttribute( fp, f, "Cell", "/faces", h5name, Nf, dim,
                                    scalarBytes );

    std::fprintf( fp, "  </Grid>\n" );
    std::fprintf( fp, "</Domain></Xdmf>\n" );
    std::fclose( fp );
}

} // namespace Tessera

#endif // TESSERA_XDMF_HPP
