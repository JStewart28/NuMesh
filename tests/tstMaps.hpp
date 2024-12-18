#ifndef _TSTMAPS_HPP_
#define _TSTMAPS_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

#include "tstMesh2D.hpp"

#include <mpi.h>

namespace NuMeshTest
{

template <class T>
class MapsTest : public Mesh2DTest<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
    using vertex_data = typename numesh_t::vertex_data;
    using edge_data = typename numesh_t::edge_data;
    using face_data = typename numesh_t::face_data;
    using v_array_type = Cabana::AoSoA<vertex_data, Kokkos::HostSpace, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, Kokkos::HostSpace, 4>;
    using f_array_type = Cabana::AoSoA<face_data, Kokkos::HostSpace, 4>;
    

  protected:

    void SetUp() override
    {
        Mesh2DTest<T>::SetUp();
    }

    void TearDown() override
    { 
        Mesh2DTest<T>::TearDown();
    }
    
    /**
     * Tests the v2e_map by checking that for each edge,
     * its local ID appears in the correct spot in the map
     */
    void test_v2e()
    {
        this->gatherAndCopyToHost();

        auto v2e = NuMesh::V2E_Map(this->mesh_);
        auto vertex_edge_offsets_d = v2e.vertex_edge_offsets();
        auto vertex_edge_indices_d = v2e.vertex_edge_indices();
        auto vertex_edge_offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vertex_edge_offsets_d);
        auto vertex_edge_indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vertex_edge_indices_d);

        // if (this->rank_ != 0) return;
        if (this->rank_ == 0) return;

        auto e_gid = Cabana::slice<E_GID>(this->edges);
        auto v_gid = Cabana::slice<V_GID>(this->vertices);
        auto e_vid = Cabana::slice<E_VIDS>(this->edges);

        int num_vertices = this->vertices.size();
        int num_edges = this->edges.size();

        /**
         * For some reason, gatherAndCopyToHost does not update the size of the 
         * mesh in mesh_, only the local vertices, edges, and faces values.
         * 
         * This means v2e only builds on the values in mesh_ before
         * gatherAndCopyToHost is called. Since gatherAndCopyToHost copies
         * all values to rank 0, rank 0 needs to adjust num_vertices
         * and num_edges
         */
        if (this->rank_ == 0)
        {
            num_vertices = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
            num_edges = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
        }

        for (int i = 0; i < num_edges; i++)
        {
            int egid = e_gid(i);
            int elid = NuMesh::Utils::get_lid(e_gid, egid, 0, num_edges);
            for (int v = 0; v < 2; v++)
            {
                int vgid = e_vid(i, v);
                int vlid = NuMesh::Utils::get_lid(v_gid, vgid, 0, num_vertices);
                if (vlid == -1) continue;

                int offset = vertex_edge_offsets(vlid);

                // **Handle the last vertex case**
                int next_offset = (vlid + 1 < (int)vertex_edge_offsets.extent(0)) ? 
                                vertex_edge_offsets(vlid + 1) : 
                                (int)vertex_edge_indices.extent(0);

                bool is_present = false;
                while (offset < next_offset)
                {
                    int elid_in_map = vertex_edge_indices(offset);
                    //printf("R%d: vlid: %d, elid: %d, elid in map: %d, offset: %d\n", this->rank_, vlid, elid, elid_in_map, offset);
                    if (elid_in_map == elid)
                    {
                        is_present = true;
                        break;
                    }
                    offset++;
                }
                ASSERT_TRUE(is_present) << "Rank " << this->rank_ << ": VIDS: " << vlid << "/" << vgid << ", offset: " << offset << ", EIDS: " << elid << "/" << egid << std::endl;
            }
        }
    }
};

} // end namespace NuMeshTest

#endif // _TSTMAPS_HPP_
