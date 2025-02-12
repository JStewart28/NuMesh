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
     * Tests the V2E by checking that for each edge,
     * its local ID appears in the correct spot in the map
     */
    void test_v2e()
    {
        this->gatherAndCopyToHost();

        auto v2e = NuMesh::Maps::V2E(this->mesh_);
        auto offsets_d = v2e.offsets();
        auto indices_d = v2e.indices();
        auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets_d);
        auto indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), indices_d);
        
        auto e_gid = Cabana::slice<E_GID>(this->edges);
        auto v_gid = Cabana::slice<V_GID>(this->vertices);
        auto e_vid = Cabana::slice<E_VIDS>(this->edges);

        /**
         * For some reason, gatherAndCopyToHost does not update the size of the 
         * mesh in mesh_, only the local vertices, edges, and faces values.
         * 
         * This means v2e only builds on the values in mesh_ before
         * gatherAndCopyToHost is called. To not index out-of-bounds,
         * we need to get the size from the mesh_ object
         */
        int num_vertices = this->mesh_->vertices().size();
        int num_edges = this->mesh_->edges().size();

        for (int i = 0; i < num_edges; i++)
        {
            int egid = e_gid(i);
            int elid = NuMesh::Utils::get_lid(e_gid, egid, 0, num_edges);
            for (int v = 0; v < 2; v++)
            {
                int vgid = e_vid(i, v);
                int vlid = NuMesh::Utils::get_lid(v_gid, vgid, 0, num_vertices);
                if (vlid == -1) continue;

                int offset = offsets(vlid);

                // Handle the last vertex case
                int next_offset = (vlid + 1 < (int)offsets.extent(0)) ? 
                                offsets(vlid + 1) : 
                                (int)indices.extent(0);

                bool is_present = false;
                while (offset < next_offset)
                {
                    int elid_in_map = indices(offset);
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

    /**
     * Tests the V2F by checking that for each face,
     * its local ID appears in the correct spot in the map
     */
    void test_v2f(int level)
    {
        this->copytoHost();

        auto v2f = NuMesh::Maps::V2F(this->mesh_, level);
        auto offsets_d = v2f.offsets();
        auto indices_d = v2f.indices();
        auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets_d);
        auto indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), indices_d);

        printf("faces size: %d\n", this->faces.size());
        
        auto f_gid = Cabana::slice<F_GID>(this->faces);
        auto v_gid = Cabana::slice<V_GID>(this->vertices);
        auto f_vid = Cabana::slice<E_VIDS>(this->faces);
        auto f_level = Cabana::slice<F_LAYER>(this->faces);

        int num_vertices = this->vertices.size();
        int num_faces = this->faces.size();

        // Check that offsets are properly increasing
        for (int v = 0; v < num_vertices-1; v++) {
            ASSERT_LE(offsets(v), offsets(v + 1)) << "Offsets should be monotonically increasing.";
        }

        // Verify the vertex-to-face connectivity
        for (int v = 0; v < num_vertices; v++)
        {
            std::set<int> expected_neighbors;
    
            // Iterate over all faces connected to this vertex
            for (int f = 0; f < num_faces; f++)
            {
                if (f_level(f) < level) continue; // Only consider faces at or above level
                printf("vlid %d, checking flid %d, l%d\n", v, f, f_level(f));
                for (int j = 0; j < 3; j++) {
                    printf("fvid(%d, %d) = %d, v_gid(%d) = %d\n", f, j, f_vid(f, j), v, v_gid(v));
                    if (f_vid(f, j) == v_gid(v)) {
                        printf("vlid %d has flid %d\n", v, f);
                        expected_neighbors.insert(f); // Insert this face
                        break;
                    }
                }
            }
    
            // Check that the actual neighbors match the expected ones
            int offset = offsets(v);
            int next_offset = (v + 1 < (int)offsets.extent(0)) ? offsets(v + 1) : (int)offsets.extent(0);
            std::set<int> actual_neighbors;
            for (int i = offset; i < next_offset; i++) {
                actual_neighbors.insert(f_gid(indices(i)));
            }
    
            ASSERT_EQ(actual_neighbors, expected_neighbors)
                << "Vertex " << v_gid(v) << " has incorrect face connectivity.";
        }
    }

    void test_v2v(int level)
    {
        this->copytoHost();
    
        auto v2v = NuMesh::Maps::V2V(this->mesh_, level);
        auto offsets_d = v2v.offsets();
        auto indices_d = v2v.indices();
        auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets_d);
        auto indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), indices_d);
    
        auto f_gid = Cabana::slice<F_GID>(this->faces);
        auto f_vid = Cabana::slice<F_VIDS>(this->faces);
        auto v_gid = Cabana::slice<V_GID>(this->vertices);
    
        int total_vertices = this->vertices.size();
    
        // Check that offsets are properly increasing
        for (int v = 0; v < total_vertices-1; v++) {
            ASSERT_LE(offsets(v), offsets(v + 1)) << "Offsets should be monotonically increasing.";
        }
    
        // Verify the vertex-to-vertex connectivity
        for (int v = 0; v < total_vertices; v++) {
            printf("R%d: checking vlid %d\n", this->rank_, v);
            std::set<int> expected_neighbors;
    
            // Iterate over all faces connected to this vertex
            for (int f = 0; f < (int)this->faces.size(); f++) {
                bool vertex_in_face = false;
                for (int j = 0; j < 3; j++) {
                    if (f_vid(f, j) == v_gid(v)) {
                        vertex_in_face = true;
                        break;
                    }
                }
    
                if (vertex_in_face) {
                    // Collect all neighboring vertices from this face
                    for (int j = 0; j < 3; j++) {
                        if (f_vid(f, j) != v_gid(v)) { // Avoid adding self
                            expected_neighbors.insert(f_vid(f, j));
                        }
                    }
                }
            }
    
            // Check that the actual neighbors match the expected ones
            int offset = offsets(v);
            int next_offset = (v + 1 < (int)offsets.extent(0)) ? offsets(v + 1) : (int)offsets.extent(0);
            std::set<int> actual_neighbors;
            for (int i = offsets(v); i < offsets(v + 1); i++) {
                actual_neighbors.insert(v_gid(indices(i)));
            }
    
            ASSERT_EQ(actual_neighbors, expected_neighbors)
                << "Vertex " << v_gid(v) << " has incorrect neighbors.";
        }
    }
    
};

} // end namespace NuMeshTest

#endif // _TSTMAPS_HPP_
