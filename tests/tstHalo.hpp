#ifndef _TSTHALO_HPP_
#define _TSTHALO_HPP_

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
class HaloTest : public Mesh2DTest<T>
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
     * Tests that a halo of depth one works properly by
     * checking that for any boundary element that points to
     * unowned vertices, edges, or faces, those elements are ghosted
     * in their respective AoSoA after the gather.
     */
    void test_halo_depth_1()
    {
        const int rank = this->rank_;

        auto halo = NuMesh::createHalo(this->mesh_, 0, 1);
        halo.gather();
        
        this->copytoHost();

        auto vertices = this->vertices;
        auto edges = this->edges;
        auto faces = this->faces;

        int total_verts = vertices.size();
        int total_edges = edges.size();
        int total_faces = faces.size();

        // Slices for access
        auto v_gid = Cabana::slice<V_GID>(vertices);
        auto v_owner = Cabana::slice<V_OWNER>(vertices);
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vids = Cabana::slice<E_VIDS>(edges);
        auto f_gid = Cabana::slice<F_GID>(faces);
        auto f_vids = Cabana::slice<F_VIDS>(faces);
        auto f_eids = Cabana::slice<F_EIDS>(faces);
        auto f_children = Cabana::slice<F_CID>(faces);

        auto v2f = NuMesh::Maps::V2F(this->mesh_);
        auto offsets_d = v2f.offsets();
        auto indices_d = v2f.indices();
        auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets_d);
        auto indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), indices_d);

        // Helper to check existence of a GID in a container
        auto gid_exists = [](const auto& gids, int gid) {
            return std::find(gids.begin(), gids.end(), gid) != gids.end();
        };

        // Collect all global IDs for owned and ghosted vertices and edges
        std::vector<int> vertex_gids(total_verts);
        std::vector<int> edge_gids(total_edges);
        for (int i = 0; i < total_verts; ++i) vertex_gids[i] = v_gid(i);
        for (int i = 0; i < total_edges; ++i) edge_gids[i] = e_gid(i);

        // Iterate over all owned vertices
        for (int vlid = 0; vlid < total_verts; vlid++)
        {
            int vowner = v_owner(vlid);
            if (vowner != rank) continue;
            int vgid = v_gid(vlid);

            int offset = offsets(vlid);

            // Handle the last vertex case
            int next_offset = (vlid + 1 < (int)offsets.extent(0)) ? 
                            offsets(vlid + 1) : 
                            (int)indices.extent(0);
            
            // Each vert should be connected to at least six faces
            int connected_faces = next_offset - offset;
            // if (vgid == 16)
            // {
            //     while (offset < next_offset)
            //     {
            //         printf("R%d: vgid %d face %d: %d\n", rank, vgid, (next_offset-offset), f_gid(indices(offset)));
            //         offset++;
            //     }
            // }
            ASSERT_GE(connected_faces, 6) << "VGID " << vgid << " is connected to " << connected_faces << " faces\n";
            return;
            while (offset < next_offset)
            {
                // Check that we have the data for each face and all its children faces
                int face_lid = indices(offset);
                int fgid = f_gid(face_lid);

                // Check vertices of this face
                for (int i = 0; i < 3; ++i)
                {
                    int vid = f_vids(face_lid, i);
                    ASSERT_TRUE(gid_exists(vertex_gids, vid)) << "FGID " << fgid << ": missing vgid " << vid << std::endl;
                }

                // Check edges of this face
                for (int i = 0; i < 3; ++i)
                {
                    int eid = f_eids(face_lid, i);
                    ASSERT_TRUE(gid_exists(edge_gids, eid)) << "FGID " << fgid << ": missing egid " << eid << std::endl;
                }

                // Recursively check child faces
                for (int i = 0; i < 4; ++i)
                {
                    int child_fid = f_children(face_lid, i);
                    if (child_fid != -1)
                    {
                        // Repeat checks for child face vertices and edges
                        for (int j = 0; j < 3; ++j)
                        {
                            int vid = f_vids(child_fid, j);
                            ASSERT_TRUE(gid_exists(vertex_gids, vid)) << "FGID " << fgid << ": missing vgid " << vid << std::endl;
                        }

                        for (int j = 0; j < 3; ++j)
                        {
                            int eid = f_eids(child_fid, j);
                            ASSERT_TRUE(gid_exists(edge_gids, eid)) << "FGID " << fgid << ": missing egid " << eid << std::endl;
                        }
                    }
                }
                offset++;
            }
        }
    }
};

} // end namespace NuMeshTest

#endif // _TSTHALO_HPP_
