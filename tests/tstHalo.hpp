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
        auto halo = NuMesh::createHalo(this->mesh_, 0, 1);
        halo.gather();
        
        this->copytoHost();

        auto vertices = this->vertices;
        auto edges = this->edges;
        auto faces = this->faces;

        int total_verts = vertices.size();
        int total_edges = edges.size();
        int total_faces = faces.size();

        // Slices we need
        auto v_gid = Cabana::slice<V_GID>(vertices);
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vids = Cabana::slice<E_VIDS>(edges);
        auto f_gid = Cabana::slice<F_GID>(faces);
        auto f_vids = Cabana::slice<F_VIDS>(faces);
        auto f_eids = Cabana::slice<F_EIDS>(faces);

        // Check that we have all vertices on edges we own
        int owned_edges = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
        for (int elid = 0; elid < owned_edges; elid++)
        {
            int egid = e_gid(elid);
            for (int v = 0; v < 2; v++)
            {
                int evgid = e_vids(elid, v);
                int evlid = NuMesh::Utils::get_lid(v_gid, evgid, 0, total_verts);
                ASSERT_NE(evlid, -1) << "Rank " << this->rank_ << ": EGID " << egid << " vgid " << evgid << " not found";
            }
        }

        // Check that we have all edges on faces we own
        int owned_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
        for (int flid = 0; flid < owned_faces; flid++)
        {
            int fgid = f_gid(flid);
            for (int e = 0; e < 3; e++)
            {
                int fegid = f_eids(flid, e);
                int felid = NuMesh::Utils::get_lid(e_gid, fegid, 0, total_edges);
                // printf("R%d: checking fgid %d: egid/lid %d, %d\n", this->rank_, fgid, fegid, felid);
                ASSERT_NE(felid, -1) << "Rank " << this->rank_ << ": FGID " << fgid << " egid " << fegid << " not found";
            }
        }
    }
};

} // end namespace NuMeshTest

#endif // _TSTHALO_HPP_
