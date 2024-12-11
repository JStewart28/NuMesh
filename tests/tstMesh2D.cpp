#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstMesh2D.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(Mesh2DTest, DeviceTypes);

/**
 * Tests refinement of two neighoring faces
 */
TYPED_TEST(Mesh2DTest, test0_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    Kokkos::View<int[2], Kokkos::HostSpace> fin("fin");
    fin(0) = 30; 
    fin(1) = 31;


    this->verifyRefinement(fin);
}

/**
 * Tests uniform refinement
 */
TYPED_TEST(Mesh2DTest, test1_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    int num_local_faces = this->numesh->count(NuMesh::Own(), NuMesh::Face());
    auto vef_gid_start = this->numesh->get_vef_gid_start();
    int face_gid_start = vef_gid_start(this->rank_, 2);

    Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
    for (int i = 0; i < num_local_faces; i++)
    {
        fin(i) = face_gid_start + i;
    }

    this->verifyRefinement(fin);
}

} // end namespace NuMeshTest
