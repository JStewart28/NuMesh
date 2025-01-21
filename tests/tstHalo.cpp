#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstHalo.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(HaloTest, DeviceTypes);

/**
 * Tests that the v2e map is built correctly without any refinement
 */
TYPED_TEST(HaloTest, test_halo_depth_1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    this->test_halo_depth_1();
}

/**
 * Tests that the v2e map is built correctly with one layer of
 * uniform refinement
 */
TYPED_TEST(HaloTest, test_halo_depth_1_uniform_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    int num_local_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    auto vef_gid_start = this->mesh_->vef_gid_start();
    int face_gid_start = vef_gid_start(this->rank_, 2);

    Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
    for (int i = 0; i < num_local_faces; i++)
    {
        fin(i) = face_gid_start + i;
    }

    this->performRefinement(fin);

    this->test_halo_depth_1();
}

} // end namespace NuMeshTest
