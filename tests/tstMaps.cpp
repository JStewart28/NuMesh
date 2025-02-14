#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstMaps.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(MapsTest, DeviceTypes);

/**
 * Tests that the v2e map is built correctly without any refinement
 */
TYPED_TEST(MapsTest, test_v2e_refinement0)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    this->test_v2e();
}

/**
 * Tests that the v2f map is built correctly without any refinement
 */
TYPED_TEST(MapsTest, test_v2f_refinement0)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    this->test_v2f(0);
}

/**
 * Tests that the v2f map is built correctly with one iteration
 * of uniform refinement
 */
TYPED_TEST(MapsTest, test_v2f_refinement1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto vef_gid_start = this->mesh_->vef_gid_start();

    // Uniform refinement
    for (int i = 0; i < 2; i++)
    {
        int num_local_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
        int face_gid_start = vef_gid_start(this->rank_, 2);
        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }
        this->performRefinement(fin);
    }

    this->test_v2f(0);
}

/**
 * Tests that the v2v map is built correctly without any refinement
 */
TYPED_TEST(MapsTest, test_v2v_refinement0)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    this->test_v2v(0);
}


} // end namespace NuMeshTest
