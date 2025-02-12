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
TYPED_TEST(MapsTest, test_v2e_no_refinement)
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
TYPED_TEST(MapsTest, test_v2f_no_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    this->test_v2f();
}

/**
 * Tests that the v2v map is built correctly without any refinement
 */
TYPED_TEST(MapsTest, test_v2v_no_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    this->test_v2v();
}


} // end namespace NuMeshTest
