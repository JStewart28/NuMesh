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

} // end namespace NuMeshTest
