#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstMesh2D.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(Mesh2DTest, DeviceTypes);

TYPED_TEST(Mesh2DTest, test_refinement)
{
    int mesh_size = 8;
    
    this->init(mesh_size, 1);

    int fin[10] = {30, 31, -1, -1, -1, -1, -1, -1, -1, -1};

    this->verifyRefinement(fin);
    
}
} // end namespace NuMeshTest
