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

    Kokkos::View<int[2], Kokkos::HostSpace> fin("fin");
    fin(0) = 30; 
    fin(1) = 31;


    this->verifyRefinement(fin);
    
}
} // end namespace NuMeshTest
