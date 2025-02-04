#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

#include "tstArray.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(ArrayTest, DeviceTypes);

/**
 * Tests that the v2e map is built correctly without any refinement
 */
TYPED_TEST(ArrayTest, test_cloneCopy)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto correct = this->populateArray(NuMesh::Vertex());
    auto test = NuMesh::Array::ArrayOp::cloneCopy(*correct, NuMesh::Own());
    this->checkEqual(*correct, *test);
}


} // end namespace NuMeshTest
