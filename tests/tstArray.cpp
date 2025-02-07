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
 * Tests cloneCopy
 */
TYPED_TEST(ArrayTest, test_cloneCopy)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto correct = this->populateTripleArray(NuMesh::Vertex());
    auto test = NuMesh::Array::ArrayOp::cloneCopy(*correct, NuMesh::Own());
    this->checkEqual(*correct, *test, 3, 3);
}

/**
 * Tests elementMultiply with
 *  B having dim 1
 *  A having dim 3
 */
TYPED_TEST(ArrayTest, test_elementMultiplyDim1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto three_size = this->populateTripleArray(NuMesh::Vertex());
    auto one_size = this->populateScalarArray(NuMesh::Vertex());

    // Manually multiply the first dimension of three_size into one_size
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*one_size, NuMesh::Own());
    auto slice3 = Cabana::slice<0>(three_size->aosoa());
    auto slice1 = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < three_size->aosoa().size(); i++)
    {
        slice1(i) = slice1(i) * slice3(i, 0);
    }

    auto test = NuMesh::Array::ArrayOp::element_multiply(*one_size, *three_size, NuMesh::Own());
    this->checkEqual(*correct, *test);
}


} // end namespace NuMeshTest
