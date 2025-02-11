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

    auto correct = this->populateTripleArray(NuMesh::Vertex(), 847);
    auto test = NuMesh::Array::ArrayOp::cloneCopy(*correct, NuMesh::Own());
    this->checkEqual(*correct, *test, 3, 3);
}

/**
 * Tests assign with a three-tuple
 */
TYPED_TEST(ArrayTest, test_assignDim3)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto three_size = this->populateTripleArray(NuMesh::Vertex(), 938);

    // Manually assign
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*three_size, NuMesh::Own());
    auto slice_c = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < three_size->aosoa().size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            slice_c(i, j) = 8.8;
        }
    }

    NuMesh::Array::ArrayOp::assign(*three_size, 8.8, NuMesh::Own());
    this->checkEqual(*correct, *three_size, 3, 3);
}

/**
 * Tests assign with a one-tuple
 */
TYPED_TEST(ArrayTest, test_assignDim1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto one_size = this->populateScalarArray(NuMesh::Vertex(), 938);

    // Manually assign
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*one_size, NuMesh::Own());
    auto slice_c = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < one_size->aosoa().size(); i++)
    {
        slice_c(i) = 8.8;
    }

    NuMesh::Array::ArrayOp::assign(*one_size, 8.8, NuMesh::Own());
    this->checkEqual(*correct, *one_size, 1, 1);
}

/**
 * Tests scale with a three-tuple
 */
TYPED_TEST(ArrayTest, test_scaleDim3)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto three_size = this->populateTripleArray(NuMesh::Vertex(), 938);

    // Manually scale
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*three_size, NuMesh::Own());
    auto slice_c = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < three_size->aosoa().size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            slice_c(i, j) *= 8.8;
        }
    }

    NuMesh::Array::ArrayOp::scale(*three_size, 8.8, NuMesh::Own());
    this->checkEqual(*correct, *three_size, 3, 3);
}

/**
 * Tests scale with a one-tuple
 */
TYPED_TEST(ArrayTest, test_scaleDim1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto one_size = this->populateScalarArray(NuMesh::Vertex(), 938);

    // Manually scale
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*one_size, NuMesh::Own());
    auto slice_c = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < one_size->aosoa().size(); i++)
    {
        slice_c(i) *= 8.8;
    }

    NuMesh::Array::ArrayOp::scale(*one_size, 8.8, NuMesh::Own());
    this->checkEqual(*correct, *one_size, 1, 1);
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

    auto three_size = this->populateTripleArray(NuMesh::Vertex(), 938);
    auto one_size = this->populateScalarArray(NuMesh::Vertex(), 235);

    // Manually multiply the first dimension of three_size into one_size
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*one_size, NuMesh::Own());
    auto slice3 = Cabana::slice<0>(three_size->aosoa());
    auto slice1 = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < three_size->aosoa().size(); i++)
    {
        double tmp = slice1(i);
        slice1(i) = slice1(i) * slice3(i, 0);
    }

    auto test = NuMesh::Array::ArrayOp::element_multiply(*one_size, *three_size, NuMesh::Own());
    this->checkEqual(*correct, *test, 1, 1);
}

/**
 * Tests elementMultiply with
 *  B having dim 3
 *  A having dim 3
 */
TYPED_TEST(ArrayTest, test_elementMultiplyDim3)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto three_size0 = this->populateTripleArray(NuMesh::Vertex(), 857);
    auto three_size1 = this->populateTripleArray(NuMesh::Vertex(), 286);

    // Manually multiply the first dimension of three_size into one_size
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*three_size0, NuMesh::Own());
    auto slice0 = Cabana::slice<0>(three_size1->aosoa());
    auto slice_c = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < three_size0->aosoa().size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            slice_c(i, j) = slice_c(i, j) * slice0(i, j);
        }
    }

    auto test = NuMesh::Array::ArrayOp::element_multiply(*three_size0, *three_size1, NuMesh::Own());
    this->checkEqual(*correct, *test, 3, 3);
}

/**
 * Tests copyDim with
 *  A having dim 2
 *  B having dim 1
 *  Copy dim 1 of B into dim 0 of A
 * This is a use case in Beatnik
 */
TYPED_TEST(ArrayTest, test_copyDim0)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto double_array = this->populateDoubleArray(NuMesh::Vertex(), 482);
    auto scalar_array = this->populateScalarArray(NuMesh::Vertex(), 915);

    // Manually copy the first dimension of B in to the first dim of A
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*double_array, NuMesh::Own());
    auto slice_s = Cabana::slice<0>(scalar_array->aosoa());
    auto slice_d = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < double_array->aosoa().size(); i++)
    {
        slice_d(i, 0) = slice_s(i);
    }

    NuMesh::Array::ArrayOp::copyDim(*double_array, 0, *scalar_array, 0, NuMesh::Own());
    this->checkEqual(*correct, *double_array, 2, 2);
}

/**
 * Tests copyDim with
 *  A having dim 2
 *  B having dim 1
 *  Copy dim 0 of B into dim 1 of A
 * This is a use case in Beatnik
 */
TYPED_TEST(ArrayTest, test_copyDim1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init(mesh_size, 1);

    auto double_array = this->populateDoubleArray(NuMesh::Vertex(), 207);
    auto scalar_array = this->populateScalarArray(NuMesh::Vertex(), 374);

    // Manually copy the first dimension of B in to the first dim of A
    auto correct = NuMesh::Array::ArrayOp::cloneCopy(*double_array, NuMesh::Own());
    auto slice_s = Cabana::slice<0>(scalar_array->aosoa());
    auto slice_d = Cabana::slice<0>(correct->aosoa());
    for (size_t i = 0; i < double_array->aosoa().size(); i++)
    {
        slice_d(i, 1) = slice_s(i);
    }

    NuMesh::Array::ArrayOp::copyDim(*double_array, 1, *scalar_array, 0, NuMesh::Own());
    this->checkEqual(*correct, *double_array, 2, 2);
}


} // end namespace NuMeshTest
