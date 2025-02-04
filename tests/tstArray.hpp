#ifndef _TSTARRAY_HPP_
#define _TSTARRAY_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

#include "tstMesh2D.hpp"

#include <mpi.h>

namespace NuMeshTest
{

template <class T>
class ArrayTest : public Mesh2DTest<T>
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using mesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
    using vertex_data = typename mesh_t::vertex_data;
    using edge_data = typename mesh_t::edge_data;
    using face_data = typename mesh_t::face_data;
    using v_array_type = Cabana::AoSoA<vertex_data, Kokkos::HostSpace, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, Kokkos::HostSpace, 4>;
    using f_array_type = Cabana::AoSoA<face_data, Kokkos::HostSpace, 4>;

    using tuple_type = Cabana::MemberTypes<double[3]>;
    

  protected:
    int rank_, comm_size_;
    int periodic_;
    std::shared_ptr<mesh_t> mesh_ = NuMesh::createEmptyMesh<ExecutionSpace, MemorySpace>(MPI_COMM_WORLD);
    v_array_type vertices;
    e_array_type edges;
    f_array_type faces;
    // l = local, g = ghost
    int lv = -1, le = -1, lf = -1, gv = -1, ge = -1, gf = -1;

    void SetUp() override
    {
        Mesh2DTest<T>::SetUp();
    }

    void TearDown() override
    { 
        Mesh2DTest<T>::TearDown();
    }

  public:

    template <class EntityType>
    auto populateArray(EntityType)
    {
        auto vertex_triple_layout = NuMesh::Array::createArrayLayout<tuple_type>(this->mesh_, 3, EntityType());
        auto array = NuMesh::Array::createArray<Kokkos::HostSpace>("positions", vertex_triple_layout);
        auto aosoa = array->aosoa();
        auto slice = Cabana::slice<0>(aosoa);
        size_t max0 = slice.extent(0);
        size_t max1 = slice.extent(1);

        for (size_t i = 0; i < max0; i++)
        {
            for (size_t j = 0; j < max1; j++)
            {
                slice(i, j) = (double) ((i*j)+j);
            }
        }

        // Copy to device memory
        auto array_d = NuMesh::Array::createArray<MemorySpace>("positions_d", vertex_triple_layout);
        auto aosoa_d = array_d->aosoa();
        Cabana::deep_copy(aosoa_d, aosoa);

        return array_d;
    }
    
    template <class Array_t>
    void checkEqual(const Array_t& a, const Array_t& b)
    {
        using execution_space = typename Array_t::execution_space;
        using entity_type = typename Array_t::entity_type;

        auto a_aosoa = a.aosoa();
        auto b_aosoa = b.aosoa();
        auto a_slice = Cabana::slice<0>(a_aosoa);
        auto b_slice = Cabana::slice<0>(b_aosoa);
        size_t amax0 = a_slice.extent(0);
        size_t amax1 = a_slice.extent(1);
        size_t bmax0 = b_slice.extent(0);
        size_t bmax1 = b_slice.extent(1);

        // Ensure bounds are the same
        ASSERT_EQ(amax0, bmax0); ASSERT_EQ(amax1, bmax1);

        // Copy data to host memory
        Cabana::AoSoA<tuple_type, Kokkos::HostSpace> ahost("ahost", amax0);
        Cabana::AoSoA<tuple_type, Kokkos::HostSpace> bhost("bhost", bmax0);
        Cabana::deep_copy(ahost, a_aosoa);
        Cabana::deep_copy(bhost, b_aosoa);
        auto ah_slice = Cabana::slice<0>(ahost);
        auto bh_slice = Cabana::slice<0>(bhost);

        // Ensure values are the same
       for (size_t i = 0; i < amax0; i++)
        {
            for (size_t j = 0; j < amax1; j++)
            {
                double correct = ah_slice(i, j);
                double test = bh_slice(i, j);
                ASSERT_DOUBLE_EQ(correct, test);
            }
        }
    }

};

} // end namespace NuMeshTest

#endif // _TSTARRAY_HPP_