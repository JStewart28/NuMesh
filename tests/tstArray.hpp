#ifndef _TSTARRAY_HPP_
#define _TSTARRAY_HPP_

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

#include "tstMesh2D.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>

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
    
    using triple_tuple_type = Cabana::MemberTypes<double[3]>;
    using double_tuple_type = Cabana::MemberTypes<double[2]>;
    using scalar_tuple_type = Cabana::MemberTypes<double>;

  protected:
    void SetUp() override
    {
        Mesh2DTest<T>::SetUp();
    }

    void TearDown() override
    { 
        Mesh2DTest<T>::TearDown();
    }

  public:
    double generateUniqueDouble(int i, int j, int seed) {
        // Mix the inputs using bitwise operations for better uniqueness
        std::uint64_t hash = (std::uint64_t(i) * 73856093) ^ 
                            (std::uint64_t(j) * 19349663) ^ 
                            (std::uint64_t(seed) * 83492791);
        
        // Convert hash to a double in the range [0,1] using normalization
        double normalized = (hash % 1000000) / 1000000.0;
        
        // Scale it up to ensure non-trivial values and uniqueness
        return (normalized + 1.0) * seed * 0.01;
    }


    template <class EntityType>
    auto populateTripleArray(EntityType, int seed)
    {
        constexpr int tuple_size = NuMesh::ExtractArraySize<triple_tuple_type>::value;
        auto layout = NuMesh::Array::createArrayLayout<triple_tuple_type>(this->mesh_, tuple_size, EntityType());
        auto array = NuMesh::Array::createArray<Kokkos::HostSpace>("array", layout);
        auto aosoa = array->aosoa();
        auto slice = Cabana::slice<0>(aosoa);
        int max0 = aosoa.size();
        int max1 = tuple_size;

        for (int i = 0; i < max0; i++)
        {
            for (int j = 0; j < max1; j++)
            {
                double val = generateUniqueDouble(i, j, seed);
                slice(i, j) = val;
            }
        }

        // Copy to device memory
        auto array_d = NuMesh::Array::createArray<MemorySpace>("array_d", layout);
        auto aosoa_d = array_d->aosoa();
        Cabana::deep_copy(aosoa_d, aosoa);

        return array_d;
    }

    template <class EntityType>
    auto populateDoubleArray(EntityType, int seed)
    {
        constexpr int tuple_size = NuMesh::ExtractArraySize<double_tuple_type>::value;
        auto layout = NuMesh::Array::createArrayLayout<double_tuple_type>(this->mesh_, tuple_size, EntityType());
        auto array = NuMesh::Array::createArray<Kokkos::HostSpace>("array", layout);
        auto aosoa = array->aosoa();
        auto slice = Cabana::slice<0>(aosoa);
        int max0 = aosoa.size();
        int max1 = tuple_size;

        for (int i = 0; i < max0; i++)
        {
            for (int j = 0; j < max1; j++)
            {
                slice(i, j) = (double) (((i*j*seed)+j)%seed);
            }
        }

        // Copy to device memory
        auto array_d = NuMesh::Array::createArray<MemorySpace>("array_d", layout);
        auto aosoa_d = array_d->aosoa();
        Cabana::deep_copy(aosoa_d, aosoa);

        return array_d;
    }

    template <class EntityType>
    auto populateScalarArray(EntityType, int seed)
    {
        constexpr int tuple_size = NuMesh::ExtractArraySize<scalar_tuple_type>::value;
        auto layout = NuMesh::Array::createArrayLayout<scalar_tuple_type>(this->mesh_, tuple_size, EntityType());
        auto array = NuMesh::Array::createArray<Kokkos::HostSpace>("array", layout);
        auto aosoa = array->aosoa();
        auto slice = Cabana::slice<0>(aosoa);
        int max0 = aosoa.size();

        for (int i = 0; i < max0; i++)
        {
            slice(i) = (double) (i*29*seed);
        }

        // Copy to device memory
        auto array_d = NuMesh::Array::createArray<MemorySpace>("array_d", layout);
        auto aosoa_d = array_d->aosoa();
        Cabana::deep_copy(aosoa_d, aosoa);

        return array_d;
    }
    
    template <class A_t, class B_t>
    void checkEqual(const A_t& a, const B_t& b, int asize, int bsize)
    {
        using a_entity_type = typename A_t::entity_type;
        using b_entity_type = typename B_t::entity_type;
        using a_memory_space = typename A_t::memory_space;
        using b_memory_space = typename B_t::memory_space;
        using a_execution_space = typename A_t::execution_space;
        using b_execution_space = typename B_t::execution_space;
        using a_tuple_type = typename A_t::tuple_type;
        using b_tuple_type = typename B_t::tuple_type;

        // Check that the types are equal for both arrays
        static_assert(std::is_same<a_entity_type, b_entity_type>::value,
            "tstArray::checkEqual: Types are not the same!");
        static_assert(std::is_same<a_memory_space, b_memory_space>::value,
            "tstArray::checkEqual: Types are not the same!");
        static_assert(std::is_same<a_execution_space, b_execution_space>::value,
            "tstArray::checkEqual: Types are not the same!");

        // Check dimensions
        auto a_aosoa = a.aosoa();
        auto b_aosoa = b.aosoa();
        auto a_slice = Cabana::slice<0>(a_aosoa);
        auto b_slice = Cabana::slice<0>(b_aosoa);

        const int amax0 = a_aosoa.size();
        const int bmax0 = b_aosoa.size();
        constexpr int amax1 = NuMesh::ExtractArraySize<a_tuple_type>::value;
        constexpr int bmax1 = NuMesh::ExtractArraySize<b_tuple_type>::value;

        // Ensure bounds are the same
        ASSERT_EQ(amax0, bmax0); ASSERT_EQ(amax1, asize); ASSERT_EQ(bmax1, bsize);

        // Copy data to host memory
        Cabana::AoSoA<a_tuple_type, Kokkos::HostSpace> ahost("ahost", amax0);
        Cabana::AoSoA<b_tuple_type, Kokkos::HostSpace> bhost("bhost", bmax0);
        Cabana::deep_copy(ahost, a_aosoa);
        Cabana::deep_copy(bhost, b_aosoa);
        auto ah_slice = Cabana::slice<0>(ahost);
        auto bh_slice = Cabana::slice<0>(bhost);

        // Ensure values are the same
        if constexpr (amax1 == 1)
        {
            for (int i = 0; i < amax0; i++)
            {
                double correct = ah_slice(i);
                double test = bh_slice(i);
                // printf("Test: R%d: i%d: correct: %0.1lf, test: %0.1lf\n", this->rank_, i, correct, test);
                ASSERT_DOUBLE_EQ(correct, test);
            }
        }
        else if constexpr (amax1 > 1)
        {
            for (int i = 0; i < amax0; i++)
            {
                for (int j = 0; j < amax1; j++)
                {
                    if (j >= bmax1) break; // b could have a smaller tuple than a
                    double correct = ah_slice(i, j);
                    double test = bh_slice(i, j);
                    ASSERT_DOUBLE_EQ(correct, test);
                }
            }
        }
        else
        {
            throw std::runtime_error("checkEqual: Invalid tuple sizes");
        }
        
    }

};

} // end namespace NuMeshTest

#endif // _TSTARRAY_HPP_