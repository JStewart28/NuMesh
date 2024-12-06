#ifndef _TSTMESH2D_HPP_
#define _TSTMESH2D_HPP_

#include <iostream>
#include <filesystem>
#include <regex>

#include "gtest/gtest.h"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>

// #include "TestingUtils.hpp"

#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

template <class T>
class Mesh2DTest : public ::testing::Test
{
    using ExecutionSpace = typename T::ExecutionSpace;
    using MemorySpace = typename T::MemorySpace;
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
    using vertex_data = typename numesh_t::vertex_data;
    using edge_data = typename numesh_t::edge_data;
    using face_data = typename numesh_t::face_data;
    using v_array_type = Cabana::AoSoA<vertex_data, Kokkos::HostSpace, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, Kokkos::HostSpace, 4>;
    using f_array_type = Cabana::AoSoA<face_data, Kokkos::HostSpace, 4>;
    

  protected:
    int rank_, comm_size_;
    std::shared_ptr<numesh_t> numesh = NuMesh::createEmptyMesh<ExecutionSpace, MemorySpace>(MPI_COMM_WORLD);
    v_array_type vertices;
    e_array_type edges;
    f_array_type faces;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
    }

    void TearDown() override
    { 
    }

  public:
    void init(int mesh_size, int periodic)
    {
        std::array<int, 2> global_num_cell = { mesh_size, mesh_size };
        std::array<double, 2> global_low_corner = { -1.0, -1.0 };
        std::array<double, 2> global_high_corner = { 1.0, 1.0 };
        // std::array<double, 6> global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
        std::array<bool, 2> periodic_a = { (bool)periodic, (bool)periodic };
        Cabana::Grid::DimBlockPartitioner<2> partitioner;

        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        auto global_grid = Cabana::Grid::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, periodic_a, partitioner );
        int halo_width = 2;
        auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        auto layout = Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Node());
        auto array = Cabana::Grid::createArray<double, MemorySpace>("for_initialization", layout);
        this->numesh->initializeFromArray(*array);
    }

    /**
     * Gather the entire mesh to rank 0 and copy to host memory
     */
    void gatherAndCopyToHost()
    {
        int rank = rank_;

        auto vertices_ptr = numesh->vertices();
        auto edges_ptr = numesh->edges();
        auto faces_ptr = numesh->faces();

         // Local counts for each rank
        int local_vef_count[3] = {numesh->count(NuMesh::Own(), NuMesh::Vertex()),
                                numesh->count(NuMesh::Own(), NuMesh::Edge()),
                                numesh->count(NuMesh::Own(), NuMesh::Face())};
        return;
        // Get vertices
        Kokkos::View<int*, MemorySpace> element_export_ids("element_export_ids", local_vef_count[0]);
        Kokkos::View<int*, MemorySpace> element_export_ranks("element_export_ranks", local_vef_count[0]);
        Kokkos::parallel_for("init_halo_data", Kokkos::RangePolicy<ExecutionSpace>(0, local_vef_count[0]),
            KOKKOS_LAMBDA(int i) {
            
            if (rank == 0)
            {
                element_export_ids(i) = 0;
                element_export_ranks(i) = -1;
            }
            else
            {
                element_export_ids(i) = i;
                element_export_ranks(i) = 0;
            }

        });

        auto vert_halo = Cabana::Halo<MemorySpace>(MPI_COMM_WORLD, local_vef_count[0], element_export_ids,
            element_export_ranks);

        vertices_ptr.resize(vert_halo.numLocal() + vert_halo.numGhost());
        vertices.resize(vert_halo.numLocal() + vert_halo.numGhost());

        Cabana::gather(vert_halo, vertices_ptr);

        // Get edges
        Kokkos::resize(element_export_ids, local_vef_count[1]);
        Kokkos::resize(element_export_ranks, local_vef_count[1]);
        Kokkos::parallel_for("init_halo_data", Kokkos::RangePolicy<ExecutionSpace>(0, local_vef_count[1]),
            KOKKOS_LAMBDA(int i) {
            
            if (rank == 0)
            {
                element_export_ids(i) = 0;
                element_export_ranks(i) = -1;
            }
            else
            {
                element_export_ids(i) = i;
                element_export_ranks(i) = 0;
            }

        });

        auto edge_halo = Cabana::Halo<MemorySpace>(MPI_COMM_WORLD, local_vef_count[1], element_export_ids,
            element_export_ranks);

        edges_ptr.resize(edge_halo.numLocal() + edge_halo.numGhost());
        edges.resize(edge_halo.numLocal() + edge_halo.numGhost());

        Cabana::gather(edge_halo, edges_ptr);

        // Get Faces
        Kokkos::resize(element_export_ids, local_vef_count[2]);
        Kokkos::resize(element_export_ranks, local_vef_count[2]);
        Kokkos::parallel_for("init_halo_data", Kokkos::RangePolicy<ExecutionSpace>(0, local_vef_count[2]),
            KOKKOS_LAMBDA(int i) {
            
            if (rank == 0)
            {
                element_export_ids(i) = 0;
                element_export_ranks(i) = -1;
            }
            else
            {
                element_export_ids(i) = i;
                element_export_ranks(i) = 0;
            }

        });

        auto face_halo = Cabana::Halo<MemorySpace>(MPI_COMM_WORLD, local_vef_count[2], element_export_ids,
            element_export_ranks);

        faces_ptr.resize(face_halo.numLocal() + face_halo.numGhost());
        faces.resize(face_halo.numLocal() + face_halo.numGhost());

        Cabana::gather(face_halo, faces_ptr);

        // Copy data to host
        // if (rank == 0)
        // {
        //     Cabana::deep_copy(faces, faces_ptr);
        //     Cabana::deep_copy(edges, edges_ptr);
        //     Cabana::deep_copy(vertices, vertices_ptr);
        // }

    }

    /**
     * Verify the faces in fin were refined corrrectly
     */
    template <class HostView_t>
    void verifyRefinement(HostView_t fids)
    {
        int size = fids.extent(0);
        Kokkos::View<int*, MemorySpace> fids_d("fids_d", size);
        Kokkos::deep_copy(fids_d, fids);
        numesh->refine(fids_d);

        //gatherAndCopyToHost();
    }
};

} // end namespace NuMeshTest

#endif // _TSTMESH2D_HPP_
