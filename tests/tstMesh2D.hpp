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
    int periodic_;
    std::shared_ptr<numesh_t> numesh = NuMesh::createEmptyMesh<ExecutionSpace, MemorySpace>(MPI_COMM_WORLD);
    v_array_type vertices;
    e_array_type edges;
    f_array_type faces;
    // l = local, g = ghost
    int lv = -1, le = -1, lf = -1, gv = -1, ge = -1, gf = -1;

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
        periodic_ = periodic;

        std::array<int, 2> global_num_cell = { mesh_size, mesh_size };
        std::array<double, 2> global_low_corner = { -1.0, -1.0 };
        std::array<double, 2> global_high_corner = { 1.0, 1.0 };
        // std::array<double, 6> global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
        std::array<bool, 2> periodic_a = { (bool)periodic_, (bool)periodic_ };
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

        lv = vert_halo.numLocal(); gv = vert_halo.numGhost();
        vertices_ptr.resize(lv + gv);
        vertices.resize(lv + gv);

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

        le = edge_halo.numLocal(); ge = edge_halo.numGhost();
        edges_ptr.resize(le + ge);
        edges.resize(le + ge);

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

        lf = face_halo.numLocal(); gf = face_halo.numGhost();
        faces_ptr.resize(lf + gf);
        faces.resize(lf + gf);

        Cabana::gather(face_halo, faces_ptr);

        // Copy data to host
        // Rank 0 holds the entire mesh
        Cabana::deep_copy(faces, faces_ptr);
        Cabana::deep_copy(edges, edges_ptr);
        Cabana::deep_copy(vertices, vertices_ptr);
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

        gatherAndCopyToHost();

        checkGIDSpaceBounds();
        checkGIDSpaceGaps();        // Performed on entire mesh on Rank 0
        checkEdgeEndpoints();       // Performed on entire mesh on Rank 0
    }

    /**
     * Verify that GIDs are not out-of-bounds of owned GID space
     */
    void checkGIDSpaceBounds()
    {
        auto v_gid = Cabana::slice<V_GID>(vertices);
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto f_gid = Cabana::slice<F_GID>(faces);

        auto vef_gid_start = numesh->get_vef_gid_start();

        for (int i = 0; i < lv; i++)
        {
            int predicted_rank = -1;
            int gid = v_gid(i);
            int gid_start = vef_gid_start(rank_, 0);
            for (int r = 0; r < comm_size_; r++)
            {
                if (gid >= vef_gid_start(r, 0))
                {
                    predicted_rank = r;
                }
            }
            EXPECT_EQ(predicted_rank, rank_) << "Rank " << rank_ << " VGID space: [" << gid_start << ", "
                << gid_start+lv << "), VLID: " << i << ", VGID: " << gid << "\n";
        }
        for (int i = 0; i < le; i++)
        {
            int predicted_rank = -1;
            int gid = e_gid(i);
            int gid_start = vef_gid_start(rank_, 1);
            
            for (int r = 0; r < comm_size_; r++)
            {
                if (gid >= vef_gid_start(r, 1))
                {
                    predicted_rank = r;
                }
            }
            EXPECT_EQ(predicted_rank, rank_) << "Rank " << rank_ << " EGID space: [" << gid_start << ", "
                << gid_start+le << "), ELID: " << i << ", EGID: " << gid << "\n";
        }
        for (int i = 0; i < lf; i++)
        {
            int predicted_rank = -1;
            int gid = f_gid(i);
            int gid_start = vef_gid_start(rank_, 2);
            for (int r = 0; r < comm_size_; r++)
            {
                if (gid >= vef_gid_start(r, 2))
                {
                    predicted_rank = r;
                }
            }
            EXPECT_EQ(predicted_rank, rank_) << "Rank " << rank_ << " FGID space: [" << gid_start << ", "
                << gid_start+lf << "), FLID: " << i << ", FGID: " << gid << "\n";
        }
    }

    /**
     * Verify there are no gaps in the GID space.
     * Performed on Rank 0
     */
    void checkGIDSpaceGaps()
    {
        if (rank_ != 0) return;

        auto v_gid = Cabana::slice<V_GID>(vertices);
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto f_gid = Cabana::slice<F_GID>(faces);

        // Sort by GID
        auto sort_verts = Cabana::sortByKey( v_gid );
        Cabana::permute( sort_verts, vertices );
        auto sort_edges = Cabana::sortByKey( e_gid );
        Cabana::permute( sort_edges, edges );
        auto sort_faces = Cabana::sortByKey( f_gid );
        Cabana::permute( sort_faces, faces );

        for (int i = 0; i < lv+gv; i++)
        {
            int gid = v_gid(i);
            EXPECT_EQ(i, gid) << "Rank " << rank_ << ": Gap in vertex GID space\n";
            
        }
        for (int i = 0; i < le+ge; i++)
        {
            int gid = e_gid(i);
            EXPECT_EQ(i, gid) << "Rank " << rank_ << ": Gap in edge GID space\n";
        }
        for (int i = 0; i < lf+gf; i++)
        {
            int gid = f_gid(i);
            EXPECT_EQ(i, gid) << "Rank " << rank_ << ": Gap in face GID space\n";
        }
    }

    /**
     * Verify that:
     *  1. Only one edge connects any two vertices
     *  2. All vertices are connected by an edge
     * Performed on Rank 0
     */
    void checkEdgeEndpoints()
    {
        if (rank_ != 0) return;

        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_gid = Cabana::slice<E_GID>(edges);

        int num_verts = lv + gv;
        Kokkos::View<int**, Kokkos::HostSpace> v2e("v2e", num_verts, num_verts);
        Kokkos::deep_copy(v2e, -1);

        // Check #1
        for (int i = 0; i < le+ge; i++)
        {
            std::vector<int> v = {e_vid(i, 0), e_vid(i, 1)};
            sort(v.begin(), v.end());
            if (v2e(v[0], v[1]) == -1)
            {
                v2e(v[0], v[1]) = e_gid(i);
            }
            else
            {
                FAIL() << "Edges " << v2e(v[0], v[1]) << " and " << e_gid(i) << " share vertices "
                    << v[0] << " and " << v[1] << "\n";
            }

        }

        // Check #2
        for (int i = 0; i < num_verts; i++)
        {
            int connected_edges = 0;
            for (int j = 0, j < num_verts; j++)
            {
                if (v2e(i, j) != -1)
                {
                    connected_edges++;
                }
            }
            if (periodic_)
            {
                // If the mesh is periodic, all vertices should have at
                // least 6 edges (more if a neighboring face has been refined)
                EXPECT_GE(connected_edges, 6) << "VGID " << v_gid(i) << " only has " << connected_edges << "edges\n";
            }
            else
            {
                // If the mesh is not periodic, check that the edge has at least
                // two vertices
            }
        }
    }
};

} // end namespace NuMeshTest

#endif // _TSTMESH2D_HPP_
