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

    // Helper functions
    template <class e_verts_slice>
    bool shareExactlyOneEndpoint(e_verts_slice verts, int elid0, int elid1)
    {
        int shared = 0;
        for (int i = 0; i < 2; i++)
        {
            int e0v = verts(elid0, i);
            for (int j = 0; j < 2; j++)
            {
                int e1v = verts(elid1, j);
                if (e0v == e1v) shared++;
            }
        }
        return (shared == 1);
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
        
        printf("R%d halo local/ghost: %d, %d, import/export: %d, %d\n", rank_,
            edge_halo.numLocal(), edge_halo.numGhost(),
            edge_halo.totalNumImport(), edge_halo.totalNumExport());


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
     * Refines the mesh
     */
    template <class HostView_t>
    void performRefinement(HostView_t fids)
    {
        int size = fids.extent(0);
        Kokkos::View<int*, MemorySpace> fids_d("fids_d", size);
        Kokkos::deep_copy(fids_d, fids);
        numesh->refine(fids_d);
    }

    /**
     * Verify the faces in fin were refined corrrectly
     */
    void verifyRefinement()
    {
        gatherAndCopyToHost();

        // The following tests are performed on the entire mesh on Rank 0
        checkGIDSpaceGaps();
        // checkEdgeUnique(2); 

        // These checks are performed on each rank
        // checkGIDSpaceBounds();
        // checkEdgeChildren();
        // checkFaceEdges();

              
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
            ASSERT_EQ(predicted_rank, rank_) << "Rank " << rank_ << " VGID space: [" << gid_start << ", "
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
            ASSERT_EQ(predicted_rank, rank_) << "Rank " << rank_ << " EGID space: [" << gid_start << ", "
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
            ASSERT_EQ(predicted_rank, rank_) << "Rank " << rank_ << " FGID space: [" << gid_start << ", "
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
        auto e_owner = Cabana::slice<E_OWNER>(edges);
        auto f_gid = Cabana::slice<F_GID>(faces);

        // Sort by GID
        auto sort_verts = Cabana::sortByKey( v_gid );
        Cabana::permute( sort_verts, vertices );
        auto sort_edges = Cabana::sortByKey( e_gid );
        //Cabana::permute( sort_edges, edges );
        auto sort_faces = Cabana::sortByKey( f_gid );
        Cabana::permute( sort_faces, faces );

        for (int i = 0; i < lv+gv; i++)
        {
            int gid = v_gid(i);
            ASSERT_EQ(i, gid) << "Rank " << rank_ << ": Gap in vertex GID space\n";
            
        }
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_children = Cabana::slice<E_CIDS>(edges);
        auto e_parent = Cabana::slice<E_PID>(edges);
        for (int i = 0; i < numesh->count(NuMesh::Own(), NuMesh::Edge()); i++)
        {
            printf("R%d: e%d, v(%d, %d, %d), c(%d, %d), p(%d) %d, %d\n", rank_,
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1), e_vid(i, 2),
                e_children(i, 0), e_children(i, 1),
                e_parent(i),
                e_owner(i), rank_);
        }
        for (int i = 0; i < le+ge; i++)
        {
            int gid = e_gid(i);
            //printf("R%d: egid.%d, elid.%d, owner: %d\n", rank_, gid, i, e_owner(i));
            //ASSERT_EQ(i, gid) << "Rank " << rank_ << ": Gap in edge GID space\n";
        }
        return;
        for (int i = 0; i < lf+gf; i++)
        {
            int gid = f_gid(i);
            ASSERT_EQ(i, gid) << "Rank " << rank_ << ": Gap in face GID space\n";
        }
    }

    /**
     * Verify that:
     *   1. Only one edge connects any two vertices
     *   2. Verify that all vertices are connected by 'x' edges
     *          - If periodic with uniform refinement all vertices should have 6 edges
     *          - If non-periodic or with partial refinement, some vertices could
     *                 have as few as 2 edges.
     * Performed on Rank 0
     */
    void checkEdgeUnique(int x)
    {
        if (rank_ != 0) return;

        auto v_gid = Cabana::slice<V_GID>(vertices);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_gid = Cabana::slice<E_GID>(edges);

        int num_verts = lv + gv;
        Kokkos::View<int**, Kokkos::HostSpace> v2e("v2e", num_verts, num_verts);
        Kokkos::deep_copy(v2e, -1);

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

        for (int i = 0; i < num_verts; i++)
        {
            int connected_edges = 0;
            for (int j = 0; j < num_verts; j++)
            {
                // printf("%3d ", v2e(i, j));
                if ((v2e(i, j) != -1) || (v2e(j, i) != -1))
                {
                    connected_edges++;
                }
            }
            // printf("\n");
            // All vertices should be connected by at least 2 edges
            ASSERT_GE(connected_edges, x) << "VGID " << v_gid(i) << " only has " << connected_edges << " edge\n";
        }
    }

    /**
     * Check that if an edge has children, those children share
     * a common vertex and have one endpoint on their parent edge.
     * 
     * Each process checks their local part of the mesh
     */
    void checkEdgeChildren()
    {
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_rank = Cabana::slice<E_OWNER>(edges);
        auto e_cid = Cabana::slice<E_CIDS>(edges);
        auto e_pid = Cabana::slice<E_PID>(edges);
        auto e_layer = Cabana::slice<E_LAYER>(edges);

        for (int i = 0; i < le+ge; i++)
        {
            // Skip edges without children
            if (e_cid(i, 0) == -1) continue;

            // Child edges
            int ce0, ce1;
            // Parent vertices
            int pv0, pv1, pvm;
            // Child vertices
            int c0v0, c0v1, c1v0, c1v1;

            ce0 = NuMesh::get_lid(e_gid, e_cid(i, 0), 0, edges.size()); ce1 = NuMesh::get_lid(e_gid, e_cid(i, 1), 0, edges.size());
            pv0 = e_vid(i, 0); pv1 = e_vid(i, 1); pvm = e_vid(i, 2);

            // Check child edge 0
            c0v0 = e_vid(ce0, 0); c0v1 = e_vid(ce0, 1);
            ASSERT_EQ(c0v0, pv0);
            ASSERT_EQ(c0v1, pvm);

            // Check child edge 1
            c1v0 = e_vid(ce1, 0); c1v1 = e_vid(ce1, 1);
            ASSERT_EQ(c1v0, pvm);
            ASSERT_EQ(c1v1, pv1);

            // Check child edge 0 v1 = child edge 1 v0
            ASSERT_EQ(c0v1, c1v0); 
        }
    }

    /**
     * Check for any face, you can follow its edges from its
     * first vertex and return to the first vertex
     */
    void checkFaceEdges()
    {
        if (rank_ != 0) return;
        auto f_cid = Cabana::slice<F_CID>(faces);
        auto f_gid = Cabana::slice<F_GID>(faces);
        auto f_eid = Cabana::slice<F_EIDS>(faces);
        auto f_pid = Cabana::slice<F_PID>(faces);
        auto f_layer = Cabana::slice<F_LAYER>(faces);
        auto f_owner = Cabana::slice<F_OWNER>(faces);

        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_rank = Cabana::slice<E_OWNER>(edges);
        auto e_cid = Cabana::slice<E_CIDS>(edges);
        auto e_pid = Cabana::slice<E_PID>(edges);
        auto e_layer = Cabana::slice<E_LAYER>(edges);

        for (int i = 0; i < lf+gf; i++)
        {
            int e0, e1, e2;
            e0 = NuMesh::get_lid(e_gid, f_eid(i, 0), 0, edges.size());
            e1 = NuMesh::get_lid(e_gid, f_eid(i, 1), 0, edges.size());
            e2 = NuMesh::get_lid(e_gid, f_eid(i, 2), 0, edges.size());

            ASSERT_TRUE(shareExactlyOneEndpoint(e_vid, e0, e1)) << "FGID " << f_gid(i) << "\n";
            ASSERT_TRUE(shareExactlyOneEndpoint(e_vid, e1, e2)) << "FGID " << f_gid(i) << "\n";
            ASSERT_TRUE(shareExactlyOneEndpoint(e_vid, e2, e0)) << "FGID " << f_gid(i) << "\n";
        }
    }
};

} // end namespace NuMeshTest

#endif // _TSTMESH2D_HPP_
