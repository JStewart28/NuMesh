#ifndef _TSTMESH2D_HPP_
#define _TSTMESH2D_HPP_

#include <iostream>
#include <filesystem>
#include <regex>

#include "gtest/gtest.h"

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
    using View_t = Kokkos::View<double***, Kokkos::HostSpace>;

  protected:
    int rank_, comm_size_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size_);
    }

    void TearDown() override
    { 
    }

    template <class EdgeTuple>
    void tstEdgeEqual(EdgeTuple ce, EdgeTuple te)
    {
        int cev0, cev1, tev0, tev1;                 // Vertices
        int cep, cec0, cec1, tep, tec0, tec1;       // Parents and children
        int cegid, tegid;                           // Global IDs
        int cer, ter;                               // Owning rank

        cev0 = Cabana::get<S_E_VIDS>(ce, 0); cev1 = Cabana::get<S_E_VIDS>(ce, 1);
        tev0 = Cabana::get<S_E_VIDS>(te, 0); tev1 = Cabana::get<S_E_VIDS>(te, 1);
        cep = Cabana::get<S_E_PID>(ce); tep = Cabana::get<S_E_PID>(te);
        cec0 = Cabana::get<S_E_CIDS>(ce, 0); cec1 = Cabana::get<S_E_CIDS>(ce, 1);
        tec0 = Cabana::get<S_E_CIDS>(te, 0); tec1 = Cabana::get<S_E_CIDS>(te, 1);
        cegid = Cabana::get<S_E_GID>(ce); tegid = Cabana::get<S_E_GID>(te);
        cer = Cabana::get<S_E_OWNER>(ce); ter = Cabana::get<S_E_OWNER>(te);

        EXPECT_EQ(cev0, tev0) << "Vertex 0 mismatch";
        EXPECT_EQ(cev1, tev1) << "Vertex 1 mismatch";
        EXPECT_EQ(cep, tep) << "Parent edge mismatch";
        EXPECT_EQ(cec0, tec0) << "Child edge 0 mismatch";
        EXPECT_EQ(cec1, tec1) << "Child edge 1 mismatch";
        EXPECT_EQ(cegid, tegid) << "Global ID mismatch";
        EXPECT_EQ(cer, ter) << "Owning rank mismatch";
    }

    template <class FaceTuple>
    void tstFaceEqual(FaceTuple cf, FaceTuple tf)
    {
        int cfv0, cfv1, cfv2, tfv0, tfv1, tfv2;     // Vertices
        int cfe0, cfe1, cfe2, tfe0, tfe1, tfe2;     // Edges
        int cfgid, tfgid;                           // Global IDs
        int cfp, tfp, cfc0, tfc0, cfc1, tfc1, cfc2, tfc2, cfc3, tfc3; // Parents and children
        int cfr, tfr;                               // Owning rank

        cfv0 = Cabana::get<S_F_VIDS>(cf, 0); cfv1 = Cabana::get<S_F_VIDS>(cf, 1); cfv2 = Cabana::get<S_F_VIDS>(cf, 2);
        tfv0 = Cabana::get<S_F_VIDS>(tf, 0); tfv1 = Cabana::get<S_F_VIDS>(tf, 1); tfv2 = Cabana::get<S_F_VIDS>(tf, 2);
        cfe0 = Cabana::get<S_F_EIDS>(cf, 0); cfe1 = Cabana::get<S_F_EIDS>(cf, 1); cfe2 = Cabana::get<S_F_EIDS>(cf, 2);
        tfe0 = Cabana::get<S_F_EIDS>(tf, 0); tfe1 = Cabana::get<S_F_EIDS>(tf, 1); tfe2 = Cabana::get<S_F_EIDS>(tf, 2);
        cfgid = Cabana::get<S_F_GID>(cf); tfgid = Cabana::get<S_F_GID>(tf);
        cfp = Cabana::get<S_F_PID>(cf); tfp = Cabana::get<S_F_PID>(tf);
        cfc0 = Cabana::get<S_F_CID>(cf, 0); cfc1 = Cabana::get<S_F_CID>(cf, 1); cfc2 = Cabana::get<S_F_CID>(cf, 2); cfc3 = Cabana::get<S_F_CID>(cf, 3);
        tfc0 = Cabana::get<S_F_CID>(tf, 0); tfc1 = Cabana::get<S_F_CID>(tf, 1); tfc2 = Cabana::get<S_F_CID>(tf, 2); tfc3 = Cabana::get<S_F_CID>(tf, 3);
        cfr = Cabana::get<S_F_OWNER>(cf); tfr = Cabana::get<S_F_OWNER>(tf);

        EXPECT_EQ(cfv0, tfv0) << "Vertex 0 mismatch";
        EXPECT_EQ(cfv1, tfv1) << "Vertex 1 mismatch";
        EXPECT_EQ(cfv2, tfv2) << "Vertex 2 mismatch";
        EXPECT_EQ(cfe0, tfe0) << "Edge 0 mismatch";
        EXPECT_EQ(cfe1, tfe1) << "Edge 1 mismatch";
        EXPECT_EQ(cfe2, tfe2) << "Edge 2 mismatch";
        EXPECT_EQ(cfgid, tfgid) << "Global ID mismatch";
        EXPECT_EQ(cfp, tfp) << "Parent face mismatch";
        EXPECT_EQ(cfc0, tfc0) << "Child face 0 mismatch";
        EXPECT_EQ(cfc1, tfc1) << "Child face 1 mismatch";
        EXPECT_EQ(cfc2, tfc2) << "Child face 2 mismatch";
        EXPECT_EQ(cfc3, tfc3) << "Child face 3 mismatch";
        EXPECT_EQ(cfr, tfr) << "Owning rank mismatch";
    }

  public:
    void testSingleInteriorRefineFunc()
    {
        int mesh_size;
        if (comm_size_ == 1)
        {
            mesh_size = 5;
        }
        else
        {
            printf("Unsupported communicator size. Supported sizes are: 1. Skipping testSingleInteriorRefine\n");
        }

        std::array<int, 2> global_num_cell = { mesh_size, mesh_size };
        std::array<double, 2> global_low_corner = { -1.0, -1.0 };
        std::array<double, 2> global_high_corner = { 1.0, 1.0 };
        // std::array<double, 6> global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
        std::array<bool, 2> periodic = { true, true };
        Cabana::Grid::DimBlockPartitioner<2> partitioner;

        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        auto global_grid = Cabana::Grid::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, periodic, partitioner );
        int halo_width = 2;
        auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        auto numesh = NuMesh::createEmptyMesh<ExecutionSpace, MemorySpace>(MPI_COMM_WORLD);

        auto layout = Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Node());
        auto array = Cabana::Grid::createArray<double, MemorySpace>("for_initialization", layout);
        numesh->initializeFromArray(*array);

        if (comm_size_ == 1)
        {
            int size = 1;
            Kokkos::View<int*, MemorySpace> fids("fids", size);
            Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<ExecutionSpace>(0, size),
                KOKKOS_LAMBDA(int i) {
                    
                if (i == 0)
                {
                    fids(i) = 12;
                }

            });
            numesh->refine(fids);
            auto edges = numesh->edges();
            auto faces = numesh->faces();
            
            // Edge 18
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge18;
            Cabana::get<S_E_VIDS>(edge18, 0) = 6; 
            Cabana::get<S_E_VIDS>(edge18, 1) = 7;
            Cabana::get<S_E_PID>(edge18) = -1;
            Cabana::get<S_E_CIDS>(edge18, 0) = 78; 
            Cabana::get<S_E_CIDS>(edge18, 1) = 81;
            Cabana::get<S_E_GID>(edge18) = 18;
            Cabana::get<S_E_OWNER>(edge18) = 0;
            auto te18 = edges.getTuple(18);
            tstEdgeEqual(edge18, te18);

            // Edge 19
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge19;
            Cabana::get<S_E_VIDS>(edge19, 0) = 6;
            Cabana::get<S_E_VIDS>(edge19, 1) = 12;
            Cabana::get<S_E_PID>(edge19) = -1; 
            Cabana::get<S_E_CIDS>(edge19, 0) = 80;
            Cabana::get<S_E_CIDS>(edge19, 1) = 83;
            Cabana::get<S_E_GID>(edge19) = 19;
            Cabana::get<S_E_OWNER>(edge19) = 0;
            auto te19 = edges.getTuple(19);
            tstEdgeEqual(edge19, te19);

            // Edge 23
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge23;
            Cabana::get<S_E_VIDS>(edge23, 0) = 7; 
            Cabana::get<S_E_VIDS>(edge23, 1) = 12;
            Cabana::get<S_E_PID>(edge23) = -1;
            Cabana::get<S_E_CIDS>(edge23, 0) = 79; 
            Cabana::get<S_E_CIDS>(edge23, 1) = 82;
            Cabana::get<S_E_GID>(edge23) = 23;
            Cabana::get<S_E_OWNER>(edge23) = 0;
            auto te23 = edges.getTuple(23);
            tstEdgeEqual(edge23, te23);

            // Edge 75
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge75;
            Cabana::get<S_E_VIDS>(edge75, 0) = 25; 
            Cabana::get<S_E_VIDS>(edge75, 1) = 26;
            Cabana::get<S_E_PID>(edge75) = -1;
            Cabana::get<S_E_CIDS>(edge75, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge75, 1) = -1;
            Cabana::get<S_E_GID>(edge75) = 75;
            Cabana::get<S_E_OWNER>(edge75) = 0;
            auto te75 = edges.getTuple(75);
            tstEdgeEqual(edge75, te75);

            // Edge 76
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge76;
            Cabana::get<S_E_VIDS>(edge76, 0) = 26; 
            Cabana::get<S_E_VIDS>(edge76, 1) = 27;
            Cabana::get<S_E_PID>(edge76) = -1;
            Cabana::get<S_E_CIDS>(edge76, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge76, 1) = -1;
            Cabana::get<S_E_GID>(edge76) = 76;
            Cabana::get<S_E_OWNER>(edge76) = 0;
            auto te76 = edges.getTuple(76);
            tstEdgeEqual(edge76, te76);

            // Edge 77
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge77;
            Cabana::get<S_E_VIDS>(edge77, 0) = 25; 
            Cabana::get<S_E_VIDS>(edge77, 1) = 27;
            Cabana::get<S_E_PID>(edge77) = -1;
            Cabana::get<S_E_CIDS>(edge77, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge77, 1) = -1;
            Cabana::get<S_E_GID>(edge77) = 77;
            Cabana::get<S_E_OWNER>(edge77) = 0;
            auto te77 = edges.getTuple(77);
            tstEdgeEqual(edge77, te77);

            // Edge 78
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge78;
            Cabana::get<S_E_VIDS>(edge78, 0) = 6; 
            Cabana::get<S_E_VIDS>(edge78, 1) = 25;
            Cabana::get<S_E_PID>(edge78) = 18;
            Cabana::get<S_E_CIDS>(edge78, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge78, 1) = -1;
            Cabana::get<S_E_GID>(edge78) = 78;
            Cabana::get<S_E_OWNER>(edge78) = 0;
            auto te78 = edges.getTuple(78);
            tstEdgeEqual(edge78, te78);

            // Edge 79
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge79;
            Cabana::get<S_E_VIDS>(edge79, 0) = 7; 
            Cabana::get<S_E_VIDS>(edge79, 1) = 26;
            Cabana::get<S_E_PID>(edge79) = 23;
            Cabana::get<S_E_CIDS>(edge79, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge79, 1) = -1;
            Cabana::get<S_E_GID>(edge79) = 79;
            Cabana::get<S_E_OWNER>(edge79) = 0;
            auto te79 = edges.getTuple(79);
            tstEdgeEqual(edge79, te79);

            // Edge 80
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge80;
            Cabana::get<S_E_VIDS>(edge80, 0) = 12; 
            Cabana::get<S_E_VIDS>(edge80, 1) = 27;
            Cabana::get<S_E_PID>(edge80) = 19;
            Cabana::get<S_E_CIDS>(edge80, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge80, 1) = -1;
            Cabana::get<S_E_GID>(edge80) = 80;
            Cabana::get<S_E_OWNER>(edge80) = 0;
            auto te80 = edges.getTuple(80);
            tstEdgeEqual(edge80, te80);

            // Edge 81
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge81;
            Cabana::get<S_E_VIDS>(edge81, 0) = 25; 
            Cabana::get<S_E_VIDS>(edge81, 1) = 7;
            Cabana::get<S_E_PID>(edge81) = 18;
            Cabana::get<S_E_CIDS>(edge81, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge81, 1) = -1;
            Cabana::get<S_E_GID>(edge81) = 81;
            Cabana::get<S_E_OWNER>(edge81) = 0;
            auto te81 = edges.getTuple(81);
            tstEdgeEqual(edge81, te81);

            // Edge 82
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge82;
            Cabana::get<S_E_VIDS>(edge82, 0) = 26; 
            Cabana::get<S_E_VIDS>(edge82, 1) = 12;
            Cabana::get<S_E_PID>(edge82) = 23;
            Cabana::get<S_E_CIDS>(edge82, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge82, 1) = -1;
            Cabana::get<S_E_GID>(edge82) = 82;
            Cabana::get<S_E_OWNER>(edge82) = 0;
            auto te82 = edges.getTuple(82);
            tstEdgeEqual(edge82, te82);

            // Edge 83
            Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::edge_data> edge83;
            Cabana::get<S_E_VIDS>(edge83, 0) = 27; 
            Cabana::get<S_E_VIDS>(edge83, 1) = 6;
            Cabana::get<S_E_PID>(edge83) = 19;
            Cabana::get<S_E_CIDS>(edge83, 0) = -1; 
            Cabana::get<S_E_CIDS>(edge83, 1) = -1;
            Cabana::get<S_E_GID>(edge83) = 83;
            Cabana::get<S_E_OWNER>(edge83) = 0;
            auto te83 = edges.getTuple(83);
            tstEdgeEqual(edge83, te83);

            /* Faces */
            // // Face 50
            // Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::face_data> face50;
            // Cabana::get<S_F_VIDS>(face50, 0) = 6;
            // Cabana::get<S_F_VIDS>(face50, 1) = 25;
            // Cabana::get<S_F_VIDS>(face50, 2) = 27;
            // Cabana::get<S_F_EIDS>(face50, 0) = 77;
            // Cabana::get<S_F_EIDS>(face50, 1) = 78;
            // Cabana::get<S_F_EIDS>(face50, 2) = 83;
            // Cabana::get<S_F_GID>(face50) = 50;
            // Cabana::get<S_F_CID>(face50, 0) = -1;
            // Cabana::get<S_F_CID>(face50, 1) = -1;
            // Cabana::get<S_F_CID>(face50, 2) = -1;
            // Cabana::get<S_F_CID>(face50, 3) = -1;
            // Cabana::get<S_F_PID>(face50) = 12;
            // Cabana::get<S_F_OWNER>(face50) = 0;
            // auto tf50 = faces.getTuple(50);
            // tstFaceEqual(face50, tf50);

            // // Face 51
            // Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::face_data> face51;
            // Cabana::get<S_F_VIDS>(face51, 0) = 25;
            // Cabana::get<S_F_VIDS>(face51, 1) = 7;
            // Cabana::get<S_F_VIDS>(face51, 2) = 26;
            // Cabana::get<S_F_EIDS>(face51, 0) = 75;
            // Cabana::get<S_F_EIDS>(face51, 1) = 79;
            // Cabana::get<S_F_EIDS>(face51, 2) = 81;
            // Cabana::get<S_F_GID>(face51) = 51;
            // Cabana::get<S_F_CID>(face51, 0) = -1;
            // Cabana::get<S_F_CID>(face51, 1) = -1;
            // Cabana::get<S_F_CID>(face51, 2) = -1;
            // Cabana::get<S_F_CID>(face51, 3) = -1;
            // Cabana::get<S_F_PID>(face51) = 12;
            // Cabana::get<S_F_OWNER>(face51) = 0;
            // auto tf51 = faces.getTuple(51);
            // tstFaceEqual(face51, tf51);

            // // Face 52
            // Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::face_data> face52;
            // Cabana::get<S_F_VIDS>(face52, 0) = 27;
            // Cabana::get<S_F_VIDS>(face52, 1) = 26;
            // Cabana::get<S_F_VIDS>(face52, 2) = 12;
            // Cabana::get<S_F_EIDS>(face52, 0) = 76;
            // Cabana::get<S_F_EIDS>(face52, 1) = 80;
            // Cabana::get<S_F_EIDS>(face52, 2) = 82;
            // Cabana::get<S_F_GID>(face52) = 52;
            // Cabana::get<S_F_CID>(face52, 0) = -1;
            // Cabana::get<S_F_CID>(face52, 1) = -1;
            // Cabana::get<S_F_CID>(face52, 2) = -1;
            // Cabana::get<S_F_CID>(face52, 3) = -1;
            // Cabana::get<S_F_PID>(face52) = 12;
            // Cabana::get<S_F_OWNER>(face52) = 0;
            // auto tf52 = faces.getTuple(52);
            // tstFaceEqual(face52, tf52);

            // // Face 53
            // Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::face_data> face53;
            // Cabana::get<S_F_VIDS>(face53, 0) = 25;
            // Cabana::get<S_F_VIDS>(face53, 1) = 26;
            // Cabana::get<S_F_VIDS>(face53, 2) = 27;
            // Cabana::get<S_F_EIDS>(face53, 0) = 75;
            // Cabana::get<S_F_EIDS>(face53, 1) = 76;
            // Cabana::get<S_F_EIDS>(face53, 2) = 77;
            // Cabana::get<S_F_GID>(face53) = 53;
            // Cabana::get<S_F_CID>(face53, 0) = -1;
            // Cabana::get<S_F_CID>(face53, 1) = -1;
            // Cabana::get<S_F_CID>(face53, 2) = -1;
            // Cabana::get<S_F_CID>(face53, 3) = -1;
            // Cabana::get<S_F_PID>(face53) = 12;
            // Cabana::get<S_F_OWNER>(face53) = 0;
            // auto tf53 = faces.getTuple(53);
            // tstFaceEqual(face53, tf53);

            // // Face 12
            // Cabana::Tuple<typename NuMesh::Mesh<ExecutionSpace, MemorySpace>::face_data> face12;
            // Cabana::get<S_F_VIDS>(face12, 0) = 6;
            // Cabana::get<S_F_VIDS>(face12, 1) = 7;
            // Cabana::get<S_F_VIDS>(face12, 2) = 12;
            // Cabana::get<S_F_EIDS>(face12, 0) = 18;
            // Cabana::get<S_F_EIDS>(face12, 1) = 23;
            // Cabana::get<S_F_EIDS>(face12, 2) = 19;
            // Cabana::get<S_F_GID>(face12) = 12;
            // Cabana::get<S_F_CID>(face12, 0) = 50;
            // Cabana::get<S_F_CID>(face12, 1) = 51;
            // Cabana::get<S_F_CID>(face12, 2) = 52;
            // Cabana::get<S_F_CID>(face12, 3) = 53;
            // Cabana::get<S_F_PID>(face12) = -1;
            // Cabana::get<S_F_OWNER>(face12) = 0;
            // auto tf12 = faces.getTuple(12);
            // tstFaceEqual(face12, tf12);
        }

    }
};

} // end namespace NuMeshTest

#endif // _TSTMESH2D_HPP_
