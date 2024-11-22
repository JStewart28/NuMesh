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
    using numesh_t = NuMesh::Mesh<ExecutionSpace, MemorySpace>;
    using edge_data = typename numesh_t::edge_data;
    using e_array_type = Cabana::AoSoA<edge_data, Kokkos::HostSpace, 4>;
    

  protected:
    int rank_, comm_size_;
    std::shared_ptr<numesh_t> numesh = NuMesh::createEmptyMesh<ExecutionSpace, MemorySpace>(MPI_COMM_WORLD);
    e_array_type edges;

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
        int gid;                                    // Global ID of test edge
        int cev0, cev1, tev0, tev1;                 // Vertices
        int cep, cec0, cec1, tep, tec0, tec1;       // Parents and children
        int cegid, tegid;                           // Global IDs
        int cer, ter;                               // Owning rank

        gid = Cabana::get<E_GID>(te);
        cev0 = Cabana::get<E_VIDS>(ce, 0); cev1 = Cabana::get<E_VIDS>(ce, 1);
        tev0 = Cabana::get<E_VIDS>(te, 0); tev1 = Cabana::get<E_VIDS>(te, 1);
        cep = Cabana::get<E_PID>(ce); tep = Cabana::get<E_PID>(te);
        cec0 = Cabana::get<E_CIDS>(ce, 0); cec1 = Cabana::get<E_CIDS>(ce, 1);
        tec0 = Cabana::get<E_CIDS>(te, 0); tec1 = Cabana::get<E_CIDS>(te, 1);
        cegid = Cabana::get<E_GID>(ce); tegid = Cabana::get<E_GID>(te);
        cer = Cabana::get<E_OWNER>(ce); ter = Cabana::get<E_OWNER>(te);

        EXPECT_EQ(cev0, tev0) << "TGID: " << gid << ", Vertex 0 mismatch";
        EXPECT_EQ(cev1, tev1) << "TGID: " << gid << ", Vertex 1 mismatch";
        EXPECT_EQ(cep, tep) << "TGID: " << gid << ", Parent edge mismatch";
        EXPECT_EQ(cec0, tec0) << "TGID: " << gid << ", Child edge 0 mismatch";
        EXPECT_EQ(cec1, tec1) << "TGID: " << gid << ", Child edge 1 mismatch";
        EXPECT_EQ(cegid, tegid) << "TGID: " << gid << ", Global ID mismatch";
        EXPECT_EQ(cer, ter) << "TGID: " << gid << ", Owning rank mismatch";
    }

    template <class FaceTuple>
    void tstFaceEqual(FaceTuple cf, FaceTuple tf)
    {
        int cfv0, cfv1, cfv2, tfv0, tfv1, tfv2;     // Vertices
        int cfe0, cfe1, cfe2, tfe0, tfe1, tfe2;     // Edges
        int cfgid, tfgid;                           // Global IDs
        int cfp, tfp, cfc0, tfc0, cfc1, tfc1, cfc2, tfc2, cfc3, tfc3; // Parents and children
        int cfr, tfr;                               // Owning rank

        cfv0 = Cabana::get<F_VIDS>(cf, 0); cfv1 = Cabana::get<F_VIDS>(cf, 1); cfv2 = Cabana::get<F_VIDS>(cf, 2);
        tfv0 = Cabana::get<F_VIDS>(tf, 0); tfv1 = Cabana::get<F_VIDS>(tf, 1); tfv2 = Cabana::get<F_VIDS>(tf, 2);
        cfe0 = Cabana::get<F_EIDS>(cf, 0); cfe1 = Cabana::get<F_EIDS>(cf, 1); cfe2 = Cabana::get<F_EIDS>(cf, 2);
        tfe0 = Cabana::get<F_EIDS>(tf, 0); tfe1 = Cabana::get<F_EIDS>(tf, 1); tfe2 = Cabana::get<F_EIDS>(tf, 2);
        cfgid = Cabana::get<F_GID>(cf); tfgid = Cabana::get<F_GID>(tf);
        cfp = Cabana::get<F_PID>(cf); tfp = Cabana::get<F_PID>(tf);
        cfc0 = Cabana::get<F_CID>(cf, 0); cfc1 = Cabana::get<F_CID>(cf, 1); cfc2 = Cabana::get<F_CID>(cf, 2); cfc3 = Cabana::get<F_CID>(cf, 3);
        tfc0 = Cabana::get<F_CID>(tf, 0); tfc1 = Cabana::get<F_CID>(tf, 1); tfc2 = Cabana::get<F_CID>(tf, 2); tfc3 = Cabana::get<F_CID>(tf, 3);
        cfr = Cabana::get<F_OWNER>(cf); tfr = Cabana::get<F_OWNER>(tf);

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
    
    std::string get_filename(int comm_size, int mesh_size, int periodic)
    {
        std::string filename = "../tests/data/edges_";
        filename += std::to_string(comm_size);
        filename += "_n";
        filename += std::to_string(mesh_size);
        if (periodic == 1) filename += "_periodic";
        else filename += "_free";
        filename += ".txt";
        return filename;
    }

    // Function to read edges from file and populate AoSoA using regex
    void readEdgesFromFile(const std::string& filename, e_array_type& edges)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        std::string line;

        // Define a regex pattern for the line format
        std::regex pattern(R"(^(\d+),\s*v\((\d+),\s*(\d+)\),\s*f\(([-\d]+),\s*([-\d]+)\),\s*c\(([-\d]+),\s*([-\d]+)\),\s*p\(([-\d]+)\)\s*(\d+),\s*\d+$)");

        while (std::getline(file, line))
        {
            std::smatch matches;
            if (std::regex_match(line, matches, pattern))
            {
                int gid = std::stoi(matches[1].str());
                int vids[2] = { std::stoi(matches[2].str()), std::stoi(matches[3].str()) };
                int fids[2] = { std::stoi(matches[4].str()), std::stoi(matches[5].str()) };
                int cids[2] = { std::stoi(matches[6].str()), std::stoi(matches[7].str()) };
                int pid = std::stoi(matches[8].str());
                int owner = std::stoi(matches[9].str());

                // Create and populate an edge tuple
                Cabana::Tuple<edge_data> edge_tuple;
                Cabana::get<E_GID>(edge_tuple) = gid;
                Cabana::get<E_VIDS>(edge_tuple, 0) = vids[0];
                Cabana::get<E_VIDS>(edge_tuple, 1) = vids[1];
                Cabana::get<E_FIDS>(edge_tuple, 0) = fids[0];
                Cabana::get<E_FIDS>(edge_tuple, 1) = fids[1];
                Cabana::get<E_CIDS>(edge_tuple, 0) = cids[0];
                Cabana::get<E_CIDS>(edge_tuple, 1) = cids[1];
                Cabana::get<E_PID>(edge_tuple) = pid;
                Cabana::get<E_OWNER>(edge_tuple) = owner;

                // Add the tuple to the AoSoA at the current gid
                edges.setTuple(gid, edge_tuple);
            }
            else
            {
                std::cerr << "Error parsing line: " << line << std::endl;
            }
        }
        file.close();
    }

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

    void refineEdges(int fin[10])
    {
        int size = 10;
        int counter = 0;
        int fin_cp[10];
        for (int i = 0; i < size; i++)
        {
            fin_cp[i] = fin[i];
            if (fin_cp[i] != -1) counter++;
        }

        Kokkos::View<int*, MemorySpace> fids("fids", counter);
        Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<ExecutionSpace>(0, size),
            KOKKOS_LAMBDA(int i) {
            
            if (fin_cp[i] == -1) return;
                
            if (i == 0) fids(i) = fin_cp[i];
                
            fids(i) = fin_cp[i];

        });
        numesh->refine(fids);
    }   

    void testEdges(e_array_type& c_edges)
    {
        e_array_type t_edges_host;
        int size = (int) numesh->edges().size();
        t_edges_host.resize(size);
        //printf("ce/te: %d, %d\n", (int)c_edges.size(), (int)t_edges_host.size());
        Cabana::deep_copy(t_edges_host, numesh->edges());
        for (int i = 0; i < size; i++)
        {
            auto t_edge = t_edges_host.getTuple(i);
            int gid = Cabana::get<E_GID>(t_edge);
            auto c_edge = c_edges.getTuple(gid);
            tstEdgeEqual(c_edge, t_edge);
        }
    }

    void test1_refineAndAddEdges()
    {
        e_array_type edges_host;
        int size = (int) numesh->edges().size();
        auto vef_gid_start = numesh->get_vef_gid_start();
        int min_vgid = vef_gid_start(this->rank_, 0);
        int max_vgid;
        if (this->rank_ == comm_size_ - 1) max_vgid = min_vgid + (int) numesh->vertices().size();
        else max_vgid = vef_gid_start(this->rank_+1, 0);
        edges_host.resize(size);
        Cabana::deep_copy(edges_host, numesh->edges());
        auto e_vids = Cabana::slice<E_VIDS>(edges_host);
        auto e_gids = Cabana::slice<E_GID>(edges_host);
        auto e_layer = Cabana::slice<E_LAYER>(edges_host);
        for (int i = 0; i < size; i ++)
        {
            int e_gid = e_gids(i);

            /**
             * On child edges, check that no edge connects vertices greater than the max locally owned vertex GID
             * 
             * Level zero edges may connect accoss MPI boundaries and violate this condition
             */ 
            if (e_layer(i) > 0)
            {
                for (int j = 0; j < 2; j++)
                {
                    int vid = e_vids(i, j);
                    EXPECT_GE(vid, min_vgid) << "Rank " << this->rank_ << ": Vertex " << j << " of edge " << e_gid << " : " << vid << " !>= " << min_vgid;
                    EXPECT_LT(vid, max_vgid) << "Rank " << this->rank_ << ": Vertex " << j << " of edge " << e_gid << " : " << vid << " !< " << max_vgid;
                }
            }
        }
    }
};

} // end namespace NuMeshTest

#endif // _TSTMESH2D_HPP_
