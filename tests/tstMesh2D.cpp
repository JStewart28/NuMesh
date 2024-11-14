#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstMesh2D.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(Mesh2DTest, DeviceTypes);

/**
 * Check that refining a single interior face, with no 
 * neighboring faces having been refined, works properly
 */
TYPED_TEST(Mesh2DTest, testInteriorRefine)
{
    std::string filename;
    int mesh_size;
    if (this->comm_size_ == 1)
    {
        this->edges.resize(229);
        mesh_size = 8;
    }
    if (this->comm_size_ == 4)
    {
        this->edges.resize(229);
        mesh_size = 8;
    }
    filename = this->get_filename(this->comm_size_, mesh_size, 1);
     
    this->readEdgesFromFile(filename, this->edges);
    this->testEdges(this->edges, mesh_size);
}

} // end namespace NuMeshTest
