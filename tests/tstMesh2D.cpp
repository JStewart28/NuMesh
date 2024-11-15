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
 * Check that refining interior faces, with no 
 * neighboring faces having been refined, works properly
 * with one process and 4 processes
 */
TYPED_TEST(Mesh2DTest, testInteriorRefine)
{
    std::string filename;
    int mesh_size = 8;

    filename = this->get_filename(this->comm_size_, mesh_size, 1);
    
    this->init(mesh_size, 1);

    int fin[10] = {106, 5, 75, 51, -1, -1, -1, -1, -1, -1};
    this->refineEdges(fin);

    this->edges.resize(this->numesh->edges().size());
    this->readEdgesFromFile(filename, this->edges);
    this->testEdges(this->edges);
}

} // end namespace NuMeshTest
