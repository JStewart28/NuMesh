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
 * Tests _refineAndAddEdges using a known correct output
 * from a file
 */
TYPED_TEST(Mesh2DTest, test_refineAndAddEdges0)
{
    std::string filename;
    int mesh_size = 8;

    filename = this->get_filename(this->comm_size_, mesh_size, 1);
    
    this->init(mesh_size, 1);

    int fin[10] = {106, 5, 75, 51, -1, -1, -1, -1, -1, -1};
    this->refineEdges(fin);

    // Make edges slightly larger than needed because it's easier
    this->edges.resize(this->numesh->edges().size()*(this->comm_size_+1));
    this->readEdgesFromFile(filename, this->edges);
    //this->testEdges(this->edges);
}

/**
 * Tests _refineAndAddEdges for general correctness refining internal edges:
 *  - No edge connects vertices greater than the max locally owned vertex GID
 *      - Note: This only holds true when refining internal edges
 *  - No edges with different IDs connect the same vertices
 *  - 
 */
TYPED_TEST(Mesh2DTest, test1_refineAndAddInternalEdges)
{
    int mesh_size = 8;
    
    this->init(mesh_size, 1);
    
    this->test1_refineAndAddInternalEdges();

}
} // end namespace NuMeshTest
