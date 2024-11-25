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
 * 
 * This test can only be run serially because the ID assignment
 * using GPUs is non-deterministic
 */
TYPED_TEST(Mesh2DTest, test0_refineAndAddEdges)
{
    this->test0_refineAndAddEdges();
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
