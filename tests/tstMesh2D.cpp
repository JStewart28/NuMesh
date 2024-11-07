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
TYPED_TEST(Mesh2DTest, testSingleInteriorRefine)
{
    this->testSingleInteriorRefineFunc();
}

} // end namespace NuMeshTest
