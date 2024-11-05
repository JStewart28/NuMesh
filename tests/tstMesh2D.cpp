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
 * Check that file IO works properly before running any other tests.
 * Run the low-order solve, write the results, then read the results and
 * compare them to the in-memory results.
 */
TYPED_TEST(Mesh2DTest, testFileIO)
{
    // ClArgs cl;
    // init_default_ClArgs(cl);

    // // Adjust command-line args for this test
    // cl.num_nodes = {64, 64};
    // cl.boundary = Beatnik::MeshBoundaryType::PERIODIC;
    // cl.params.solver_order = SolverOrder::ORDER_LOW;
    // finalize_ClArgs(cl);

    // // Run rocketrig
    // this->init(cl);
    // this->rg_->rocketrig();
    // auto z = this->rg_->template get_positions<Cabana::Grid::Node>();
    // auto w = this->rg_->template get_vorticities<Cabana::Grid::Node>();

    // // Write views
    // int mesh_size = cl.num_nodes[0];
    // int periodic = !(cl.boundary);
    // BeatnikTest::Utils::writeView(this->rank_, this->comm_size_, mesh_size, periodic, z);
    // BeatnikTest::Utils::writeView(this->rank_, this->comm_size_, mesh_size, periodic, w);

    // // Read views
    // std::string z_name = Utils::get_filename(this->rank_, this->comm_size_, mesh_size, periodic, 'z');
    // std::string w_name = Utils::get_filename(this->rank_, this->comm_size_, mesh_size, periodic, 'w');
    // auto z_file = this->read_z(z_name);
    // auto w_file = this->read_w(w_name);

    // // Compare views
    // this->compare_views(z, z_file);
    // this->compare_views(w, w_file);

    // // Remove view files
    // this->remove_view_files();
}

} // end namespace NuMeshTest
