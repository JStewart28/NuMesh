#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstHalo.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace TesseraTest
{

TYPED_TEST_SUITE(HaloTest, DeviceTypes);

/**
 * Tests that halo of depth 1 works without any refinement
 */
TYPED_TEST(HaloTest, grid_test_halo_depth_1_no_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init_from_grid(mesh_size, 1);

    this->test_halo_depth_1(true);
}

/**
 * Tests that halo of depth 1 works with one layer of
 * uniform refinement
 */
TYPED_TEST(HaloTest, grid_test_halo_depth_1_uniform_refinement1)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init_from_grid(mesh_size, 1);

    auto vef_gid_start = this->mesh_->vef_gid_start();

    // Uniform refinement
    for (int i = 0; i < 1; i++)
    {
        int num_local_faces = this->mesh_->count(Tessera::Own(), Tessera::Face());
        int face_gid_start = vef_gid_start(this->rank_, 2);
        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }
        this->performRefinement(fin);
    }

    this->test_halo_depth_1(true);
}

/**
 * Tests that halo of depth 1 works with two layers of
 * uniform refinement
 */
TYPED_TEST(HaloTest, grid_test_halo_depth_1_uniform_refinement2)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init_from_grid(mesh_size, 1);

    auto vef_gid_start = this->mesh_->vef_gid_start();

    // Uniform refinement
    for (int i = 0; i < 2; i++)
    {
        int num_local_faces = this->mesh_->count(Tessera::Own(), Tessera::Face());
        int face_gid_start = vef_gid_start(this->rank_, 2);
        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }
        this->performRefinement(fin);
    }

    this->test_halo_depth_1(true);
}

/************************************************************************
 ************************   Sphere mesh tests   ************************* 
 ***********************************************************************/

/**
 * Tests that halo of depth 1 works without any refinement
 */
TYPED_TEST(HaloTest, sphere_test_halo_depth_1_no_refinement)
{
    if ((this->comm_size_ != 4) && (this->comm_size_ != 16))
    {
        printf("sphere_test0_refinement: only communicator size of 4 or 16 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test0_refinement: Initialization error.\n");
        return;
    }

    this->test_halo_depth_1(false);
}

/**
 * Tests that halo of depth 1 works with one layer of
 * uniform refinement
 */
TYPED_TEST(HaloTest, sphere_test_halo_depth_1_uniform_refinement1)
{
    if ((this->comm_size_ != 4) && (this->comm_size_ != 16))
    {
        printf("sphere_test0_refinement: only communicator size of 4 or 16 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test0_refinement: Initialization error.\n");
        return;
    }

    auto vef_gid_start = this->mesh_->vef_gid_start();

    // Uniform refinement
    for (int i = 0; i < 1; i++)
    {
        int num_local_faces = this->mesh_->count(Tessera::Own(), Tessera::Face());
        int face_gid_start = vef_gid_start(this->rank_, 2);
        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }
        this->performRefinement(fin);
    }

    this->test_halo_depth_1(false);
}

/**
 * Tests that halo of depth 1 works with two layers of
 * uniform refinement
 */
TYPED_TEST(HaloTest, sphere_test_halo_depth_1_uniform_refinement2)
{
    if ((this->comm_size_ != 4) && (this->comm_size_ != 16))
    {
        printf("sphere_test0_refinement: only communicator size of 4 or 16 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test0_refinement: Initialization error.\n");
        return;
    }

    auto vef_gid_start = this->mesh_->vef_gid_start();

    // Uniform refinement
    for (int i = 0; i < 2; i++)
    {
        int num_local_faces = this->mesh_->count(Tessera::Own(), Tessera::Face());
        int face_gid_start = vef_gid_start(this->rank_, 2);
        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }
        this->performRefinement(fin);
    }

    this->test_halo_depth_1(1);
}

 
} // end namespace TesseraTest
