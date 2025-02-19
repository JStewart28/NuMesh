#include "gtest/gtest.h"

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include "tstMesh.hpp"
#include "tstDriver.hpp"

#include <mpi.h>

namespace NuMeshTest
{

TYPED_TEST_SUITE(MeshTest, DeviceTypes);

/**
 * Tests refinement of two neighoring faces
 */
TYPED_TEST(MeshTest, grid_test0_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init_from_grid(mesh_size, 1);

    Kokkos::View<int[2], Kokkos::HostSpace> fin("fin");
    fin(0) = 30; 
    fin(1) = 31;

    this->performRefinement(fin);
    this->verifyRefinement();
}

/**
 * Tests uniform refinement
 */
TYPED_TEST(MeshTest, grid_test1_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init_from_grid(mesh_size, 1);

    int num_local_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    auto vef_gid_start = this->mesh_->vef_gid_start();
    auto vef_gid_start_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vef_gid_start);
    int face_gid_start = vef_gid_start_h(this->rank_, 2);

    Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
    for (int i = 0; i < num_local_faces; i++)
    {
        fin(i) = face_gid_start + i;
    }

    this->performRefinement(fin);
    this->verifyRefinement();
}

/**
 * Tests two iterations of uniform refinement
 */
TYPED_TEST(MeshTest, grid_test2_refinement)
{
    int mesh_size = this->comm_size_ * 2;
    if (this->comm_size_ == 1)
    {
        mesh_size = 5;
    }
    
    this->init_from_grid(mesh_size, 1);

    for (int i = 0; i < 3; i++)
    {
        int num_local_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
        auto vef_gid_start = this->mesh_->vef_gid_start();
        auto vef_gid_start_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vef_gid_start);
        int face_gid_start = vef_gid_start_h(this->rank_, 2);

        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }

        this->performRefinement(fin);
    }

    this->verifyRefinement();
}

/************************************************************************
 ************************   Sphere mesh tests   ************************* 
 ***********************************************************************/

/**
 * Tests refinement of two neighoring faces
 */
TYPED_TEST(MeshTest, sphere_test0_refinement)
{
    if (this->comm_size_ != 4)
    {
        printf("sphere_test0_refinement: only communicator size of 4 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test0_refinement: Initialization error.\n");
        return;
    }

    Kokkos::View<int[2], Kokkos::HostSpace> fin("fin");
    fin(0) = 166; 
    fin(1) = 140;

    this->performRefinement(fin);
    this->verifyRefinement();
}

/**
 * Tests one iteration of uniform refinement
 */
TYPED_TEST(MeshTest, sphere_test1_refinement)
{
    if (this->comm_size_ != 4)
    {
        printf("sphere_test1_refinement: only communicator size of 4 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test1_refinement: Initialization error.\n");
        return;
    }

    for (int i = 0; i < 1; i++)
    {
        int num_local_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
        auto vef_gid_start = this->mesh_->vef_gid_start();
        auto vef_gid_start_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vef_gid_start);
        int face_gid_start = vef_gid_start_h(this->rank_, 2);

        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }

        this->performRefinement(fin);
    }

    this->verifyRefinement();
}

/**
 * Tests two iteration2 of uniform refinement
 */
TYPED_TEST(MeshTest, sphere_test2_refinement)
{
    if (this->comm_size_ != 4)
    {
        printf("sphere_test2_refinement: only communicator size of 4 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test2_refinement: Initialization error.\n");
        return;
    }

    for (int i = 0; i < 2; i++)
    {
        int num_local_faces = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
        auto vef_gid_start = this->mesh_->vef_gid_start();
        auto vef_gid_start_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vef_gid_start);
        int face_gid_start = vef_gid_start_h(this->rank_, 2);

        Kokkos::View<int*, Kokkos::HostSpace> fin("fin", num_local_faces);
        for (int i = 0; i < num_local_faces; i++)
        {
            fin(i) = face_gid_start + i;
        }

        this->performRefinement(fin);
    }

    this->verifyRefinement();
}

} // end namespace NuMeshTest
