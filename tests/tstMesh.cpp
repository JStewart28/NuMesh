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

    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    this->performRefinement(fin);

    this->verifyRefinement(v+5, e+16, f+8);
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

    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    this->performRefinement(fin);

    this->verifyRefinement(v+e, e+e*2+f*3, f+f*4);
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

    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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
    this->verifyRefinement(v+3*e+3*f, 7*e+21*f, 21*f);
}

/************************************************************************
 ************************   Sphere mesh tests   ************************* 
 ***********************************************************************/

/**
 * Tests refinement of two neighoring faces
 */
TYPED_TEST(MeshTest, sphere_test0_refinement0)
{
    if (this->comm_size_ != 4)
    {
        printf("sphere_test0_refinement: only communicator size of 4 is supported.\n");
        // Otherwise the faces will not be neighbors based on their GIDs
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test0_refinement: Initialization error.\n");
        return;
    }

    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::View<int[2], Kokkos::HostSpace> fin("fin");
    fin(0) = 166; 
    fin(1) = 140;

    this->performRefinement(fin);
    
    this->verifyRefinement(v+5, e+16, f+8);
}

/**
 * Tests refinement of two neighoring faces, then refining
 * one of their child faces
 */
TYPED_TEST(MeshTest, sphere_test0_refinement1)
{
    if (this->comm_size_ != 4)
    {
        printf("sphere_test0_refinement: only communicator size of 4 is supported.\n");
        // Otherwise the faces will not be neighbors based on their GIDs
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test0_refinement: Initialization error.\n");
        return;
    }

    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    Kokkos::View<int[2], Kokkos::HostSpace> fin1("fin1");
    fin1(0) = 166; 
    fin1(1) = 140;
    this->performRefinement(fin1);

    Kokkos::View<int[1], Kokkos::HostSpace> fin2("fin2");
    fin2(0) = 150;
    this->performRefinement(fin2);

    // this->mesh_->printFaces(0, 0);
    this->verifyRefinement(v+5+3, e+16+9, f+8+4);
}

/**
 * Tests one iteration of uniform refinement
 */
TYPED_TEST(MeshTest, sphere_test1_refinement)
{
    if ((this->comm_size_ != 4) && (this->comm_size_ != 16))
    {
        printf("sphere_test1_refinement: only communicator size of 4 or 16 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test1_refinement: Initialization error.\n");
        return;
    }

    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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

    this->verifyRefinement(v+e, e*3+f*3, f*5);
}

/**
 * Tests two iterations of uniform refinement
 */
TYPED_TEST(MeshTest, sphere_test2_refinement)
{
    if ((this->comm_size_ != 4) && (this->comm_size_ != 16))
    {
        printf("sphere_test2_refinement: only communicator size of 4 or 16 is supported.\n");
        return;
    }
    
    int result = this->init_from_file();

    if (result)
    {
        printf("sphere_test2_refinement: Initialization error.\n");
        return;
    }
    
    int vcount = this->mesh_->count(NuMesh::Own(), NuMesh::Vertex());
    int ecount = this->mesh_->count(NuMesh::Own(), NuMesh::Edge());
    int fcount = this->mesh_->count(NuMesh::Own(), NuMesh::Face());
    int v, e, f;
    MPI_Allreduce(&vcount, &v, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ecount, &e, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&fcount, &f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

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

    this->verifyRefinement(v+3*e+3*f, 7*e+21*f, 21*f);
}

} // end namespace NuMeshTest
