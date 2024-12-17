#ifndef _TESTING_UTILS_HPP_
#define _TESTING_UTILS_HPP_

namespace Utils
{

/**
 * Initialize a mesh from a Cabana local grid
 */
template <class MemorySpace, class mesh_t>
void init(mesh_t& mesh, int mesh_size, int periodic)
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
        mesh->initializeFromArray(*array);
    }


/**
 * Gather the entire mesh to rank 0 and copy to host memory
 */
template <class ExecutionSpace, class MemorySpace, class mesh_t, class v_t, class e_t, class f_t>
void gatherAndCopyToHost(mesh_t& mesh, v_t& vertices, e_t& edges, f_t& faces)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // l = local, g = ghost
    int lv = -1, le = -1, lf = -1, gv = -1, ge = -1, gf = -1;

    auto vertices_ptr = mesh->vertices();
    auto edges_ptr = mesh->edges();
    auto faces_ptr = mesh->faces();

        // Local counts for each rank
    int local_vef_count[3] = {mesh->count(NuMesh::Own(), NuMesh::Vertex()),
                            mesh->count(NuMesh::Own(), NuMesh::Edge()),
                            mesh->count(NuMesh::Own(), NuMesh::Face())};

    // Get vertices
    Kokkos::View<int*, MemorySpace> element_export_ids("element_export_ids", local_vef_count[0]);
    Kokkos::View<int*, MemorySpace> element_export_ranks("element_export_ranks", local_vef_count[0]);
    Kokkos::parallel_for("init_halo_data", Kokkos::RangePolicy<ExecutionSpace>(0, local_vef_count[0]),
        KOKKOS_LAMBDA(int i) {
        
        if (rank == 0)
        {
            element_export_ids(i) = 0;
            element_export_ranks(i) = -1;
        }
        else
        {
            element_export_ids(i) = i;
            element_export_ranks(i) = 0;
        }

    });

    auto vert_halo = Cabana::Halo<MemorySpace>(MPI_COMM_WORLD, local_vef_count[0], element_export_ids,
        element_export_ranks);

    lv = vert_halo.numLocal(); gv = vert_halo.numGhost();
    vertices_ptr.resize(lv + gv);
    vertices.resize(lv + gv);

    Cabana::gather(vert_halo, vertices_ptr);

    // Get edges
    Kokkos::resize(element_export_ids, local_vef_count[1]);
    Kokkos::resize(element_export_ranks, local_vef_count[1]);
    Kokkos::parallel_for("init_halo_data", Kokkos::RangePolicy<ExecutionSpace>(0, local_vef_count[1]),
        KOKKOS_LAMBDA(int i) {
        
        if (rank == 0)
        {
            element_export_ids(i) = 0;
            element_export_ranks(i) = -1;
        }
        else
        {
            element_export_ids(i) = i;
            element_export_ranks(i) = 0;
        }

    });

    auto edge_halo = Cabana::Halo<MemorySpace>(MPI_COMM_WORLD, local_vef_count[1], element_export_ids,
        element_export_ranks);
    
    // printf("R%d halo local/ghost: %d, %d, import/export: %d, %d\n", rank_,
    //     edge_halo.numLocal(), edge_halo.numGhost(),
    //     edge_halo.totalNumImport(), edge_halo.totalNumExport());


    le = edge_halo.numLocal(); ge = edge_halo.numGhost();
    edges_ptr.resize(le + ge);
    edges.resize(le + ge);

    Cabana::gather(edge_halo, edges_ptr);

    // Get Faces
    Kokkos::resize(element_export_ids, local_vef_count[2]);
    Kokkos::resize(element_export_ranks, local_vef_count[2]);
    Kokkos::parallel_for("init_halo_data", Kokkos::RangePolicy<ExecutionSpace>(0, local_vef_count[2]),
        KOKKOS_LAMBDA(int i) {
        
        if (rank == 0)
        {
            element_export_ids(i) = 0;
            element_export_ranks(i) = -1;
        }
        else
        {
            element_export_ids(i) = i;
            element_export_ranks(i) = 0;
        }

    });

    auto face_halo = Cabana::Halo<MemorySpace>(MPI_COMM_WORLD, local_vef_count[2], element_export_ids,
        element_export_ranks);

    lf = face_halo.numLocal(); gf = face_halo.numGhost();
    faces_ptr.resize(lf + gf);
    faces.resize(lf + gf);

    Cabana::gather(face_halo, faces_ptr);

    // Copy data to host
    // Rank 0 holds the entire mesh
    Cabana::deep_copy(faces, faces_ptr);
    Cabana::deep_copy(edges, edges_ptr);
    Cabana::deep_copy(vertices, vertices_ptr);
}

} // end namespace Utils

#endif // _TESTING_UTILS_HPP_