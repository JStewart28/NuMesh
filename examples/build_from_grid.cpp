#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Core.hpp>

#include <mpi.h>

int main( int argc, char* argv[] )
{
    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = execution_space::memory_space;
    // using execution_space = Kokkos::Cuda;
    // using memory_space = Kokkos::CudaSpace;
    // using nu_mesh_type = NuMesh::Mesh<execution_space, memory_space>;

    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [mesh_size] [periodic]" << std::endl;
        
        return 1;  // Exit with error code
    }

    { // Scope guard


    // Convert the first command-line argument to an integer
    int mesh_size = -1;
    enum NuMesh::BoundaryType boundary_type;
    try {
        mesh_size = std::stoi(argv[1]);  // Convert argument to integer
        int val = std::stoi(argv[2]);
        if (val) boundary_type = NuMesh::BoundaryType::PERIODIC;
        else boundary_type = NuMesh::BoundaryType::FREE;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Usage: ./build_from_grid [mesh_size] [periodic]" << std::endl;
        std::cerr << "Invalid argument for mesh_size: " << argv[1] << std::endl;
        Kokkos::finalize(); // Finalize Kokkos
        MPI_Finalize();     // Finalize MPI
        return 1;  // Exit with error code
    } catch (const std::out_of_range& e) {
        std::cerr << "Usage: ./build_from_grid [mesh_size] [periodic]" << std::endl;
        std::cerr << "Argument out of range for mesh_size: " << argv[1] << std::endl;
        Kokkos::finalize(); // Finalize Kokkos
        MPI_Finalize();     // Finalize MPI
        return 1;  // Exit with error code
    }

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    if (rank == 0)
    {
        std::cout << "Mesh size: " << mesh_size << std::endl;
        std::cout << "Periodic: " << boundary_type << std::endl;
    }


    std::array<int, 2> global_num_cell = { mesh_size, mesh_size };
    std::array<double, 2> global_low_corner = { -1.0, -1.0 };
    std::array<double, 2> global_high_corner = { 1.0, 1.0 };
    // std::array<double, 6> global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    std::array<bool, 2> periodic = { (bool)boundary_type, (bool)boundary_type };
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, periodic, partitioner );
    int halo_width = 2;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

    auto mesh = NuMesh::createEmptyMesh<execution_space, memory_space>(MPI_COMM_WORLD);

    auto layout = Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Node());
    auto array = Cabana::Grid::createArray<double, memory_space>("for_initialization", layout);
    mesh->initializeFromArray(*array);
    auto vef_gid_start = mesh->vef_gid_start();

    // Uniform refinement
    // for (int i = 0; i < 2; i++)
    // {
    //     int num_local_faces = mesh->count(NuMesh::Own(), NuMesh::Face());
    //     int face_gid_start = vef_gid_start(rank, 2);
    //     Kokkos::View<int*, memory_space> fin("fin", num_local_faces);
    //     Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, num_local_faces),
    //         KOKKOS_LAMBDA(int i) {

    //             fin(i) = face_gid_start + i;

    //         });
    //     mesh->refine(fin);
    //     printf("R%d: refine %d: num faces: %d\n", rank, i+1, mesh->faces().size());
    // }
    // mesh->printFaces(1, 258);
    // mesh->printFaces(1, 259);
    // mesh->printFaces(1, 260);
    // mesh->printFaces(1, 261);
    // mesh->printFaces(1, 52);
    // mesh->printFaces(1, 0);

    // // Face 0
    // mesh->printEdges(1, 0);
    // mesh->printEdges(1, 5);
    // mesh->printEdges(1, 1);

    // // Face 52
    // mesh->printEdges(1, 226);
    // mesh->printEdges(1, 86);
    // mesh->printEdges(1, 78);

    // // Face 258
    // mesh->printEdges(1, 677);
    // mesh->printEdges(1, 983);
    
    /**
     * Face 258 has issue. Parent face is 52
     * Parent face of 52 is 0.
     */
    
    
    // mesh->printFaces(1, 258);
    // mesh->printEdges(1, 677);
    // mesh->printEdges(1, 983);
    // mesh->printEdges(1, 0);

    // auto halo = NuMesh::createHalo(mesh, 0, 1);
    // halo.gather();
    // printf("R%d: finished no refinement gather\n", rank);

    // Single refinement
    int sizerefine = 1;
    Kokkos::View<int*, memory_space> fin("fin", sizerefine);
    Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, sizerefine),
        KOKKOS_LAMBDA(int i) {

            fin(i) = 94;

        });
    mesh->refine(fin);
    Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, sizerefine),
        KOKKOS_LAMBDA(int i) {

            fin(i) = 96;

        });
    mesh->refine(fin);
    // mesh->printFaces(1, 94);
    // Test haloing
    // auto v2e = NuMesh::Maps::V2E(mesh);
    // auto v2f = NuMesh::Maps::V2F(mesh);
    auto halo = NuMesh::createHalo(mesh, 0, 1);
    halo.gather();

    // printf("R%d: finished gather 1\n", rank);

    // halo.gather();

    // mesh->printFaces(0, 0);

    // size_t vsize = 1;
    // Kokkos::View<int*, memory_space> verts("verts", vsize);
    // Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, vsize),
    //     KOKKOS_LAMBDA(int i) {

    //         verts(i) = i;

    //     });
    

    // for (size_t i = 0; i < halo_verts.extent(0); i++)
    // {
    //     printf("R%d: vert %d added\n", rank, halo_verts(i));
    // }

    } // Scope guard

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI
    return 0;
}
