#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Core.hpp>

#include <mpi.h>

int main( int argc, char* argv[] )
{
    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = execution_space::memory_space;
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

    auto numesh = NuMesh::createEmptyMesh<execution_space, memory_space>(MPI_COMM_WORLD);

    auto layout = Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Node());
    auto array = Cabana::Grid::createArray<double, memory_space>("for_initialization", layout);
    numesh->initializeFromArray(*array);
    numesh->_refine(12);
    numesh->_refine(13);
    //numesh->_refine(22);
    numesh->printEdges(3);
    printf("**********\n");
    numesh->printFaces();


    } // Scope guard

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI
    return 0;
}
