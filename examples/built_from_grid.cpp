#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Core.hpp>

#include <mpi.h>

int main( int argc, char* argv[] )
{
    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = execution_space::memory_space;
    using nu_mesh_type = NuMesh::Mesh<execution_space, memory_space>;

    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mesh_size> <periodic>" << std::endl;
        
        return 1;  // Exit with error code
    }

    { // Scope guard


    // Convert the first command-line argument to an integer
    int mesh_size = 8;
    bool periodic = false;
    try {
        mesh_size = std::stoi(argv[1]);  // Convert argument to integer
        periodic = std::stoi(argv[2]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument for mesh_size: " << argv[1] << std::endl;
        Kokkos::finalize(); // Finalize Kokkos
        MPI_Finalize();     // Finalize MPI
        return 1;  // Exit with error code
    } catch (const std::out_of_range& e) {
        std::cerr << "Argument out of range for mesh_size: " << argv[1] << std::endl;
        Kokkos::finalize(); // Finalize Kokkos
        MPI_Finalize();     // Finalize MPI
        return 1;  // Exit with error code
    }

    std::cout << "Mesh size: " << mesh_size << std::endl;  // Print the mesh_size

    // MPI Info
    int comm_size, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size ); // Number of Ranks
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );      // My Rank

    std::array<int, 2> global_num_cell = { mesh_size, mesh_size };
    std::array<double, 2> global_low_corner = { -1.0, -1.0 };
    std::array<double, 2> global_high_corner = { 1.0, 1.0 };
    std::array<bool, 2> is_dim_periodic = { periodic, periodic };
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    std::shared_ptr<nu_mesh_type> nu_mesh;
    nu_mesh = std::make_shared<nu_mesh_type>(global_low_corner, global_high_corner,
	        global_num_cell, is_dim_periodic, partitioner, MPI_COMM_WORLD);
    nu_mesh->initialize_from_grid();
    nu_mesh->initialize_faces();
    nu_mesh->assign_edges_to_faces();
    // int ranks_in_xy = (int) floor(sqrt((float) comm_size));
    // if (ranks_in_xy*ranks_in_xy != comm_size) 
    // {
    //     if (rank == 0) printf("ERROR: The number of ranks must be a square number to use the cutoff solver. There are %d ranks.\n", comm_size);
    //     Kokkos::finalize();
    //     MPI_Finalize();
    //     return 1;
    // }
    // std::array<int, 2> input_ranks_per_dim = { ranks_in_xy, ranks_in_xy};


    

    // printView(local_L2G, rank, z, 1, 5, 5);



    
    } // Scope guard

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI
    return 0;
}
