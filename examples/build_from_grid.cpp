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

    auto numesh = NuMesh::createEmptyMesh<execution_space, memory_space>(MPI_COMM_WORLD);

    auto layout = Cabana::Grid::createArrayLayout(local_grid, 1, Cabana::Grid::Node());
    auto array = Cabana::Grid::createArray<double, memory_space>("for_initialization", layout);
    numesh->initializeFromArray(*array);
    int size = 2;
    Kokkos::View<int*, memory_space> fids("fids", size);
    Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, size),
        KOKKOS_LAMBDA(int i) {
        
        if (i == 0) fids(i) = 30;
        if (i == 1) fids(i) = 31;
        //if (i == 1) fids(i) = 13;

        if (i == 2) fids(i) = 106;          // rank 3
        
        // else if (i == 1) fids(i) = 5;       // rank 0
        
        else if (i == 3) fids(i) = 75;      // rank 2

        else if (i == 4) fids(i) = 51;      // rank 1

    });

    numesh->refine(fids);
    numesh->printFaces(1, 14);
    numesh->printEdges(1, 21);
    numesh->printEdges(1, 110);
    numesh->printEdges(1, 22);

    // Uniform refinement
    // for (int i = 0; i < 2; i++)
    // {
    //     int num_local_faces = numesh->count(NuMesh::Own(), NuMesh::Face());
    //     auto vef_gid_start = numesh->get_vef_gid_start();
    //     int face_gid_start = vef_gid_start(rank, 2);
    //     Kokkos::View<int*, memory_space> fin("fin", num_local_faces);
    //     Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, num_local_faces),
    //         KOKKOS_LAMBDA(int i) {

    //             fin(i) = face_gid_start + i;

    //         });
    //     numesh->refine(fin);
    // }

    // numesh->printFaces(1, 296);
    // numesh->printEdges(1, 530);
    // numesh->printEdges(1, 824);
    // numesh->printEdges(1, 0);
    // e(530, 824, 0),

    // Second refine
    // size = 1;
    // Kokkos::resize(fids, 1);
    // Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, size),
    //     KOKKOS_LAMBDA(int i) {
        
    //     if (i == 0) fids(i) = 37;

    // });
    // numesh->refine(fids);


    //numesh->_refine(13);
    //numesh->_refine(22);
    // if(rank == 1) numesh->printEdges(2, 0);
    // numesh->printFaces(0, 30);
    //numesh->printVertices();
    //printf("**********\n");
    //numesh->printFaces(1, 30);
    //numesh->printFaces(1, 31);
    // for (int i = 32; i < 44; i++)
    // {
    //     numesh->printFaces(1, i);
    // }
    // numesh->printEdges(1, 45);
    // numesh->printEdges(1, 105);
    // numesh->printEdges(1, 144);
    // numesh->printEdges(1, 145);
    // numesh->printEdges(1, 46);
    // numesh->printEdges(1, 86);

    // if (rank == 0) numesh->printFaces(0, 0);
    // if (rank == 0) numesh->printEdges(3, 0);

    // numesh->printEdges(2, 105);

    // Rank 0 edges, face 30
    // numesh->printEdges(1, 45);
    // numesh->printEdges(1, 46);
    // numesh->printEdges(1, 47);

    // Rank 1 edges
    // numesh->printEdges(1, 98);
    // numesh->printEdges(1, 108);
    // numesh->printEdges(1, 109);

    // Rank 2 edges
    // numesh->printEdges(1, 119);
    // numesh->printEdges(1, 158);
    // numesh->printEdges(1, 159);

    } // Scope guard

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI
    return 0;
}
