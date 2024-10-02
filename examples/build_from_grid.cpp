#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Core.hpp>

#include <mpi.h>

enum InitialConditionModel {IC_COS = 0, IC_SECH2, IC_GAUSSIAN, IC_RANDOM, IC_FILE};

// Initialize field to a constant quantity and velocity
struct MeshInitFunc
{
    // Initialize Variables

    MeshInitFunc( std::array<double, 6> box, enum InitialConditionModel i,
                  double t, double m, double v, double p, 
                  const std::array<int, 2> nodes, enum NuMesh::BoundaryType boundary )
        : _i(i)
        , _t( t )
        , _m( m )
        , _v( v)
        , _p( p )
        , _b( boundary )
    {
	    _ncells[0] = nodes[0] - 1;
        _ncells[1] = nodes[1] - 1;

        _dx = (box[3] - box[0]) / _ncells[0];
        _dy = (box[4] - box[1]) / _ncells[1]; 


    };

    template <class RandNumGenType>
    KOKKOS_INLINE_FUNCTION
    bool operator()( Cabana::Grid::Node, NuMesh::Vertex,
                     RandNumGenType random_pool,
                     [[maybe_unused]] const int index[2],
                     const double coord[2],
                     double &z1, double &z2, double &z3) const
    {
        double lcoord[2];
        /* Compute the physical position of the interface from its global
         * coordinate in mesh space */
        for (int i = 0; i < 2; i++) {
            lcoord[i] = coord[i];
            if (_b == NuMesh::BoundaryType::FREE && (_ncells[i] % 2 == 1) ) {
                lcoord[i] += 0.5;
            }
        }
        z1 = _dx * lcoord[0];
        z2 = _dy * lcoord[1];

        // We don't currently support tilting the initial interface

        /* Need to initialize these values here to avoid "jump to case label "case IC_FILE:"
         * crosses initialization of ‘double gaussian’, etc." errors */
        auto generator = random_pool.get_state();
        double rand_num = generator.drand(-1.0, 1.0);
        double mean = 0.0;
        double std_dev = 1.0;
        double gaussian = (1 / (std_dev * Kokkos::sqrt(2 * Kokkos::numbers::pi_v<double>))) *
            Kokkos::exp(-0.5 * Kokkos::pow(((rand_num - mean) / std_dev), 2));
        switch (_i) {
        case IC_COS:
            z3 = _m * cos(z1 * (2 * M_PI / _p)) * cos(z2 * (2 * M_PI / _p));
            break;
        case IC_SECH2:
            z3 = _m * pow(1.0 / cosh(_p * (z1 * z1 + z2 * z2)), 2);
            break;
        case IC_RANDOM:
            z3 = _m * (2*rand_num - 1.0);
            break;
        case IC_GAUSSIAN:
            /* The built-in C++ std::normal_distribution<double> doesn't
             * work here, so coding the gaussian distribution itself.
             */
            z3 = _m * gaussian;
            break;
        case IC_FILE:
            break;
        }
        
        random_pool.free_state(generator);

        return true;
    };

    enum InitialConditionModel _i;
    double _t, _m, _v, _p;
    Kokkos::Array<int, 3> _ncells;
    double _dx, _dy;
    enum NuMesh::BoundaryType _b;
};

int main( int argc, char* argv[] )
{
    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = execution_space::memory_space;
    // using nu_mesh_type = NuMesh::Mesh<execution_space, memory_space>;

    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [mesh_size] [periodic] [initial_condition]" << std::endl;
        
        return 1;  // Exit with error code
    }

    { // Scope guard


    // Convert the first command-line argument to an integer
    int mesh_size = 8;
    enum InitialConditionModel initial_condition;
    enum NuMesh::BoundaryType boundary_type;
    try {
        mesh_size = std::stoi(argv[1]);  // Convert argument to integer
        int val = std::stoi(argv[2]);
        if (!val) boundary_type = NuMesh::BoundaryType::PERIODIC;
        else boundary_type = NuMesh::BoundaryType::FREE;
        int ic = std::stoi(argv[3]);
        if (ic == 0) initial_condition = InitialConditionModel::IC_COS;
        else if (ic == 1) initial_condition = InitialConditionModel::IC_SECH2;
        else if (ic == 2) initial_condition = InitialConditionModel::IC_GAUSSIAN;
        else if (ic == 3) initial_condition = InitialConditionModel::IC_RANDOM;
        else initial_condition = InitialConditionModel::IC_FILE;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Usage: ./build_from_grid [mesh_size] [periodic] [initial_condition]" << std::endl;
        std::cerr << "Invalid argument for mesh_size: " << argv[1] << std::endl;
        Kokkos::finalize(); // Finalize Kokkos
        MPI_Finalize();     // Finalize MPI
        return 1;  // Exit with error code
    } catch (const std::out_of_range& e) {
        std::cerr << "Usage: ./build_from_grid [mesh_size] [periodic] [initial_condition]" << std::endl;
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
    std::array<double, 6> global_bounding_box = {-1.0, -1.0, -1.0, 1.0, 1.0, 1.0};
    std::array<bool, 2> is_dim_periodic = { !(boundary_type), !(boundary_type) };
    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    double tilt = 0.0;
    double magnitude = 0.05;
    double variation = 0.00;
    double period = 1.0;
    std::array<int, 2> num_nodes = { 128, 128 };

    MeshInitFunc initializer( global_bounding_box, initial_condition,
                              tilt, magnitude, variation, period,
                              num_nodes, boundary_type );


    auto nu_mesh = NuMesh::createMesh<execution_space, memory_space>(MPI_COMM_WORLD);
    nu_mesh->initialize_ve(initializer, global_low_corner, global_high_corner, global_num_cell,
        is_dim_periodic, partitioner, period, MPI_COMM_WORLD);
    // nu_mesh->initialize_from_grid();
    nu_mesh->initialize_faces();
    nu_mesh->initialize_edges();
    // nu_mesh->gather_edges();
    // nu_mesh->assign_ghost_edges_to_faces();

    auto index_space_o = nu_mesh->indexSpace(NuMesh::Own(), NuMesh::Vertex(), NuMesh::Local());
    auto index_space_g = nu_mesh->indexSpace(NuMesh::Ghost(), NuMesh::Vertex(), NuMesh::Local());
    //printf("owned edges: (%d, %d)\n", index_space_o.min(0), index_space_o.max(0));
    //printf("ghost edges: (%d, %d)\n", index_space_g.min(0), index_space_g.max(0));

    // auto edge_triple_layout = NuMesh::Array::createArrayLayout(nu_mesh, 3, NuMesh::Vertex());
    // auto edge_triple_array = NuMesh::Array::createArray<double, memory_space>("edge_triple_array", edge_triple_layout);
    // auto extent0 = edge_triple_layout->indexSpace(NuMesh::Ghost(), NuMesh::Vertex(), NuMesh::Local()).extent(0);
    // auto extent1 = edge_triple_layout->indexSpace(NuMesh::Ghost(), NuMesh::Vertex(), NuMesh::Local()).extent(1);
    // //printf("extents: (%d, %d)\n", extent0, extent1);
    // auto edge_view = edge_triple_array->view();
    // //printf("Edge view extents: (%d, %d)\n", edge_view.extent(0), edge_view.extent(1));
    // auto copy = NuMesh::Array::ArrayOp::cloneCopy(*edge_triple_array, NuMesh::Own());
    // NuMesh::Array::ArrayOp::assign(*copy, 1.0, NuMesh::Own());
    
    // auto node_triple_layout =
    //     Cabana::Grid::createArrayLayout( pm.mesh().localGrid(), 3, Cabana::Grid::Node() );
    // auto node_pair_layout =
    //     Cabana::Grid::createArrayLayout( pm.mesh().localGrid(), 2, Cabana::Grid::Node() );

    // _zdot = Cabana::Grid::createArray<double, mem_space>("velocity", 
    //                                                 node_triple_layout);
    // _wdot = Cabana::Grid::createArray<double, mem_space>("vorticity derivative",
    //                                                 node_pair_layout);



    //nu_mesh->assign_edges_to_faces_orig();
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
