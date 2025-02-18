#include <Kokkos_Core.hpp>
#include <NuMesh_Core.hpp>
#include <mpi.h>
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// Custom hash function for pairs
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ (std::hash<T2>()(p.second) << 1);
    }
};

struct Vertex {
    int lid;
    int gid;
    int owner; // Rank which owns this vertex
    double x, y, z;
};

struct Cell {
    int id;
    int v0, v1, v2;  // Vertex IDs forming the triangle
    bool contains_ghost;  // Flag indicating if the cell contains a ghost point
};

struct Edge {
    int id;
    int v0, v1;  // Endpoints of the edge
};

// Function to ensure edges are always stored in (lower, higher) order
std::pair<int, int> make_sorted_edge(int a, int b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

int main(int argc, char** argv) {
    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = execution_space::memory_space;
    // using execution_space = Kokkos::Cuda;
    // using memory_space = Kokkos::CudaSpace;
    using nu_mesh_type = NuMesh::Mesh<execution_space, memory_space>;

    MPI_Init( &argc, &argv );         // Initialize MPI
    Kokkos::initialize( argc, argv ); // Initialize Kokkos

    { // Scope guard
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Construct filename based on rank
    std::string filename = "mesh_" + std::to_string(rank) + ".vtu";
    std::cout << "Rank " << rank << " reading " << filename << std::endl;

    // Read the VTU file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    vtkSmartPointer<vtkUnstructuredGrid> grid = reader->GetOutput();
    vtkSmartPointer<vtkPoints> points = grid->GetPoints();
    
    if (!points) {
        std::cerr << "Rank " << rank << " failed to read points from " << filename << std::endl;
        Kokkos::finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int num_points = points->GetNumberOfPoints();
    int num_cells = grid->GetNumberOfCells();

    // Read ghost point flags
    vtkSmartPointer<vtkDataArray> ghost_flags_array = grid->GetPointData()->GetArray("ghost_points");
    if (!ghost_flags_array) {
        std::cerr << "Rank " << rank << " failed to read ghost point flags from " << filename << std::endl;
        Kokkos::finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    vtkSmartPointer<vtkDataArray> vertex_owners_array = grid->GetPointData()->GetArray("vertex_owner");
    if (!vertex_owners_array) {
        std::cerr << "Rank " << rank << " failed to read vertex owner data from " << filename << std::endl;
        Kokkos::finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    vtkSmartPointer<vtkDataArray> vertex_gids_array = grid->GetPointData()->GetArray("vertex_gids");
    if (!vertex_gids_array) {
        std::cerr << "Rank " << rank << " failed to read vertex gid data from " << filename << std::endl;
        Kokkos::finalize();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    std::unordered_map<int, bool> is_ghost_point;
    for (int i = 0; i < num_points; ++i) {
        int ghost_flag = static_cast<int>(ghost_flags_array->GetComponent(i, 0));
        is_ghost_point[i] = (ghost_flag == 1);  // Mark as ghost point if flag is 1
    }

    // Mapping from global VTK point index to local vertex ID
    std::unordered_map<int, int> global_to_local;
    std::vector<Vertex> vertices;
        
    int num_ghost_vertices = 0;
    for (int i = 0; i < num_points; ++i) {
        double coords[3];
        points->GetPoint(i, coords);
        global_to_local[i] = i;  // Assign local ID
        int vertex_owner = static_cast<int>(vertex_owners_array->GetComponent(i, 0));
        int vertex_gid = static_cast<int>(vertex_gids_array->GetComponent(i, 0));
        if (is_ghost_point[i]) {
            num_ghost_vertices++;
        }
        vertices.push_back({i, vertex_gid, vertex_owner, coords[0], coords[1], coords[2]});
    }

    std::vector<Cell> cells;
    std::unordered_map<std::pair<int, int>, int, pair_hash> edge_map;
    std::vector<Edge> edges;

    // Read cells (triangles) and assign local cell IDs
    for (int i = 0; i < num_cells; ++i) {
        vtkCell* cell = grid->GetCell(i);
        if (cell->GetNumberOfPoints() != 3) continue;  // Skip non-triangle cells

        int v0 = cell->GetPointId(0);
        int v1 = cell->GetPointId(1);
        int v2 = cell->GetPointId(2);

        // Check if the cell contains a ghost point
        bool contains_ghost = is_ghost_point[v0] || is_ghost_point[v1] || is_ghost_point[v2];

        cells.push_back({i, v0, v1, v2, contains_ghost});

        // Create edges
        for (const auto& edge : {make_sorted_edge(v0, v1), make_sorted_edge(v1, v2), make_sorted_edge(v0, v2)}) {
            if (edge_map.find(edge) == edge_map.end()) {
                int edge_id = edge_map.size();
                edge_map[edge] = edge_id;
                edges.push_back({edge_id, edge.first, edge.second});
            }
        }
    }

    // Output results
    // std::cout << "Rank " << rank << " processed:\n"
    //           << "  - " << vertices.size() << " vertices\n"
    //           << "  - " << cells.size() << " cells\n"
    //           << "  - " << edges.size() << " edges\n";

    // Output the number of cells containing ghost points
    int ghost_cells_count = 0;
    for (const auto& cell : cells) {
        if (cell.contains_ghost) {
            ++ghost_cells_count;
        }
    }

    // std::cout << "Rank " << rank << " has " << ghost_cells_count << " cells containing ghost points.\n";

    auto mesh = NuMesh::createEmptyMesh<execution_space, memory_space>(MPI_COMM_WORLD);
    mesh->initializeFromVectors(vertices, edges, cells);

    } // Scope guard

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI
    return 0;
}
