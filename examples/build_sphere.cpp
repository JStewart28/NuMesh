#include <mpi.h>
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkTriangle.h>
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
    int id;
    double x, y, z;
};

struct Cell {
    int id;
    int v0, v1, v2;  // Vertex IDs forming the triangle
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
    MPI_Init(&argc, &argv);
    
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
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int num_points = points->GetNumberOfPoints();
    int num_cells = grid->GetNumberOfCells();

    // Mapping from global VTK point index to local vertex ID
    std::unordered_map<int, int> global_to_local;
    std::vector<Vertex> vertices;
    
    // Read points and assign local vertex IDs
    for (int i = 0; i < num_points; ++i) {
        double coords[3];
        points->GetPoint(i, coords);
        global_to_local[i] = i;  // Assign local ID
        vertices.push_back({i, coords[0], coords[1], coords[2]});
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

        cells.push_back({i, v0, v1, v2});

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
    std::cout << "Rank " << rank << " processed:\n"
              << "  - " << vertices.size() << " vertices\n"
              << "  - " << cells.size() << " cells\n"
              << "  - " << edges.size() << " edges\n";

    MPI_Finalize();
    return 0;
}
