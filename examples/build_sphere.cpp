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

// Host-side AoSoAs for storing VTU data
using vertices_d = Cabana::MemberTypes<int,       // Vertex global ID                                 
                                       int       // Owning rank
                                       >;
using face_d = Cabana::MemberTypes<int[3],       // Vertex LIDs forming the triangle                                
                                   bool         // Flag indicating if the cell contains a ghost point
                                   >;
using triple_d = Cabana::MemberTypes<double[3]>; // Vertex positions

using vert_aosoa = Cabana::AoSoA<vertices_d, Kokkos::HostSpace, 4>;
using face_aosoa = Cabana::AoSoA<face_d, Kokkos::HostSpace, 4>;
using triple_aosoa = Cabana::AoSoA<triple_d, Kokkos::HostSpace, 4>;


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
    // std::cout << "Rank " << rank << " reading " << filename << std::endl;

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

    // Create AoSoAs
    vert_aosoa vertices("vertices", num_points);
    face_aosoa faces("faces", num_cells);
    triple_aosoa positions_h("positions_h", num_cells);
    auto v_gid = Cabana::slice<0>(vertices);
    auto v_owner = Cabana::slice<1>(vertices);
    auto f_vids = Cabana::slice<0>(faces);
    auto f_isGhost = Cabana::slice<1>(faces);
    auto p_xyz = Cabana::slice<0>(positions_h);

    std::unordered_map<int, bool> is_ghost_point;
    for (int i = 0; i < num_points; ++i) {
        int ghost_flag = static_cast<int>(ghost_flags_array->GetComponent(i, 0));
        is_ghost_point[i] = (ghost_flag == 1);  // Mark as ghost point if flag is 1
    }

    // Mapping from global VTK point index to local vertex ID
    std::unordered_map<int, int> global_to_local;
    // std::vector<Vertex> vertices;
        
    int owned_verts = 0;
    for (int i = 0; i < num_points; ++i) {
        global_to_local[i] = i;  // Assign local ID
        int vertex_owner = static_cast<int>(vertex_owners_array->GetComponent(i, 0));
        int vertex_gid = static_cast<int>(vertex_gids_array->GetComponent(i, 0));
        v_gid(i) = vertex_gid;
        v_owner(i) = vertex_owner;
        if (vertex_owner == rank) owned_verts++;

        // Populate coordinates
        double coords[3];
        points->GetPoint(i, coords);
        for (int j = 0; j < 3; j++) p_xyz(i, j) = coords[j];
        // vertices.push_back({i, vertex_gid, vertex_owner, coords[0], coords[1], coords[2]});
    }
    positions_h.resize(owned_verts);

    // std::vector<Cell> cells;
    std::unordered_map<std::pair<int, int>, int, pair_hash> edge_map;
    // std::vector<Edge> edges;

    // Read cells (triangles) and assign local cell IDs
    for (int i = 0; i < num_cells; ++i) {
        vtkCell* cell = grid->GetCell(i);
        if (cell->GetNumberOfPoints() != 3) continue;  // Skip non-triangle cells

        int v0 = cell->GetPointId(0);
        int v1 = cell->GetPointId(1);
        int v2 = cell->GetPointId(2);

        // Check if the cell contains a ghost point
        bool contains_ghost = is_ghost_point[v0] || is_ghost_point[v1] || is_ghost_point[v2];

        f_vids(i, 0) = v0; f_vids(i, 1) = v1; f_vids(i, 2) = v2;
        f_isGhost(i) = contains_ghost;

        // cells.push_back({i, v0, v1, v2, contains_ghost});

        // Create edges
        for (const auto& edge : {make_sorted_edge(v0, v1), make_sorted_edge(v1, v2), make_sorted_edge(v0, v2)}) {
            if (edge_map.find(edge) == edge_map.end()) {
                int edge_id = edge_map.size();
                edge_map[edge] = edge_id;
                // edges.push_back({edge_id, edge.first, edge.second});
            }
        }
    }

    // Output results
    // std::cout << "Rank " << rank << " processed:\n"
    //           << "  - " << vertices.size() << " vertices\n"
    //           << "  - " << cells.size() << " cells\n"
    //           << "  - " << edges.size() << " edges\n";

    // Output the number of cells containing ghost points
    // int ghost_cells_count = 0;
    // for (const auto& cell : cells) {
    //     if (cell.contains_ghost) {
    //         ++ghost_cells_count;
    //     }
    // }

    // std::cout << "Rank " << rank << " has " << ghost_cells_count << " cells containing ghost points.\n";

    auto mesh = NuMesh::createEmptyMesh<execution_space, memory_space>(MPI_COMM_WORLD);

    // Copy AoSoAs to deivce, then initialize
    using vert_aosoa_device = Cabana::AoSoA<vertices_d, memory_space, 4>;
    using face_aosoa_device = Cabana::AoSoA<face_d, memory_space, 4>;
    using triple_aosoa = Cabana::AoSoA<triple_d, memory_space, 4>;
    vert_aosoa_device vertices_device("vertices_device", vertices.size());
    face_aosoa_device faces_device("faces_device", faces.size());
    triple_aosoa positions_device("positions_device", positions_h.size());
    Cabana::deep_copy(vertices_device, vertices);
    Cabana::deep_copy(faces_device, faces);
    Cabana::deep_copy(positions_device, positions_h);

    mesh->initializeFromConnectivity(vertices_device, faces_device);

    // Create positions array
    using tuple_type = Cabana::MemberTypes<double[3]>;
    auto vertex_triple_layout = NuMesh::Array::createArrayLayout<tuple_type>(mesh, 3, NuMesh::Vertex());
    auto positions = NuMesh::Array::createArray<memory_space>("positions", vertex_triple_layout);
    auto paosoa = positions->aosoa();

    // auto numesh_halo = NuMesh::createHalo(mesh, 0, 1, NuMesh::Vertex());
    // positions->update();
    // NuMesh::gather(numesh_halo, positions);
    // // printf("After gather / before refine: R%d: pos: %d, verts: %d\n", rank, paosoa->size(), mesh->vertices().size());


    // Kokkos::View<int[1], memory_space> fin("fin");
    // Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, fin.extent(0)),
    // KOKKOS_LAMBDA(int i) {

    //     fin(i) = 15;

    // });
    
    // mesh->refine(fin);
    // printf("Before update: R%d: pos: %d, verts: %d\n", rank, paosoa->size(), mesh->vertices().size());

    // positions->update();
    // printf("After refine and update: R%d: pos: %d, verts: %d\n", rank, paosoa->size(), mesh->vertices().size());

    // Uniform refinement
    for (int i = 0; i < 4; i++)
    {
        int num_local_faces = mesh->count(NuMesh::Own(), NuMesh::Face());
        auto vef_gid_start = mesh->vef_gid_start();
        int face_gid_start = vef_gid_start(rank, 2);
        Kokkos::View<int*, memory_space> fin("fin", num_local_faces);
        Kokkos::parallel_for("mark_faces_to_refine", Kokkos::RangePolicy<execution_space>(0, num_local_faces),
            KOKKOS_LAMBDA(int i) {

                fin(i) = face_gid_start + i;

            });
        if (rank  == 0) printf("R%d: Starting refine %d...\n", rank, i+1);
        mesh->refine(fin);
        int global_max_tree_depth = mesh->global_max_max_tree_depth();
        int global_min_tree_depth = mesh->global_min_max_tree_depth();
        // if (rank == 0) printf("R%d: global min/max depths: %d, %d\n", rank, global_min_tree_depth, global_max_tree_depth);
        if (rank == 0) printf("R%d: Creating halo %d at level %d...\n", rank, i+1, global_min_tree_depth);

        auto numesh_halo = NuMesh::createHalo(mesh, global_min_tree_depth, 1, NuMesh::Vertex());
        // mesh->gather(global_min_tree_depth, 1);
        if (rank == 0) printf("R%d: gathering positions...\n", rank);
        positions->update();
        NuMesh::gather(numesh_halo, positions);

    }
    if (rank == 0) printf("R%d: done\n", rank);

    // mesh->gather(1, 1);
    
    // printf("After gather/update: R%d: pos: %d, verts: %d\n", rank, paosoa->size(), mesh->vertices().size());



    // auto v2e = NuMesh::Maps::V2E(mesh);

    } // Scope guard

    Kokkos::finalize(); // Finalize Kokkos
    MPI_Finalize();     // Finalize MPI
    return 0;
}
