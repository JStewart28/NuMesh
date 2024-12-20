#ifndef NUMESH_MAPS_HPP
#define NUMESH_MAPS_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Mesh.hpp>

#include <mpi.h>

namespace NuMesh
{

namespace Maps
{

//---------------------------------------------------------------------------//
/*!
  \class V2E
  \brief Builds a local CSR-like structure for mapping local vertices to local edges:
        View<int*> vertex_edge_offsets for the starting index of edges per vertex.
        View<int*> vertex_edge_indices for storing local edge indices in the adjacency list.
        However, IDs stored are global IDs
*/
template <class Mesh>
class V2E
{
  public:

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;

    V2E( std::shared_ptr<Mesh> mesh )
        : _mesh ( mesh )
        , _comm ( mesh->comm() )
        , _map_version ( mesh->version() )
    {
        static_assert( isnumesh_mesh<Mesh>::value, "NuMesh::V2E: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        build();
    };

    ~V2E() {}

    /**
     * Build the V2E
     */
    void build()
    {
        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();

        // Number of local vertices and edges.
        int num_vertices = vertices.size();
        int num_edges = edges.size();

        // Allocate vertex-edge count and offsets.
        integer_view vertex_edge_count("vertex_edge_count", num_vertices);
        auto e_vid = Cabana::slice<E_VIDS>(edges); // Vertex IDs of each edge.
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Step 1: Count edges per vertex.
        auto vef_gid_start = _mesh->vef_gid_start();
        int vertex_gid_start = vef_gid_start(_rank, 0);
        Kokkos::parallel_for("Count edges per vertex", Kokkos::RangePolicy<execution_space>(0, num_edges),
            KOKKOS_LAMBDA(const int e) {
                for (int v = 0; v < 2; ++v) {  // Loop over edge endpoints.
                    int vgid = e_vid(e, v);
                    int vlid = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                    if (vlid > -1)
                        Kokkos::atomic_increment(&vertex_edge_count(vlid));
                }
            });
        Kokkos::fence();
        
        // Step 2: Create vertex-edge offsets using prefix sum.
        _vertex_edge_offsets = integer_view("vertex_edge_offsets", num_vertices);
        auto vertex_edge_offsets = _vertex_edge_offsets;
        Kokkos::parallel_scan("Prefix sum for vertex offsets", Kokkos::RangePolicy<execution_space>(0, num_vertices),
            KOKKOS_LAMBDA(const int i, int& update, const bool final) {
                int count = vertex_edge_count(i);
                if (final) {
                    vertex_edge_offsets(i) = update;
                }
                update += count;
            });
        Kokkos::fence();

        // Total number of edges in the adjacency list.
        int total_adj_edges = 0;
        Kokkos::parallel_reduce("Calculate total adjacency edges", Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(const int, int& count) {
                count = vertex_edge_offsets(num_vertices - 1) + vertex_edge_count(num_vertices - 1);
            }, total_adj_edges);

        _vertex_edge_indices = integer_view("vertex_edge_indices", total_adj_edges);
        // printf("R%d total adj edges: %d\n", rank, total_adj_edges);
        auto vertex_edge_indices = _vertex_edge_indices;

        // Step 3: Populate vertex-edge indices.
        integer_view current_offset("current_offset", num_vertices);
        Kokkos::deep_copy(current_offset, vertex_edge_offsets);  // Copy offsets for modification.

        Kokkos::parallel_for("Populate adjacency list", Kokkos::RangePolicy<execution_space>(0, num_edges),
            KOKKOS_LAMBDA(const int e) {
                for (int v = 0; v < 2; ++v) {  // Loop over edge endpoints.
                    int vgid = e_vid(e, v);
                    int vlid = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                    if (vlid > -1)
                    {
                        int insert_idx = Kokkos::atomic_fetch_add(&current_offset(vlid), 1);
                        // if (rank == 0) printf("R%d insert idx: %d, v l/g: %d, %d\n", rank, insert_idx, vlid, vgid);
                        vertex_edge_indices(insert_idx) = e;  // Store edge ID.
                    }
                }
            });
        Kokkos::fence();
    }

    auto vertex_edge_offsets() {return _vertex_edge_offsets;}
    auto vertex_edge_indices() {return _vertex_edge_indices;}

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    // Mesh version this halo is built from
    int _map_version;

    int _rank, _comm_size;
    
    integer_view _vertex_edge_offsets;
    integer_view _vertex_edge_indices;
    
};

} // end namespace Maps

} // end namespce NuMesh


#endif // NUMESH_MAPS_HPP