#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
#include <NuMesh_Mesh.hpp>

#include <mpi.h>

namespace NuMesh
{

//---------------------------------------------------------------------------//
/*!
  \class Halo
  \brief Unstructured triangle mesh
*/
template <class Mesh>
class Halo
{
  public:

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;
    using int_d = Kokkos::View<int, memory_space>;
    using RankToEntityListMap = Kokkos::UnorderedMap<int, integer_view>;

    /**
     * x_guess parameters: guesses to how many vertices, edges, and faces, per rank,
     * will be in the halo to avoid frequent resizing.
     */
    Halo( std::shared_ptr<Mesh> mesh, const int entity, const int level, const int depth,
          const int vert_guess=50, const int edge_guess=50, const int face_guess=50 )
        : _mesh ( mesh )
        , _entity ( entity )
        , _level ( level )
        , _depth ( depth )
        , _comm ( mesh->comm() )
        , _halo_version ( mesh->version() )
    {
        static_assert( isnumesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        int num_edges = _mesh->count(Own(), Edge());
        _boundary_edges = Kokkos::View<bool*>("boundary_edges", num_edges);

        // Create and initalize rank-to-boundary-entities maps
        _rank_to_boundary_vertices = RankToEntityListMap("_rank_to_boundary_vertices", _comm_size);
        _rank_to_boundary_edges = RankToEntityListMap("_rank_to_boundary_edges", _comm_size);
        rank_to_boundary_vertices = _rank_to_boundary_vertices;
        rank_to_boundary_edges = _rank_to_boundary_edges;
        Kokkos::parallel_for("Insert keys", Kokkos::RangePolicy<execution_space>(0, _comm_size),
            KOKKOS_LAMBDA(const int key) {

            if (rank_to_boundary_vertices.insert(key) != MapType::invalid_index) {
                // Allocate a view for each key
                rank_to_boundary_vertices.value_at(
                    rank_to_boundary_vertices.find(key)) = VectorType("vertex_rank_vector", vert_guess);
            }
            if (rank_to_boundary_edges.insert(key) != MapType::invalid_index) {
                // Allocate a view for each key
                rank_to_boundary_edges.value_at(
                    rank_to_boundary_edges.find(key)) = VectorType("vertex_rank_vector", vert_guess);
            }
        });

        _vertex_ids = integer_view("_vertex_ids", 0);
        _edge_ids = integer_view("_edge_ids", 0);
    };

    ~Halo()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
    }

    /**
     * Functor to add values to maps
     * 
     * XXX - currently performs linear search. Can be optimized.
     */
    auto add_value_to_map = KOKKOS_LAMBDA(int key, int value) {
        auto key_index = map.find(key);
        if (key_index != MapType::invalid_index) {
            auto& vec = map.value_at(key_index);

            // Check if the value is already in the vector
            bool found = false;
            for (size_t i = 0; i < vec.extent(0); ++i) {
                if (vec(i) == value) {
                    found = true;
                    break;
                }
            }

            // If the value is not found, add it to the vector
            if (!found) {
                auto new_size = vec.extent(0) + 1;
                VectorType temp("temp", new_size);
                Kokkos::deep_copy(temp, vec); // Copy old data
                temp(new_size - 1) = value;   // Add new value
                vec = temp;                  // Assign the new view
            }
        }
    };

    void build()
    {

    }

    void find_boundary_edges_and_vertices()
    {
        const int rank = _rank;

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();
        auto e_vids = Cabana::slice<E_VIDS>(edges);
        
        // Get vef_gid_start and copy to device
        auto vef_gid_start = _mesh->get_vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        int num_edges = _mesh->count(Own(), Edge());

        Kokkos::parallel_for("MarkBoundaryEdges", Kokkos::RangePolicy<execution_space>(0, num_edges),
            KOKKOS_LAMBDA(int edge_idx) {
        
            bool is_boundary = false;

            // Iterate over the vertices of this edge
            for (int v = 0; v < 2; v++) {
                int vgid = e_vids(edge_idx, v);
                int vertex_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start_d);

                if (vertex_owner != rank) {
                    is_boundary = true;

                    // Insert this vertex into the rank-to-vertices map
                    // Need atomics for safe parallel insertion
                    auto result = rank_to_boundary_vertices.insert(vertex_owner);
                    if (result.failed()) {
                        // Handle failed insert due to collisions
                    } else {
                        auto vertex_list = result.value();
                        Kokkos::atomic_fetch_add(&vertex_list(vertex_id), 1);
                    }
                }
            }

            if (is_boundary) {
                boundary_edges(edge_idx) = true;
            }
        });
    }

    /**
     * Given a list of vertex GIDs and a level of the tree, update
     * _vertex_ids and _edge_ids with edges connected to vertices
     * in the input list and the vertices of those connected edges.
     * 
     * Returns a list of new vertex GIDs added to the halo.
     */
    integer_view gather_local_neighbors(integer_view verts, int level)
    {
        if (_halo_version != _mesh->version())
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: mesh and halo version do not match" );
        }

        const int rank = _rank, comm_size = _comm_size;

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();

        // Get vef_gid_start and copy to device
        auto vef_gid_start = _mesh->get_vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        // Create vertices to edges map
        auto v2e = Maps::V2E(_mesh);
        auto vertex_edge_offsets = v2e.vertex_edge_offsets();
        auto vertex_edge_indices = v2e.vertex_edge_indices();

        // Vertex slices
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Edge slices
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_rank = Cabana::slice<E_OWNER>(edges);
        auto e_cid = Cabana::slice<E_CIDS>(edges);
        auto e_pid = Cabana::slice<E_PID>(edges);
        auto e_layer = Cabana::slice<E_LAYER>(edges);

        /**
         * Allocate views. Each vertex can have at most six edges and six vertices
         * extending from it.
         * 
         * XXX - optimize this size estimate
         */
        size_t size = verts.extent(0);
        integer_view vids_view("vids_view", size*6);
        integer_view eids_view("eids_view", size*6);
        int_d vertex_idx("vertex_idx");
        int_d edge_idx("edge_idx");
        Kokkos::deep_copy(vertex_idx, 0);
        Kokkos::deep_copy(edge_idx, 0);
        
        int owned_vertices = _mesh->count(Own(), Vertex());
        Kokkos::parallel_for("gather neighboring vertex and edge IDs", Kokkos::RangePolicy<execution_space>(0, size),
            KOKKOS_LAMBDA(int i) {
            
            int vgid = v_gid(i);
            int vlid = vgid - vef_gid_start_d(rank, 0);
            if ((vlid >= 0) && (vlid < owned_vertices)) // Make sure we own the vertex
            {
                int offset = vertex_edge_offsets(vlid);
                int next_offset = (vlid + 1 < (int) vertex_edge_offsets.extent(0)) ? 
                          vertex_edge_offsets(vlid + 1) : 
                          (int) vertex_edge_indices.extent(0);

                // Loop over connected edges
                for (int j = offset; j < next_offset; j++)
                {
                    int elid = vertex_edge_indices(j); // Local edge ID
                    int egid = e_gid(elid);
                    int elayer = e_layer(elid);
                    int erank = e_rank(elid);
                    // Add edge to list of edges if it's on our layer
                    // AND it is not ghosted
                    if (elayer == level && erank == rank)
                    {
                        int edx = Kokkos::atomic_fetch_add(&edge_idx(), 1);
                        eids_view(edx) = egid;
                        // printf("R%d: for VGID %d, adding EGID %d\n", rank, vgid, egid);
                    }
                    
                    // Get the vertices connected by this edge
                    for (int v = 0; v < 2; v++)
                    {
                        int neighbor_vgid = e_vid(elid, v);

                        if (neighbor_vgid != vgid)
                        {
                            int vdx = Kokkos::atomic_fetch_add(&vertex_idx(), 1);
                            vids_view(vdx) = neighbor_vgid;
                            // printf("R%d: for VGID %d, adding VGID %d\n", rank, vgid, neighbor_vgid);
                        }
                    }
                }
            }
        });
        Kokkos::fence();
        int num_verts, num_edges;
        Kokkos::deep_copy(num_verts, vertex_idx);
        Kokkos::deep_copy(num_edges, edge_idx);

        // Resize views and remove unique values
        Kokkos::resize(vids_view, num_verts);
        Kokkos::resize(eids_view, num_edges);
        auto unique_vids = Utils::filter_unique(vids_view);
        auto unique_eids = Utils::filter_unique(eids_view);

        // Resize and copy into class variables
        /**
         * Resize and copy into class variables, keeping
         * existing values and removing any added duplicates
         */
        // Kokkos::resize(_vertex_ids, unique_vids.extent(0));
        // Kokkos::resize(_edge_ids, unique_eids.extent(0));
        // Kokkos::deep_copy(_vertex_ids, unique_)

        return unique_vids;
    }

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    // Entity halo type: 1 = edge, 2 = face
    const int _entity;

    // Level of tree to halo at
    const int _level;

    // Halo depth into neighboring processes
    const int _depth;

    // Mesh version this halo is built from
    int _halo_version;

    int _rank, _comm_size;
    
    // Vertex, Edge, and Face global IDs included in this halo
    integer_view _vertex_ids;
    integer_view _edge_ids;
    integer_view _face_ids;

    Kokkos::View<bool*> _boundary_edges;
    RankToEntityListMap _rank_to_boundary_vertices;
    RankToEntityListMap _rank_to_boundary_edges;
    
};

template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, Edge, int level, int depth)
{
    return Halo(mesh, 1, level, depth);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP