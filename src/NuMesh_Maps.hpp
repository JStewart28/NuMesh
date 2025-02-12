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
  \brief Builds a local CSR-like structure for mapping local and ghosted vertices to local and ghosted edges:
        View<int*> vertex_edge_offsets for the starting index of edges per vertex.
        View<int*> vertex_edge_indices for storing local edge indices in the adjacency list.
        IDs stored are local IDs
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
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::V2E: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        rebuild();
    };

    ~V2E() {}

    /**
     * Build the V2E
     */
    void rebuild()
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
        _offsets = integer_view("_offsets", num_vertices);
        auto offsets = _offsets;
        Kokkos::parallel_scan("Prefix sum for vertex offsets", Kokkos::RangePolicy<execution_space>(0, num_vertices),
            KOKKOS_LAMBDA(const int i, int& update, const bool final) {
                int count = vertex_edge_count(i);
                if (final) {
                    offsets(i) = update;
                }
                update += count;
            });
        Kokkos::fence();

        // Total number of edges in the adjacency list.
        int total_adj_edges = 0;
        Kokkos::parallel_reduce("Calculate total adjacency edges", Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(const int, int& count) {
                count = offsets(num_vertices - 1) + vertex_edge_count(num_vertices - 1);
            }, total_adj_edges);

        _indices = integer_view("vertex_edge_indices", total_adj_edges);
        // printf("R%d total adj edges: %d\n", rank, total_adj_edges);
        auto indices = _indices;

        // Step 3: Populate vertex-edge indices.
        integer_view current_offset("current_offset", num_vertices);
        Kokkos::deep_copy(current_offset, offsets);  // Copy offsets for modification.

        Kokkos::parallel_for("Populate adjacency list", Kokkos::RangePolicy<execution_space>(0, num_edges),
            KOKKOS_LAMBDA(const int e) {
                for (int v = 0; v < 2; ++v) {  // Loop over edge endpoints.
                    int vgid = e_vid(e, v);
                    int vlid = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                    if (vlid > -1)
                    {
                        int insert_idx = Kokkos::atomic_fetch_add(&current_offset(vlid), 1);
                        // if (rank == 0) printf("R%d insert idx: %d, v l/g: %d, %d\n", rank, insert_idx, vlid, vgid);
                        indices(insert_idx) = e;  // Store edge ID.
                    }
                }
            });
        Kokkos::fence();
        _map_version = _mesh->version();
    }

    auto offsets() {return _offsets;}
    auto indices() {return _indices;}
    int version() {return _map_version;}

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    // Mesh version this halo is built from
    int _map_version;

    int _rank, _comm_size;
    
    integer_view _offsets;
    integer_view _indices;
    
};

//---------------------------------------------------------------------------//
/*!
  \class V2F
  \brief Builds a local CSR-like structure for mapping local and ghosted vertices to local and ghosted faces:
        View<int*> offsets for the starting index of faces per vertex.
        View<int*> indices for storing local face indices in the adjacency list.
        IDs stored are local IDs
*/
template <class Mesh>
class V2F
{
  public:

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;

    V2F( std::shared_ptr<Mesh> mesh, const int level )
        : _mesh ( mesh )
        , _comm ( mesh->comm() )
        , _map_version ( mesh->version() )
        , _level ( level )
    {
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::V2F: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        rebuild();
    };

    ~V2F() {}

    /**
     * Build the V2F
     */
    void rebuild()
    {
        auto vertices = _mesh->vertices();
        auto faces = _mesh->faces();

        // Number of local vertices and faces.
        int num_vertices = vertices.size();
        int num_faces = faces.size();

        // Allocate vertex-face count and offsets.
        integer_view vertex_face_count("vertex_face_count", num_vertices);
        auto f_vid = Cabana::slice<F_VIDS>(faces); // Vertex IDs of each face
        auto f_level = Cabana::slice<F_LAYER>(faces);
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Step 1: Count edges per vertex
        int level = _level;
        auto vef_gid_start = _mesh->vef_gid_start();
        int vertex_gid_start = vef_gid_start(_rank, 0);
        Kokkos::parallel_for("Count faces per vertex", Kokkos::RangePolicy<execution_space>(0, num_faces),
            KOKKOS_LAMBDA(const int f) {
                if (level < f_level(f)) return; // Only consider faces at or above this level of the tree
                for (int v = 0; v < 3; ++v) {  // Loop over face endpoints.
                    int vgid = f_vid(f, v);
                    int vlid = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                    if (vlid > -1)
                        Kokkos::atomic_increment(&vertex_face_count(vlid));
                }
            });
        Kokkos::fence();
        
        // Step 2: Create vertex-face offsets using prefix sum.
        _offsets = integer_view("offsets", num_vertices);
        auto offsets = _offsets;
        Kokkos::parallel_scan("Prefix sum for vertex offsets", Kokkos::RangePolicy<execution_space>(0, num_vertices),
            KOKKOS_LAMBDA(const int i, int& update, const bool final) {
                int count = vertex_face_count(i);
                if (final) {
                    offsets(i) = update;
                }
                update += count;
            });
        Kokkos::fence();

        // Total number of faces in the adjacency list.
        int total_adj_faces = 0;
        Kokkos::parallel_reduce("Calculate total adjacency faces", Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(const int, int& count) {
                count = offsets(num_vertices - 1) + vertex_face_count(num_vertices - 1);
            }, total_adj_faces);

        _indices = integer_view("_indices", total_adj_faces);
        // printf("R%d total adj edges: %d\n", rank, total_adj_edges);
        auto indices = _indices;

        // Step 3: Populate vertex-faces indices.
        integer_view current_offset("current_offset", num_vertices);
        Kokkos::deep_copy(current_offset, offsets);  // Copy offsets for modification.

        Kokkos::parallel_for("Populate adjacency list", Kokkos::RangePolicy<execution_space>(0, num_faces),
            KOKKOS_LAMBDA(const int f) {
                if (level < f_level(f)) return; // Only consider faces at or above this level of the tree
                for (int v = 0; v < 3; ++v) {  // Loop over face endpoints.
                    int vgid = f_vid(f, v);
                    int vlid = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                    if (vlid > -1)
                    {
                        int insert_idx = Kokkos::atomic_fetch_add(&current_offset(vlid), 1);
                        // if (rank == 0) printf("R%d flid %d, insert idx: %d, v l/g: %d, %d\n", rank, f, insert_idx, vlid, vgid);
                        indices(insert_idx) = f;  // Store face LID.
                    }
                }
            });
        Kokkos::fence();

        _map_version = _mesh->version();
    }

    auto offsets() {return _offsets;}
    auto indices() {return _indices;}
    int version() {return _map_version;}
    int level() {return _level;}

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    // Mesh version and tree level this map is built from
    int _map_version, _level;

    int _rank, _comm_size;
    
    integer_view _offsets;
    integer_view _indices;
    
};

//---------------------------------------------------------------------------//
/*!
  \class V2V
  \brief Builds a local CSR-like structure for mapping local and ghosted vertices to all local and ghosted
            vertices they share a face with at the given level of the tree and higher (more refined):
        View<int*> offsets for the starting index of vertices neigboring vertex v.
        View<int*> indices for storing local face indices in the adjacency list.
        IDs stored are local IDs
*/
template <class Mesh>
class V2V
{
  public:

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;

    V2V( std::shared_ptr<Mesh> mesh, const int level )
        : _mesh ( mesh )
        , _comm ( mesh->comm() )
        , _map_version ( mesh->version() )
        , _level ( level )
    {
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::V2V: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        _level = -1;

        rebuild();
    };

    ~V2V() {}

    /**
     * Build the V2V map
     */
    void rebuild()
    {
        // Define the hash map type
        using PairType = std::pair<int, int>;
        using KeyType = uint64_t;  // Hashable key
        using MapType = Kokkos::UnorderedMap<KeyType, int, memory_space>;

        // Hash function to combine two integers into a single key
        auto hashFunction = KOKKOS_LAMBDA(int first, int second) -> KeyType {
            return (static_cast<KeyType>(first) << 32) | (static_cast<KeyType>(second) & 0xFFFFFFFF);
        };

        // Retrieve the vertex-to-face mapping
        auto v2f = NuMesh::Maps::V2F(_mesh, _level);
        auto face_offsets = v2f.offsets();
        auto face_indices = v2f.indices();

        // Get the face data array
        auto faces = _mesh->faces();
        auto vertices = _mesh->vertices();
        auto f_vid = Cabana::slice<F_VIDS>(faces); // Vertex IDs of each face.
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Get the number of vertices
        int num_vertices = vertices.size();

        // At worst, each vert is connected 6*3*(max tree level) verts
        int max_verts = num_vertices * 6 * 3 * _mesh->max_level();
        MapType vert_vert_map(max_verts);

        // Allocate offsets and indices (first pass to count unique neighbors)
        integer_view neighbor_counts("neighbor_counts", num_vertices);
        Kokkos::parallel_for("count_neighbors", Kokkos::RangePolicy<execution_space>(0, num_vertices),
            KOKKOS_LAMBDA(int vlid) {
                int offset = face_offsets(vlid);
                int next_offset = (vlid + 1 < (int)face_offsets.extent(0)) ? face_offsets(vlid + 1) : (int)face_indices.extent(0);
                printf("vlid: %d, offsets: %d, %d\n", vlid, offset, next_offset);
                int num_neighbors = 0;
                for (int i = offset; i < next_offset; i++)
                {
                    int flid = face_indices(i);
                    // Iterate over face vertices
                    for (int j = 0; j < 3; j++)
                    {
                        int vgid = f_vid(flid, j);
                        int vlid0 = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                        printf("vlid: %d, vlid0: %d\n", vlid, vlid0);
                        if ((vlid0 != vlid) && (vlid0 > -1)) { // Exclude self and vertices not in our AoSoA
                            auto hash_key = hashFunction(vlid, vlid0);
                            auto result = vert_vert_map.insert(hash_key, vlid0); // Add (vert, neighbor vert) pair to map
                            if (result.success()) 
                            {
                                num_neighbors++; // Pair not in map; increment num neigbors                            
                                printf("vlid: %d, nieghbor: %d\n", vlid, vlid0);
                            }
                        }
                    }
                }
                neighbor_counts(vlid) = num_neighbors;
            });

        // Compute offsets using exclusive scan
        _offsets = integer_view("offsets", num_vertices + 1);
        auto offsets = _offsets;
        Kokkos::parallel_scan("compute_offsets", Kokkos::RangePolicy<execution_space>(0, num_vertices),
            KOKKOS_LAMBDA(int i, int& update, bool final) {
            if (final) offsets(i) = update;
            update += neighbor_counts(i);
        });

        // Allocate indices
        _indices = integer_view("indices", _offsets(num_vertices));
        auto indices = _indices;

        // Second pass: Fill indices
        vert_vert_map.clear();
        Kokkos::parallel_for("fill_indices", Kokkos::RangePolicy<execution_space>(0, num_vertices),
            KOKKOS_LAMBDA(int vlid) {
                int offset = offsets(vlid);
                int face_offset = face_offsets(vlid);
                int next_face_offset = (vlid + 1 < (int)face_offsets.extent(0)) ? face_offsets(vlid + 1) : (int)face_indices.extent(0);
                int idx = offset;
                for (int i = face_offset; i < next_face_offset; i++)
                {
                    int flid = face_indices(i);
                    // Iterate over face vertices
                    for (int j = 0; j < 3; j++)
                    {
                        int vgid = f_vid(flid, j);
                        int vlid0 = Utils::get_lid(v_gid, vgid, 0, num_vertices);
                        if ((vlid0 != vlid) && (vlid0 > -1)) { // Exclude self and vertices not in our AoSoA
                            auto hash_key = hashFunction(vlid, vlid0);
                            auto result = vert_vert_map.insert(hash_key, vlid0); // Add (vert, neighbor vert) pair to map
                            if (result.success()) indices(idx++) = vlid0; // Pair not in map; add to indices                           
                        }
                    }
                }
            });
        Kokkos::fence();
        _map_version = _mesh->version();
    }

    auto offsets() {return _offsets;}
    auto indices() {return _indices;}
    int version() {return _map_version;}
    int level() {return _level;}

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    // Mesh version and tree level this map is built from
    int _map_version, _level;

    int _rank, _comm_size;
    
    integer_view _offsets;
    integer_view _indices;
    
};

} // end namespace Maps

} // end namespce NuMesh


#endif // NUMESH_MAPS_HPP