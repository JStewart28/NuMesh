#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
#include <NuMesh_Mesh.hpp>
#include <NuMhes_Maps.hpp>

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

    /**
     * guess parameter: guess to how many vertices will be in the halo
     * to avoid frequent resizing.
     */
    Halo( std::shared_ptr<Mesh> mesh, const int entity, const int level, const int depth,
          const int guess=50 )
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

        _num_vertices = guess;
        _num_edges = _num_vertices; // There will be about twice as many vertices as edges,
                                    // but a split edge adds two edges and one vertex
        _num_faces = _num_edges / 3; // Each face contains three edges

        int num_edges = _mesh->count(Own(), Edge());        

        _vertex_send_offsets = integer_view("_vertex_send_offsets", _comm_size);
        _vertex_send_ids = integer_view("_vertex_send_ids", _num_vertices);
        _edge_send_offsets = integer_view("_edge_send_offsets", _comm_size);
        _edge_send_ids =integer_view("_edge_send_ids", _num_edges);
        _face_send_offsets = integer_view("_face_send_offsets", _comm_size);
        _face_send_ids = integer_view("_face_send_ids", _num_faces);

        Kokkos::deep_copy(_vertex_send_offsets, -1);
        Kokkos::deep_copy(_vertex_send_ids, -1);
        Kokkos::deep_copy(_edge_send_offsets, -1);
        Kokkos::deep_copy(_edge_send_ids, -1);
        Kokkos::deep_copy(_face_send_offsets, -1);
        Kokkos::deep_copy(_face_send_ids, -1);

        _v2e = Maps::V2E<Mesh>(_mesh);

        build();
    };

    ~Halo()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
    }

    void build()
    {


    }

    /**
     * Find the edges and vertices that we *know* we need
     * to send to other processes, hence the name push.
     */
    void create_push_halo()
    {
        if (_halo_version != _mesh->version())
        {
            throw std::runtime_error(
                "NuMesh::Halo::create_push_halo: mesh and halo version do not match" );
        }

        const int rank = _rank;

        // Reset index counters
        _vdx = 0; _edx = 0; _fdx = 0;

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vids = Cabana::slice<E_VIDS>(edges);

        auto boundary_edges = _mesh->boundary_edges();
        auto neighbor_ranks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _mesh->neighbor_ranks());
        
        // Get vef_gid_start and copy to device
        auto vef_gid_start = _mesh->vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        int edge_count = _mesh->count(Own(), Edge());
        int vertex_count = _mesh->count(Own(), Vertex());

        /**
         * For each neighbor rank:
         *  1. Iterate over the boundary edges and create a list of vertex GIDs owned by the
         *      neighbor rank for which to seed the gather_local_neighbors function.
         *  2. Call gather local neighbors until 'depth' edges away are reached.
         * 
         * XXX - can this be done in parallel?
         */
        for (size_t i = 0; i < nieghbor_ranks.extent(0); i++)
        {
            int neighbor_rank = neighbor_ranks(i);
            int v_offset = _vdx;
            int e_offset = _edx;
            auto v_subview = Kokkos::subview(_vertex_send_offsets, neighbor_rank);
            Kokkos::deep_copy(v_subview, v_offset);
            auto e_subview = Kokkos::subview(_edge_send_offsets, neighbor_rank);
            Kokkos::deep_copy(e_subview, e_offset);
            
            // Create vertex list and index counter
            Kokkos::View<int*, memory_space> vert_seeds("vert_seeds", vertex_count);
            Kokkos::View<int, memory_space> counter("counter");
            Kokkos::deep_copy(vert_seeds, -1);
            Kokkos::deep_copy(counter, 0);

            Kokkos::parallel_for("populate vertex seeds",
                Kokkos::RangePolicy<execution_space>(0, boundary_edges.extent(0)),
                KOKKOS_LAMBDA(int edge_idx) {
                
                int egid = boundary_edges(edge_idx);
                int elid = Utils::get_lid(e_gid, egid, 0, edge_count);

                for (int v = 0; v < 2; v++)
                {
                    int vgid = e_vids(i, v);
                    int vertex_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start_d);

                    if (vertex_owner == neighbor_rank)
                    {
                        int idx = Kokkos::atomic_fetch_add(&counter(), 1);
                        vert_seeds(idx) = vgid;
                    }
                }
            });
            int max_idx;
            Kokkos::deep_copy(max_idx, counter);
            Kokkos::resize(vert_seeds, max_idx+1);

            auto added_verts = gather_local_neighbors(vert_seeds);

            // _vdx = ; 

            // auto added_verts = gather_local_neighbors(vert_seeds);
            // int d = 1;

            // while (d <= _depth)
            // {
            //     Kokkos::resize(vert_seeds, added_verts.extent(0));
            //     Kokkos::deep_copy(vert_seeds, added_verts);
                
            //     l++;

            // }
            // for (int l = 0; l <= _level; l++)
            // {

            // }
            // auto added_verts = gather_local_neighbors(vert_seeds);
        }
        
    }

    /**
     * Given a list of vertex GIDs and a level of the tree, update
     * _vertex_ids and _edge_ids with edges connected to vertices
     * in the input list and the vertices of those connected edges.
     * 
     * Returns a list of new vertex GIDs added to the halo.
     */
    integer_view gather_local_neighbors(integer_view verts)
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
        auto vef_gid_start = _mesh->vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        auto vertex_edge_offsets = _v2e.vertex_edge_offsets();
        auto vertex_edge_indices = _v2e.vertex_edge_indices();

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

    /**
     * Vertex-to-edge map
     */
    Maps::V2E<Mesh> _v2e;

    
    /**
     * Vertex, Edge, and Face global IDs included in this halo
     * 
     * offsets and IDs views: for each rank, specifies the 'offset' 
     *  into 'ids' where the vertex/edge/face GID data resides
     *          
     */
    integer_view _vertex_send_offsets;
    integer_view _vertex_send_ids;
    integer_view _edge_send_offsets;
    integer_view _edge_send_ids;
    integer_view _face_send_offsets;
    integer_view _face_send_ids;

    /**
     * Current indexes into *send_ids views. These are stored
     * so that we don't have to find where we left off when calling
     * the gather_local_neighbors iteratively to "depth" elements
     */
    int _vdx, _edx, _fdx;

    // Number of vertices, edges, and faces in the halo
    int _num_vertices, _num_edges, _num_faces; 
    
};

template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, Edge, int level, int depth)
{
    return Halo(mesh, 1, level, depth);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP