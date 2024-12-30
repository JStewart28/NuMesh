#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
#include <NuMesh_Mesh.hpp>
#include <NuMesh_Maps.hpp>

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
        , _v2e ( Maps::V2E<Mesh>(_mesh) )
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

        build();
    };

    ~Halo()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
    }

    void build()
    {
        create_push_halo();

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
        auto neighbor_ranks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _mesh->neighbors());
        
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
        for (size_t i = 0; i < neighbor_ranks.extent(0); i++)
        {
            int neighbor_rank = neighbor_ranks(i);

            // Set the offsets where the data for this rank will start in the indices views
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
                // if (rank == 3) printf("R%d: elg: %d, %d\n", rank, elid, egid);

                for (int v = 0; v < 2; v++)
                {
                    int vgid = e_vids(elid, v);
                    int vertex_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start_d);
                    // if (rank == 3) printf("R%d: elg: %d, %d, vg: %d, vowner: %d, neighbor rank: %d\n", rank, elid, egid,
                    //     vgid, vertex_owner, neighbor_rank);
                    if (vertex_owner == neighbor_rank)
                    {
                        int idx = Kokkos::atomic_fetch_add(&counter(), 1);
                        vert_seeds(idx) = vgid;
                    }
                }
            });

            int max_idx;
            Kokkos::deep_copy(max_idx, counter);
            Kokkos::resize(vert_seeds, max_idx);
            auto verts_unique = Utils::filter_unique(vert_seeds);
            // printf("R%d max idx: %d\n", rank, max_idx);

            // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, verts_unique.extent(0)),
            //     KOKKOS_LAMBDA(int i) {

            //     printf("R%d: rank %d: verts_unique %d: %d\n", rank, neighbor_rank, i, verts_unique(i));

            // });
            
            // Store true if added to the halo to avoid uniqueness searches
            Kokkos::View<bool*, memory_space> vb("vb", vertex_count);
            Kokkos::View<bool*, memory_space> eb("eb", edge_count);
            Kokkos::deep_copy(vb, false);
            Kokkos::deep_copy(eb, false);
            
            // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, added_verts.extent(0)),
            //     KOKKOS_LAMBDA(int i) {

            //     if (rank == 1) printf("R%d: added_verts(%d): %d\n", rank, i, added_verts(i));

            // });
            
            // _vdx = ; 

            //auto added_verts = gather_local_neighbors(vert_seeds);
            collect_entities(verts_unique, vb, eb);

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
           
            // break;
        }

        
        auto vertex_send_offsets = _vertex_send_offsets;
        auto vertex_send_ids = _vertex_send_ids;
        Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, neighbor_ranks.extent(0)),
            KOKKOS_LAMBDA(int i) {

            if (rank == 1) printf("R%d: rank: %d, offset: %d\n", rank, neighbor_ranks(i), vertex_send_offsets(neighbor_ranks(i)));

        });

        Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, _vdx),
            KOKKOS_LAMBDA(int i) {

            if (rank == 1) printf("R%d: offset %d: VGID: %d\n", rank, i, vertex_send_ids(i));

        });
        
    }

    /**
     * Calls gather_local_neighbors 'depth' number of times to collect
     * all entities needed for the halo
     * 
     * XXX - is there a way to do this with loops?
     */
    template <class VertexView, class bool_view>
    void collect_entities(const VertexView& v, bool_view& vb, bool_view& eb)
    {
        if (_depth > 5)
        {
            throw std::runtime_error(
                "NuMesh::Halo::collect_entities: a maxmimum halo depth of 5 is supported" );
        }

        if (_depth >= 1)
        {
            if (_rank == 1) printf("R%d: start 1: vdx: %d, edx: %d, v extent: %d\n", 
                _rank, _vdx, _edx, (int)v.extent(0));
            auto out1 = gather_local_neighbors(v, vb, eb);
            // End depth 1
            if (_depth >= 2)
            {
                if (_rank == 1) printf("R%d: start 2: vdx: %d, edx: %d, out1 extent: %d\n", 
                    _rank, _vdx, _edx, (int)out1.extent(0));
                auto out2 = gather_local_neighbors(out1, vb, eb);
                // End depth 2
                if (_depth >= 3)
                {
                    auto out3 = gather_local_neighbors(out2, vb, eb);
                    // End depth 3
                    if (_depth >= 4)
                    {
                        auto out4 = gather_local_neighbors(out3, vb, eb);
                        // End depth 4
                        if (_depth >= 5)
                        {
                            auto out5 = gather_local_neighbors(out4, vb, eb);
                            // End depth 5
                        }
                    }
                }
            }
        }
    }

    /**
     * Given a list of vertex GIDs and a level of the tree, update
     * _vertex_ids and _edge_ids with edges connected to vertices
     * in the input list and the vertices of those connected edges.
     * 
     * bool_views are of length num_vertices and num_edges, and are true
     * if the element LID is already present in the push halo of GIDs
     * 
     * Returns a list of new vertex GIDs added to the halo.
     */
    template <class bool_view>
    integer_view gather_local_neighbors(const integer_view& verts, bool_view& vb, bool_view& eb)
    {
        if (_halo_version != _mesh->version())
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: mesh and halo version do not match" );
        }

        const int level = _level, rank = _rank, comm_size = _comm_size;

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();
        auto boundary_edges = _mesh->boundary_edges();
        size_t num_boundary_edges = boundary_edges.extent(0);

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
        int owned_edges = _mesh->count(Own(), Edge());
        Kokkos::parallel_for("gather neighboring vertex and edge IDs", Kokkos::RangePolicy<execution_space>(0, size),
            KOKKOS_LAMBDA(int i) {
            
            int vgid = verts(i);
            if (rank == 1) printf("R%d: seed VGID: %d\n", rank, vgid);
            int vlid = vgid - vef_gid_start_d(rank, 0);
            if ((vlid >= 0) && (vlid < owned_vertices)) // If we own the vertex
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
                    // AND it is not already added
                    if ((elayer == level) && (erank == rank))
                    {
                        // auto old = atomic_compare_exchange(&obj, expected, desired);
                        bool val = Kokkos::atomic_compare_exchange(&eb(elid), false, true);
                        if (!val)
                        {
                            int edx = Kokkos::atomic_fetch_add(&edge_idx(), 1);
                            eids_view(edx) = egid;
                            // if (rank == 1) printf("R%d: for VGID %d, adding EGID %d\n", rank, vgid, egid);

                            // Get the other vertex connected by this edge
                            for (int v = 0; v < 2; v++)
                            {
                                int neighbor_vgid = e_vid(elid, v);
                                // if (rank == 1) printf("R%d: egid %d, checking neighbor_vgid %d: %d\n", rank, egid, v, neighbor_vgid);
                                int neighbor_vlid = neighbor_vgid - vef_gid_start_d(rank, 0);
                                int vertex_owner = Utils::owner_rank(Vertex(), neighbor_vgid, vef_gid_start_d);
                                // printf("R%d: EGID: %d, neighbor_vgid: %d, neighbor_vlid")
                                if ((neighbor_vgid != vgid) && (vertex_owner == rank)) // Checks this is the other vertex, and make sure we own it
                                {
                                    // if (rank == 1) printf("R%d: egid %d, neighbor_vgid %d: %d\n", rank, egid, v, neighbor_vgid);
                                    val = Kokkos::atomic_compare_exchange(&vb(neighbor_vlid), false, true);
                                    if (!val)
                                    {
                                        int vdx = Kokkos::atomic_fetch_add(&vertex_idx(), 1);
                                        vids_view(vdx) = neighbor_vgid;
                                        // if (rank == 1) printf("R%d: from edge %d and VGID %d, adding VGID %d\n", rank, egid, vgid, neighbor_vgid);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /**
             * If these are the initial seeds, we may not own the vertex, but the vertex
             * may be connected by an edge we own. Find this edge with a linear search through
             * the boundary edges
             * XXX - Make this search more efficient
             */
            else
            {
                for (size_t e = 0; e < num_boundary_edges; e++)
                {
                    int egid = boundary_edges(e);
                    int elid = egid - vef_gid_start_d(rank, 1);
                    if ((elid >= 0) && (elid < owned_edges))
                    {
                        // Ensure this edge is on our level
                        if (e_layer(elid) != level) continue;

                        // Check if this edge connects the vertex in question
                        for (int j = 0; j < 2; j++)
                        {
                            int exvgid = e_vid(elid, j);
                            // if (vgid == 12) printf("R%d: vgid: %d, exvgid: %d\n", rank, vgid, exvgid);
                            if (exvgid == vgid)
                            {
                                // Add the other vertex to our list of new vertices
                                int neighbor_vgid = e_vid(elid, !j);
                                int neighbor_vlid = neighbor_vgid - vef_gid_start_d(rank, 0);
                                bool val = Kokkos::atomic_compare_exchange(&vb(neighbor_vlid), false, true);
                                if (!val)
                                {
                                    int vdx = Kokkos::atomic_fetch_add(&vertex_idx(), 1);
                                    vids_view(vdx) = neighbor_vgid;
                                    if (rank == 1) printf("R%d: vids_view(%d) = %d\n", rank, vdx, neighbor_vgid);
                                }

                                // Add this edge to our list of edges
                                val = Kokkos::atomic_compare_exchange(&eb(elid), false, true);
                                if (!val)
                                {
                                    int edx = Kokkos::atomic_fetch_add(&edge_idx(), 1);
                                    eids_view(edx) = egid;
                                }
                            }
                        }
                    }
                }
            }
        });
        Kokkos::fence();
        int num_verts, num_edges;
        Kokkos::deep_copy(num_verts, vertex_idx);
        Kokkos::deep_copy(num_edges, edge_idx);
        // printf("R%d: num verts, edges: %d, %d\n", _rank, num_verts, num_edges);

        // Resize views and remove unique values
        Kokkos::resize(vids_view, num_verts);
        Kokkos::resize(eids_view, num_edges);
        // auto unique_vids = Utils::filter_unique(vids_view);
        // auto unique_eids = Utils::filter_unique(eids_view);

        /**
         * Add IDs to offset views starting at offsets
         * _vdx and _edx
         */
        auto v_subview = Kokkos::subview(_vertex_send_ids, std::make_pair(_vdx, _vdx + num_verts));
        Kokkos::deep_copy(v_subview, vids_view);
        auto e_subview = Kokkos::subview(_edge_send_ids, std::make_pair(_edx, _edx + num_edges));
        Kokkos::deep_copy(e_subview, eids_view);

        // Update values of _vdx and _edx
        _vdx += num_verts; _edx += num_edges;

        return vids_view;
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
     *  into 'ids' where the vertex/edge/face GID data resides.
     * IDs are global IDs.
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