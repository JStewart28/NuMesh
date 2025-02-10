#ifndef NUMESH_MESH_HPP
#define NUMESH_MESH_HPP

// XXX - Add mapping class.

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <algorithm> // For std::find
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <mpi.h>

#include <NuMesh_Utils.hpp>

#include "_hypre_parcsr_ls.h"

#ifndef AOSOA_SLICE_INDICES
#define AOSOA_SLICE_INDICES 1
#endif

// Constants for tuple/slice indices
#if AOSOA_SLICE_INDICES
    #define V_GID 0
    #define V_OWNER 1
    #define E_VIDS 0
    #define E_CIDS 1
    #define E_PID 2
    #define E_GID 3
    #define E_LAYER 4
    #define E_OWNER 5
    #define F_CID 0
    #define F_EIDS 1
    #define F_VIDS 2
    #define F_GID 3
    #define F_PID 4
    #define F_LAYER 5
    #define F_OWNER 6
#endif

#include <NuMesh_Types.hpp>

#include <limits>

namespace NuMesh
{

//---------------------------------------------------------------------------//
/*!
  \class Mesh
  \brief Unstructured triangle mesh
*/
template <class ExecutionSpace, class MemorySpace>
class Mesh
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;

    // Placeholder mesh_type identifier
    using mesh_type = double;

    //using Node = Cabana::Grid::Node;
    //using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, Node>;
    // using node_view = Kokkos::View<double***, device_type>;

    // For halo data
    // (global ID, to_rank) tuples
    using halo_aosoa = Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4>;

    // Note: Larger types should be listed first
    using vertex_data = Cabana::MemberTypes<int,       // Vertex global ID                                 
                                            int,       // Owning rank
                                            >;
    using edge_data = Cabana::MemberTypes<  int[3],    // Vertex global IDs of edge: (endpoint, endpoint, midpoint)
                                                       // Midpoint is populated when the edge is split    
                                            int[2],    // Child edge global IDs, going clockwise from
                                                       // the first vertex of the face
                                            int,       // Parent edge global ID
                                            int,       // Edge global ID
                                            int,       // Layer of the tree this edge lives on
                                            int,       // Owning rank
                                            >;
    using face_data = Cabana::MemberTypes<  int[4],    // Child face global IDs
                                            int[3],    // Edge global IDs that make up face 
                                            int[3],    // Vertex global IDs that make up the face
                                            int,       // Face global ID
                                            int,       // Parent face global ID 
                                            int,       // Layer of the tree this face lives on                       
                                            int,       // Owning rank
                                            >;
                                            
    // XXX Change the final parameter of particle_array_type, vector type, to
    // be aligned with the machine we are using
    using v_array_type = Cabana::AoSoA<vertex_data, memory_space, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, memory_space, 4>;
    using f_array_type = Cabana::AoSoA<face_data, memory_space, 4>;
    using size_type = typename memory_space::size_type;

    // Construct a mesh.
    Mesh( MPI_Comm comm ) : _comm( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
        //MPIX_Info_init(&_xinfo);
        //MPIX_Comm_init(&_xcomm, _comm);
        //MPIX_Comm_topo_init(_xcomm);

        _version = 0;

        // Mesh starts with no refinement
        _max_tree_level = 0;

        // Mesh starts with no haloing
        _halo_level = -1; _halo_depth = 0;

        _vef_gid_start = Kokkos::View<int*[3], memory_space>("_vef_gid_start", _comm_size);
        Kokkos::deep_copy(_vef_gid_start, -1);

        _owned_vertices = 0, _owned_edges = 0, _owned_faces = 0;
        _ghost_vertices = 0, _ghost_edges = 0, _ghost_faces = 0;
    };

    ~Mesh()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
    }

    /**
     * Sorts the owned domain of the edge and face AoSoAs by
     * layer of the tree the element lives on, from 
     * highest (0), to lowest layer
     */
    void _sort_by_layer()
    {
        auto e_layer = Cabana::slice<E_LAYER>(_edges);
        auto f_layer = Cabana::slice<F_LAYER>(_faces);

        auto sort_edges = Cabana::sortByKey( e_layer );
        Cabana::permute( sort_edges, _edges );
        auto sort_faces = Cabana::sortByKey( f_layer );
        Cabana::permute( sort_faces, _faces );
    }

    /**
     * Populates vector of neighbor ranks. Derives neighbor
     * ranks from boundary edges.
     * 
     * Populates vector of GIDS of boundary edges and faces
     * along our upper boundary, i.e. elements we KNOW are on
     * the boundary.
     */
    void _populate_boundary_elements()
    {
        auto e_vids = Cabana::slice<E_VIDS>(_edges);
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto f_vids = Cabana::slice<F_VIDS>(_faces);
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_level = Cabana::slice<F_LAYER>(_faces);
        const int rank = _rank;
        const auto vef_gid_start = _vef_gid_start;

        // Identify edges and vertices not owned by this rank
        Kokkos::View<bool*, memory_space> is_neighbor("is_neighbor", _comm_size);
        Kokkos::View<bool*, memory_space> is_b_edge("is_b_edge", _owned_edges);
        Kokkos::deep_copy(is_neighbor, false);
        Kokkos::deep_copy(is_b_edge, false);
        Kokkos::parallel_for("find", Kokkos::RangePolicy<execution_space>(0, _owned_edges),
            KOKKOS_LAMBDA(int i) {

            for (int v = 0; v < 2; v++)
            {
                int vgid = e_vids(i, v);
                int vertex_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start);

                if (vertex_owner != rank)
                {
                    Kokkos::atomic_store(&is_neighbor(vertex_owner), true);
                    Kokkos::atomic_store(&is_b_edge(i), true);

                    // Once we know this is a boundary edge we don't need to check other vertices
                    // because an edge will always have at most one remote vertex
                    break;
                }
            }
        });
        // Identify faces
        Kokkos::View<bool*, memory_space> is_b_face("is_b_face", _owned_faces);
        Kokkos::deep_copy(is_b_face, false);
        Kokkos::parallel_for("find", Kokkos::RangePolicy<execution_space>(0, _owned_faces),
            KOKKOS_LAMBDA(int i) {

            // if (rank == 0) printf("R%d: checking fgid %d, level %d\n", rank, f_gid(i), f_level(i));

            for (int v = 0; v < 3; v++)
            {
                int vgid = f_vids(i, v);
                int vertex_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start);

                if (vertex_owner != rank)
                {
                    Kokkos::atomic_store(&is_b_face(i), true);

                    // Once we know this is a boundary face we don't need to check other vertices
                    break;
                }
            }
        });

        // Count neighbors
        int num_neighbors = 0;
        Kokkos::parallel_reduce("count_neighbors", Kokkos::RangePolicy<execution_space>(0, is_neighbor.extent(0)),
            KOKKOS_LAMBDA(const int i, int& local_count) {

            if (is_neighbor(i)) {
                local_count += 1;
            }
        }, num_neighbors);

        // Count boundary edges
        int num_b_edges = 0;
        Kokkos::parallel_reduce("count_boundary_edges", Kokkos::RangePolicy<execution_space>(0, is_b_edge.extent(0)),
            KOKKOS_LAMBDA(const int i, int& local_count) {

            if (is_b_edge(i)) {
                local_count += 1;
            }
        }, num_b_edges);

        // Count boundary faces
        int num_b_faces = 0;
        Kokkos::parallel_reduce("count_boundary_faces", Kokkos::RangePolicy<execution_space>(0, is_b_face.extent(0)),
            KOKKOS_LAMBDA(const int i, int& local_count) {

            if (is_b_face(i)) {
                local_count += 1;
            }
        }, num_b_faces);

        // Populate neighbors
        Kokkos::View<int*, memory_space> neighbors("neighbors", num_neighbors);
        Kokkos::View<int, memory_space> counter_n("counter_n");
        Kokkos::deep_copy(counter_n, 0);
        Kokkos::parallel_for("populate_neighbors", Kokkos::RangePolicy<execution_space>(0, is_neighbor.extent(0)),
            KOKKOS_LAMBDA(int i) {

            bool is_n = is_neighbor(i);
            if (is_n)
            {
                int idx = Kokkos::atomic_fetch_add(&counter_n(), 1);
                neighbors(idx) = i; 
            }
        });
        Kokkos::sort(neighbors);

        // Populate boundary edges
        Kokkos::View<int*, memory_space> boundary_edges("boundary_edges", num_b_edges);
        Kokkos::View<int, memory_space> counter_e("counter_e");
        Kokkos::deep_copy(counter_e, 0);
        Kokkos::parallel_for("populate_boundary_edges", Kokkos::RangePolicy<execution_space>(0, is_b_edge.extent(0)),
            KOKKOS_LAMBDA(int i) {

            bool is_n = is_b_edge(i);
            if (is_n)
            {
                int idx = Kokkos::atomic_fetch_add(&counter_e(), 1);
                boundary_edges(idx) = e_gid(i); 
            }
        });
        Kokkos::sort(boundary_edges);

        // Populate boundary faces
        Kokkos::View<int*, memory_space> boundary_faces("boundary_faces", num_b_faces);
        Kokkos::View<int, memory_space> counter_f("counter_f");
        Kokkos::deep_copy(counter_f, 0);
        Kokkos::parallel_for("populate_boundary_faces", Kokkos::RangePolicy<execution_space>(0, is_b_face.extent(0)),
            KOKKOS_LAMBDA(int i) {

            bool is_n = is_b_face(i);
            if (is_n)
            {
                int idx = Kokkos::atomic_fetch_add(&counter_f(), 1);
                boundary_faces(idx) = f_gid(i); 
            }
        });
        Kokkos::sort(boundary_faces);

        _neighbors = neighbors;
        _boundary_edges = boundary_edges;
        _boundary_faces = boundary_faces;
    }

    /**
     * Create faces from vertices and edges
     * Each vertex is associated with at least one face
     */
    void _createFaces()
    {
        /* Each vertex contributes 2 faces */
        _faces.resize(_owned_faces);
    
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        auto v_owner = Cabana::slice<V_OWNER>(_vertices);

        auto e_vid = Cabana::slice<E_VIDS>(_edges); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_owner = Cabana::slice<E_OWNER>(_edges);

        auto f_egids = Cabana::slice<F_EIDS>(_faces);
        auto f_vgids = Cabana::slice<F_VIDS>(_faces);
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_parent = Cabana::slice<F_PID>(_faces);
        auto f_child = Cabana::slice<F_CID>(_faces);
        auto f_owner = Cabana::slice<F_OWNER>(_faces);
        auto f_layer = Cabana::slice<F_LAYER>(_faces);

        int rank = _rank;
        const auto vef_gid_start = _vef_gid_start;
        Kokkos::parallel_for("Initialize faces", Kokkos::RangePolicy<execution_space>(0, _vertices.size()), KOKKOS_LAMBDA(int i) {
            // Face 1: "left" face; face 2: "right" face   
            int f_lid;

            // Find face 1 values
            // Get the three vertices and edges for face1
            int v_gid0, v_gid1, v_gid2;
            int e_gid0, e_lid0, e_gid1, e_gid2, e_lid2;
            v_gid0 = v_gid(i);
            // Follow first edge to get next vertex
            e_gid0 = v_gid0*3; e_lid0 = e_gid0 - vef_gid_start(rank, 1);
            v_gid1 = e_vid(e_lid0, 1);
            // Use second vertex to get next edge
            e_gid1 = v_gid1*3+2;
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - vef_gid_start(rank, 1);
            v_gid2 = e_vid(e_lid2, 1);
            
            // Populate face 1 values
            f_lid = i*2;
            f_egids(f_lid, 0) = e_gid0; f_egids(f_lid, 1) = e_gid1; f_egids(f_lid, 2) = e_gid2;
            f_vgids(f_lid, 0) = v_gid0; f_vgids(f_lid, 1) = v_gid1; f_vgids(f_lid, 2) = v_gid2;
            f_gid(f_lid) = f_lid + vef_gid_start(rank, 2);
            f_parent(f_lid) = -1;
            f_child(f_lid, 0) = -1; f_child(f_lid, 1) = -1; f_child(f_lid, 2) = -1; f_child(f_lid, 3) = -1;
            f_owner(f_lid) = rank;
            f_layer(f_lid) = 0;
            // printf("v_gl0: (%d, %d), e_gl0: (%d, %d), v_gid1: %d, e_gid1: %d, v_gid2: %d, e_gid2: %d, R%d\n",
            //     v_gid0, v_lid0, e_gid0, e_lid0, v_gid1, e_gid1, v_gid2, e_gid2, rank);
            // if (f_gid(f_lid) == 103)
            // {
            //     printf("F1-103-gid: %d, lid: %d, v(%d, %d, %d), e(%d, %d, %d)\n", f_gid(f_lid), f_lid,
            //         f_vgids(f_lid, 0), f_vgids(f_lid, 1), f_vgids(f_lid, 2),
            //         f_egids(f_lid, 0), f_egids(f_lid, 1), f_egids(f_lid, 2));
            // }

            // Find face 2 values
            // Edge 2 on face 1 is edge 0 on face 2
            // v_gid0 is the same
            e_gid0 = e_gid2; e_lid0 = e_lid2;
            // Get vertex 1 global ID the same way
            v_gid1 = e_vid(e_lid0, 1);
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - vef_gid_start(rank, 1);
            v_gid2 = e_vid(e_lid2, 1);
            // DIFFERENT: Use vertex 2 to get edge 1. Edge 1 the first edge of vertex 2
            e_gid1 = v_gid2*3;

            // Populate face 2 values
            f_lid = i*2+1;
            f_egids(f_lid, 0) = e_gid0; f_egids(f_lid, 1) = e_gid1; f_egids(f_lid, 2) = e_gid2;
            f_vgids(f_lid, 0) = v_gid0; f_vgids(f_lid, 1) = v_gid1; f_vgids(f_lid, 2) = v_gid2;
            f_gid(f_lid) = f_lid + vef_gid_start(rank, 2);
            f_parent(f_lid) = -1;
            f_child(f_lid, 0) = -1; f_child(f_lid, 1) = -1; f_child(f_lid, 2) = -1; f_child(f_lid, 3) = -1;
            f_owner(f_lid) = rank;
            f_layer(f_lid) = 0;

            // if (f_gid(f_lid) == 103)
            // {
            //     printf("F2-103-gid: %d, lid: %d, v(%d, %d, %d), e(%d, %d, %d)\n", f_gid(f_lid), f_lid,
            //         f_vgids(f_lid, 0), f_vgids(f_lid, 1), f_vgids(f_lid, 2),
            //         f_egids(f_lid, 0), f_egids(f_lid, 1), f_egids(f_lid, 2));
            // }
        });

        //printf("Num verts: %d, edges: %d, faces: %d\n", _v_array.size(), _e_array.size(), _f_array.size());
    }

    /**
     * After the vertex and edge connectivity has been set and
     * global IDs assigned, create faces from the edges and vertices.
     *  1. Create the faces
     *  2. Update the edges to reflect which faces they are part of
     *  3. Gather face global IDs from remotely owned faces to the edges
     */
    void _finializeInit()
    {
        _createFaces();
        // _sort_by_layer();
        _populate_boundary_elements();
        printf("R%d: num_verts: %d, func: %d\n", _rank, _owned_vertices, this->count(Own(), Vertex()));
    }

        /**
     * Refines faces
     * 
     * @param fid: Kokkos view holding the global IDs
     *             of faces to be refined
     * 
     */
    template <class View>
    void _refineFaces(View fgids)
    {
        const int rank = _rank, comm_size = _comm_size;
        const int num_face_refinements = fgids.extent(0);
        auto vef_gid_start = _vef_gid_start;

        // Lambda capture variables
        int owned_edges = _owned_edges; // Updated after edge refinement
        int owned_faces = _owned_faces;
        int ghost_edges = -1;           // Updated after haloing edges
        
        /********************************************************
         * Phase 1.0: Collect all edges that need to be refined,
         * then refine them in parallel
         *******************************************************/
        using int_vector_d = Kokkos::View<int*, memory_space>;
        using int_d = Kokkos::View<int, memory_space>;

        // (global ID, from rank) tuples
        using int_vector_aosoa = Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4>;

        /**
         * List of locally-owned face IDs this process needs to split
         * Since we don't know the size a priori, make it as large as the
         * total number of face refinements.
         */
        // 
        int_vector_d local_face_lids("local_face_lids", num_face_refinements);

        /**
         * edge_needrefine:
         *      values > 0: This edge is owned and can be refined locally
         * 
         * remote_edge_needrefine:
         *      Holds GIDs of remote edges that must be refined.
         *      XXX - currently assumes remote edges encompass no more than 1/5 of owned edges
         * 
         * distributor_export_ranks:
         *      Maps to remote_edge_needrefine. Holds the destination rank of the remote
         *          edge that must be refined.
         *      XXX - currently assumes remote edges encompass no more than 1/5 of owned edges
         */
        int_vector_d edge_needrefine("edge_needrefine", _owned_edges);
        Kokkos::deep_copy(edge_needrefine, 0);


        const int remote_edge_needrefine_size = _owned_edges/5;
        int_vector_aosoa export_edges_aosoa("remote_edges_send", remote_edge_needrefine_size);
        int_vector_d distributor_export_ranks("distributor_export_ranks", remote_edge_needrefine_size);
        Kokkos::deep_copy(distributor_export_ranks, -1);
        
        // Slices we need
        auto f_eid_slice0 = Cabana::slice<F_EIDS>(_faces);
        auto f_cid_slice0 = Cabana::slice<F_CID>(_faces);
        auto remote_edge_slice = Cabana::slice<0>(export_edges_aosoa);
        auto remote_edge_rank_slice = Cabana::slice<1>(export_edges_aosoa);
        Cabana::deep_copy(remote_edge_slice, -1);
        Cabana::deep_copy(remote_edge_rank_slice, -1);

        /**
         * Counters for new edges and vertices
         */
        int_d vert_counter("vert_counter");
        int_d edge_counter("edge_counter");
        int_d face_counter("face_counter");
        int_d remote_edge_counter("remote_edge_counter");
        Kokkos::deep_copy(vert_counter, 0);
        Kokkos::deep_copy(edge_counter, 0);
        Kokkos::deep_copy(face_counter, 0);
        Kokkos::deep_copy(remote_edge_counter, 0);
        
        Kokkos::parallel_for("populate edge_needrefine", Kokkos::RangePolicy<execution_space>(0, num_face_refinements),
            KOKKOS_LAMBDA(int i) {
            
            int f_lid = fgids(i) - vef_gid_start(rank, 2);

            if ((f_lid >= 0) && (f_lid < owned_faces)) // Make sure we own the face
            {
                // If a face has already been refined (i.e. has children), don't refine it again
                if (f_cid_slice0(f_lid, 0) != -1) return;

                int index = Kokkos::atomic_fetch_add(&face_counter(), 1);
                local_face_lids(index) = f_lid;

                // Each face refinement adds 3 new edges
                Kokkos::atomic_add(&edge_counter(), 3);
                for (int j = 0; j < 3; j++)
                {
                    int ex_gid = f_eid_slice0(f_lid, j);
                    int ex_lid = ex_gid - vef_gid_start(rank, 1);
                    if ((ex_lid >= 0) && (ex_lid < owned_edges))
                    {
                        /**
                         * Case 1: We own this edge
                         */

                        // Record this edge needs to be refined
                        int edge_counted = Kokkos::atomic_fetch_add(&edge_needrefine(ex_lid), 1);
                        // If edge_counted already equals 1, don't increment the new vert counter
                        if (edge_counted == 0)
                        {
                            // Each edge refinement contributes one new vertex and 2 new edges
                            Kokkos::atomic_increment(&vert_counter());
                            Kokkos::atomic_add(&edge_counter(), 2);
                        }
                    }
                    else
                    {
                        /**
                         * Case 2: We do not own this edge and need to tell another
                         * process to refine it
                         */
                        int idx = Kokkos::atomic_fetch_add(&remote_edge_counter(), 1);
                        remote_edge_slice(idx) = f_eid_slice0(f_lid, j);
                        remote_edge_rank_slice(idx) = rank;

                        // Find the remote rank this edge belongs to
                        int export_rank = -1;
                        for (int r = 0; r < comm_size; r++)
                        {
                            if (r == rank) continue;
                            if (ex_gid >= vef_gid_start(r, 1))
                            {
                                export_rank = r;
                            }
                        }
                        distributor_export_ranks(idx) = export_rank;
                    }
                }
            }
        });
        Kokkos::fence();

        // How many remote edges we need refined
        int remote_edges_refined;
        Kokkos::deep_copy(remote_edges_refined, remote_edge_counter);
        
        /********************************************************************************
         * Communicate remote edges that must be refined. We know how we are sending to
         * but not who we are receiving from.
         * 
         *  - remote_edge_needrefine: Kokkos view of length remote_edges that holds GIDs
         *      remote edges that must be refined
         *  - distributor_export_ranks: Maps to remote_edge_needrefine. Kokkos view of length
         *      remote_edges that holds the destination (owner) rank of each remote edge
         *      that must be refined.
         *  - 
         *******************************************************************************/
        
        auto distributor = Cabana::Distributor<memory_space>( _comm, distributor_export_ranks );

        const int distributor_total_num_import = distributor.totalNumImport();
        // printf("R%d: distributor i/e: %d, %d\n", _rank, distributor_total_num_import, distributor_export_ranks.extent(0));
        int_vector_aosoa distributor_edges_import("distributor_edges_import", distributor_total_num_import);

        Cabana::migrate(distributor, export_edges_aosoa, distributor_edges_import);

        // if (rank == 2)
        // {
            // for (int i = 0; i < (int) export_edges_aosoa.size(); i++)
            // {
            //     if (remote_edge_slice(i) != -1) printf("R%d: requesting edge %d to R%d\n", rank, remote_edge_slice(i), distributor_export_ranks(i));
            // }
            // auto recv_slice = Cabana::slice<0>(distributor_edges_import);
            // auto recv_rank_slice = Cabana::slice<1>(distributor_edges_import);
            // for (int i = 0; i < (int) distributor_edges_import.size(); i++)
            // {
            //     if (recv_slice(i) != -1) printf("R%d: asking edge %d from R%d\n", rank, recv_slice(i), recv_rank_slice(i));
            // }
        // }

        // Reset remote edge counter to track how many edges we need to send back
        Kokkos::deep_copy(remote_edge_counter, 0);

        // Iterate through our received edges and mark them for refinement
        auto distributor_edges_import_slice = Cabana::slice<0>(distributor_edges_import);
        auto distributor_ranks_import_slice = Cabana::slice<1>(distributor_edges_import);
        Kokkos::parallel_for("add remotes to edge_needrefine", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_import),
            KOKKOS_LAMBDA(int i) {

            int elid = distributor_edges_import_slice(i) - vef_gid_start(rank, 1);
            
            // Record this edge needs to be refined
            int edge_counted = Kokkos::atomic_fetch_add(&edge_needrefine(elid), 1);

            // Incrment counter to store the number of sends that need to be made to remote processes
            Kokkos::atomic_fetch_add(&remote_edge_counter(), 1);

            // If edge_counted already equals 1, don't increment new vert and edge counters
            if (edge_counted == 0)
            {
                // Each edge refinement contributes one new vertex and 2 new edges
                Kokkos::atomic_increment(&vert_counter());
                Kokkos::atomic_add(&edge_counter(), 2);
            }
        });

        // Update new vertex, edge, and face values
        int new_vertices, new_edges, remote_edges_send, face_refinements;
        Kokkos::deep_copy(new_vertices, vert_counter);
        Kokkos::deep_copy(new_edges, edge_counter);
        Kokkos::deep_copy(remote_edges_send, remote_edge_counter);
        Kokkos::deep_copy(face_refinements, face_counter);

        // printf("R%d old FGID space: %d to %d\n", rank, vef_gid_start(_rank, 2), vef_gid_start(_rank, 2)+_owned_faces);


        // Store new local ID (LID) start values for each array
        const int v_new_lid_start = _owned_vertices;
        const int e_new_lid_start = _owned_edges;
        const int f_new_lid_start = _owned_faces;

        // Resize arrays
        _owned_faces += face_refinements*4; // Each face refinement creates 4 new faces
        _faces.resize(_owned_faces);
        _owned_edges += new_edges;
        owned_edges = _owned_edges;
        _ghost_edges = remote_edges_refined * 3; // Each remotely refined edge will return the parent and two children
        _edges.resize(_owned_edges + _ghost_edges);
        _owned_vertices += new_vertices;
        _vertices.resize(_owned_vertices);

        // Vertex slices
        auto v_rank = Cabana::slice<V_OWNER>(_vertices);
        auto v_gid = Cabana::slice<V_GID>(_vertices);

        // Edge slices
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_vid = Cabana::slice<E_VIDS>(_edges);
        auto e_rank = Cabana::slice<E_OWNER>(_edges);
        auto e_cid = Cabana::slice<E_CIDS>(_edges);
        auto e_pid = Cabana::slice<E_PID>(_edges);
        auto e_layer = Cabana::slice<E_LAYER>(_edges);

        // Face slices
        auto f_cid = Cabana::slice<F_CID>(_faces);
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_eid = Cabana::slice<F_EIDS>(_faces);
        auto f_vid = Cabana::slice<F_VIDS>(_faces);
        auto f_pid = Cabana::slice<F_PID>(_faces);
        auto f_layer = Cabana::slice<F_LAYER>(_faces);
        auto f_owner = Cabana::slice<F_OWNER>(_faces);

        /**
         * Set global ID values in the expanded portions of the edge and face AoSoAs to -1.
         * They default to 0 which breaks the GID index searching we need
         * to do when assigning edges to vertices
         */
        Kokkos::parallel_for("Set EGID to -1", Kokkos::RangePolicy<execution_space>(e_new_lid_start, _owned_edges),
            KOKKOS_LAMBDA(int i) { e_gid(i) = -1; });
        Kokkos::parallel_for("Set FGID to -1", Kokkos::RangePolicy<execution_space>(f_new_lid_start, _owned_faces),
            KOKKOS_LAMBDA(int i) { f_gid(i) = -1; });


        // Update global IDs before making edits, but store the old global IDs
        // so we know how to fix our ghosted edge IDs
        Kokkos::View<int*[3], memory_space> vef_gid_start_old_d("vef_gid_start_old_d", _comm_size);
        Kokkos::deep_copy(vef_gid_start_old_d, vef_gid_start);
        _updateGlobalIDs();
        
        // printf("R%d new VEF: %d, %d, %d\n", rank, new_vertices, new_edges, face_refinements*4);
        // printf("R%d new VGID space: %d to %d\n", rank, _vef_gid_start(_rank, 0), _vef_gid_start(_rank, 0)+_owned_vertices);
        // printf("R%d new EGID space: %d to %d\n", rank, _vef_gid_start(_rank, 1), _vef_gid_start(_rank, 1)+_owned_edges);
        // printf("R%d new FGID space: %d to %d\n", rank, _vef_gid_start(_rank, 2), _vef_gid_start(_rank, 2)+_owned_faces);
        
        // Update to new _vef_gid_start
        vef_gid_start = _vef_gid_start;

        /******************************
         * Populate new vertices
         *****************************/
        Kokkos::parallel_for("populate new vertices", Kokkos::RangePolicy<execution_space>(0, new_vertices),
            KOKKOS_LAMBDA(int i) {
            
            int v_l = v_new_lid_start + i;
            int v_g = v_l + vef_gid_start(rank, 0);
            v_gid(v_l) = v_g;
            v_rank(v_l) = rank;
            // printf("R%d added VGID %d\n", rank, v_g);

        });

        /******************************
         * Populate new Edges
         *****************************/

        // Step 1: Refine existing edges
        Kokkos::deep_copy(edge_counter, 0);
        Kokkos::parallel_for("refine_edges", Kokkos::RangePolicy<execution_space>(0, e_new_lid_start),
            KOKKOS_LAMBDA(int i) {
            
            // Check if this edge needs to be split
            if (edge_needrefine(i) == 0) return;
            
            // Create new edge local IDs
            int offset = Kokkos::atomic_fetch_add(&edge_counter(), 2);
            int ec_lid0 = e_new_lid_start + offset;
            int ec_lid1 = e_new_lid_start + offset + 1;
            int ec_gid0 = vef_gid_start(rank, 1) + ec_lid0;
            int ec_gid1 = vef_gid_start(rank, 1) + ec_lid1;
            int ec_layer = e_layer(i) + 1;

            // if (rank == 1) printf("R%d refining edge %d: new edges %d, %d (offset %d)\n", rank, i+vef_gid_start(rank, 1), ec_gid0, ec_gid1, offset);

            // Global IDs = global ID start + local ID
            e_gid(ec_lid0) = ec_gid0;
            e_gid(ec_lid1) = ec_gid1;

            // Set parent edges for the split edges
            e_pid(ec_lid0) = i + vef_gid_start(rank, 1);
            e_pid(ec_lid1) = i + vef_gid_start(rank, 1);

            // Set child edges for the split edges
            e_cid(ec_lid0, 0) = -1; e_cid(ec_lid0, 1) = -1;
            e_cid(ec_lid1, 0) = -1; e_cid(ec_lid1, 1) = -1;
            
            // Set these edges to be the child edges of their parent edge
            e_cid(i, 0) = ec_gid0; e_cid(i, 1) = ec_gid1;

            // Owning rank
            e_rank(ec_lid0) = rank; e_rank(ec_lid1) = rank;

            // Layer of tree
            e_layer(ec_lid0) = ec_layer; e_layer(ec_lid1) = ec_layer;

            /**
             * Set vertices:
             *  ec_lid0: (First vertex of parent edge, new vertex, -1)
             *  ec_lid1: (new vertex, second vertex of parent edge, -1)
             */
            offset /= 2;
            int new_vgid = vef_gid_start(rank, 0) + v_new_lid_start + offset;
            e_vid(ec_lid0, 0) = e_vid(i, 0); e_vid(ec_lid0, 1) = new_vgid;
            e_vid(ec_lid1, 0) = new_vgid; e_vid(ec_lid1, 1) = e_vid(i, 1);
            e_vid(ec_lid0, 2) = -1; e_vid(ec_lid1, 2) = -1;

            // Set the parent edge third vertex value
            e_vid(i, 2) = new_vgid;
            // printf("R%d: pe%d v(%d, %d, %d)\n", rank, i+e_gid_start, e_vid(i, 0), e_vid(i, 1), e_vid(i, 2));
        });

        /**
         * Send the modified edges (parent and two children) back to any process that
         * originally requested them.
         * 
         * This value is distributor_total_num_import (how many edges other ranks asked us to refine) * 3
         */
        int_vector_d halo_export_ids("halo_export_ids", distributor_total_num_import*3);
        int_vector_d halo_export_ranks("halo_export_ranks", distributor_total_num_import*3);
        Kokkos::deep_copy(halo_export_ids, -1);
        Kokkos::deep_copy(halo_export_ranks, -1);
        Kokkos::parallel_for("populate halo_export_ids", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_import),
            KOKKOS_LAMBDA(int i) {
            
            int idx = i * 3;

            // Modify the global ID because GIDs have been updated in the mesh but 
            // not in the distributor receive buffer
            Utils::updateGlobalID(Edge(), &distributor_edges_import_slice(i), vef_gid_start, vef_gid_start_old_d);

            // Parent edge LID
            int elid = Utils::get_lid(e_gid, distributor_edges_import_slice(i), 0, owned_edges);
            assert(elid != -1);
            halo_export_ids(idx) = elid;

            // First child edge LID
            int clid = e_cid(elid, 0) - vef_gid_start(rank, 1);
            halo_export_ids(idx+1) = clid;

            // Second child edge LID
            clid = e_cid(elid, 1) - vef_gid_start(rank, 1);
            halo_export_ids(idx+2) = clid;

            // Set the export ranks
            for (int j = 0; j < 3; j++)
                halo_export_ranks(idx+j) = distributor_ranks_import_slice(i);
        });

        
        /**
         * \param neighbor_ranks List of ranks this rank will send to and receive
            from. This list can include the calling rank. This is effectively a
            description of the topology of the point-to-point communication
            plan. The elements in this list must be unique.
         */
       
        // Copy ranks received from slice to host memory
        Cabana::AoSoA<Cabana::MemberTypes<int>, Kokkos::HostSpace, 4> distributor_ranks_import_h("distributor_ranks_import_h", distributor_total_num_import);
        auto distributor_ranks_import_h_slice = Cabana::slice<0>(distributor_ranks_import_h);
        Cabana::deep_copy(distributor_ranks_import_h_slice, distributor_ranks_import_slice);
        std::vector<int> neighbor_ranks = {_rank};
        
        // Add ranks we received from
        for (int i = 0; i < distributor_total_num_import; i++)
        {
            int to_rank = distributor_ranks_import_h_slice(i);
            if ((to_rank != -1) && (std::find(neighbor_ranks.begin(), neighbor_ranks.end(), to_rank) == neighbor_ranks.end()))
            {
                neighbor_ranks.push_back(to_rank);
                //printf("R%d adding neighbor %d\n", rank, to_rank);
            }
        }
       
        // Add ranks we sent to
        auto distributor_export_ranks_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), distributor_export_ranks);
        for (int i = 0; i < remote_edges_refined; i++)
        {
            // Add ranks we sent to
            int to_rank = distributor_export_ranks_host(i);
            //printf("R%d send to: %d\n", rank, to_rank);
            if ((to_rank != -1) && (std::find(neighbor_ranks.begin(), neighbor_ranks.end(), to_rank) == neighbor_ranks.end()))
            {
                neighbor_ranks.push_back(to_rank);
                //printf("R%d adding neighbor %d\n", rank, to_rank);
            }
        }
        // printf("R%d: neighbors: %d, %d, %d, %d\n", _rank, neighbor_ranks[0], neighbor_ranks[1], neighbor_ranks[2], neighbor_ranks[3]);
        // printf("R%d num exports: %d\n", _rank, halo_export_ids.extent(0));

        // printf("R%d: neighbors: (size %d): %d, %d, %d, %d\n", rank, neighbor_ranks.size(), neighbor_ranks[0], neighbor_ranks[1],neighbor_ranks[2],neighbor_ranks[3]);
        // printf("R%d: AoSoA size: %d, owned: %d, ghost: %d eids: %d, eranks: %d\n", _rank,
        //     (int)_edges.size(), _owned_edges, _ghost_edges, halo_export_ids.extent(0), distributor_export_ranks1.extent(0));
        // Kokkos::View<int*, memory_space> exportids("exportids", 3);
        // Kokkos::View<int*, memory_space> rankids("rankids", 3);
        // Kokkos::parallel_for("populate halo_export_ids", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_import),
        //     KOKKOS_LAMBDA(int i) {
            
        //     exportids(i) = 1;
        //     rankids(i) = rank;

        // });

        // for (int i = 0; i < halo_export_ranks.extent(0); i++)
        // {
        //     printf("R%d exporting %d to rank %d\n", rank, halo_export_ids(i), halo_export_ranks(i));
        // }

        auto edge_halo = Cabana::Halo<memory_space>(_comm, _owned_edges, halo_export_ids,
            halo_export_ranks, neighbor_ranks);

        // printf("R%d halo local/ghost: %d, %d, import/export: %d, %d\n", _rank,
        //     edge_halo.numLocal(), edge_halo.numGhost(),
        //     edge_halo.totalNumImport(), edge_halo.totalNumExport());
        // if (rank == 1)
        // {
        //     printf("***************BEFORE GATHER***************\n");
        //     printEdges(2, 0);
        // }

        Cabana::gather(edge_halo, _edges);

        // if (rank == 1)
        // {
        //     printf("***************BEFORE***************\n");
        //     printEdges(3, 0);
        // }

        // Update class and lambda capture variables
        _owned_edges = edge_halo.numLocal(); _ghost_edges = edge_halo.numGhost();
        owned_edges = _owned_edges; ghost_edges = _ghost_edges;
        int num_edges = owned_edges + ghost_edges;
        // Populate the three new, internal edges for each face
        // local_face_lids(i) references face local IDs
        Kokkos::parallel_for("new internal edges", Kokkos::RangePolicy<execution_space>(0, face_refinements),
            KOKKOS_LAMBDA(int i) {
            
            int face_id = local_face_lids(i);
            int e_lid;
            // Set general values for each of the three new edges
            int offset = Kokkos::atomic_fetch_add(&edge_counter(), 3);
            // if (rank == 1) printf("R%d face %d (edges %d, %d, %d), adding edges %d, %d, %d (offset %d)\n", rank,
            //     face_id+vef_gid_start(rank, 2),
            //     f_eid(face_id, 0), f_eid(face_id, 1), f_eid(face_id, 2),
            //     e_new_lid_start+offset, e_new_lid_start+offset+1, e_new_lid_start+offset+2, offset);
            for (int j = offset; j < offset+3; j++)
            {
                e_lid = e_new_lid_start + j;

                // Global IDs = global ID start + local ID
                e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;

                // if (rank == 2) printf("R%d face %d (edges %d, %d, %d), adding edge %d (offset %d)\n", rank,
                // face_id+vef_gid_start(rank, 2),
                // f_eid(face_id, 0), f_eid(face_id, 1), f_eid(face_id, 2),
                // e_gid(e_lid), offset);

                // Set parent and child edges
                e_pid(e_lid) = -1; e_cid(e_lid, 0) = -1; e_cid(e_lid, 1) = -1;

                // Owning rank
                e_rank(e_lid) = rank;

                 /**
                 * For internal edges, get their layer from the layer of the
                 * face they are splitting
                 */
                e_layer(e_lid) = f_layer(face_id) + 1;

                /**
                 * Get the edges of the face we are refining. If any of them are outside our
                 * global ID space, check the ghosted edges to get the vertex IDs
                 * 
                 * elid[3] = local IDs of these face edges
                 */
                int elid[3];
                for (int k = 0; k < 3; k++)
                {
                    int egid = f_eid(face_id, k);
                    int lid = egid - vef_gid_start(rank, 1);
                    elid[k] = Utils::get_lid(e_gid, egid, 0, num_edges);
                    assert(elid[k] != -1);
                    // if (e_gid(e_lid) == 699) printf("elid %d: %d\n", k, elid[k]);                 
                }

                // if (e_gid(e_lid) == 699) printf("R%d G(%d, %d, %d) -> L(%d, %d, %d)\n", rank,
                //     f_eid(face_id, 0), f_eid(face_id, 1), f_eid(face_id, 2),
                //     elid[0], elid[1], elid[2]);
                
                /**
                 * Set vertices (nv0, nv1, nv2):
                 *  e_lid0: (new vertex 0, new vertex 1)
                 *  e_lid1: (new vertex 1, new vertex 2)
                 *  e_lid2: (new vertex 0, new vertex 2)
                 */
                int nv0, nv1, nv2;
                nv0 = e_vid(elid[0], 2); nv1 = e_vid(elid[1], 2); nv2 = e_vid(elid[2], 2); 
                //int elid0, elid1, elid2, nv0, nv1, nv2;
                // elid0 = f_eid(face_id, 0)-vef_gid_start(rank, 1); elid1 = f_eid(face_id, 1)-vef_gid_start(rank, 1); elid2 = f_eid(face_id, 2)-vef_gid_start(rank, 1);
                // nv0 = e_vid(elid0, 2); nv1 = e_vid(elid1, 2); nv2 = e_vid(elid2, 2); 
                // printf("R%d: nv: %d, %d, %d\n", rank, nv0, nv1, nv2);
                if (j == offset) {e_vid(e_lid, 0) = nv0; e_vid(e_lid, 1) = nv1;}
                else if (j == offset+1) {e_vid(e_lid, 0) = nv1; e_vid(e_lid, 1) = nv2;}
                else if (j == offset+2) {e_vid(e_lid, 0) =nv0; e_vid(e_lid, 1) = nv2;}

                // Middle vertex not set until new edges are further refined
                e_vid(e_lid, 2) = -1;
                // printf("R%d: ne%d: (%d, %d, %d)\n", rank, e_lid+vef_gid_start(rank, 1), e_vid(e_lid, 0), e_vid(e_lid, 1), e_vid(e_lid, 2));
            
                // if (e_gid(e_lid) == 699)
                // {
                //     printf("e%d: new verts: %d, %d, %d, j=%d, offset=%d\n", e_gid(e_lid), nv0, nv1, nv2,
                //         j, offset);
                // }
            }
        });

        // if (rank == 2) printEdges(3, 0);
        
        // Create the new faces
        // printFaces(1, 31);
        // printf("New lid start: %d\n", f_new_lid_start);
        Kokkos::deep_copy(face_counter, 0);
        Kokkos::parallel_for("new internal faces", Kokkos::RangePolicy<execution_space>(0, face_refinements),
            KOKKOS_LAMBDA(int i) {
            
            int parent_face_lid = local_face_lids(i);
            int parent_face_gid = parent_face_lid + vef_gid_start(rank, 2);
            int layer = f_layer(parent_face_lid) + 1;
            int offset = Kokkos::atomic_fetch_add(&face_counter(), 4);
            int new_face_lid, new_face_gid;
            int num_edges = owned_edges + ghost_edges; // Edge AoSoA size

            // Start with values similar to all new faces; can be looped
            for (int j = offset; j < offset+4; j++)
            {
                new_face_lid = f_new_lid_start + j;
                new_face_gid = new_face_lid + vef_gid_start(rank, 2);

                // Set new faces as children of parent face
                f_cid(parent_face_lid, j-offset) = new_face_gid;

                // Owning rank
                f_owner(new_face_lid) = rank;

                // Global ID
                f_gid(new_face_lid) = new_face_gid;

                // Layer
                f_layer(new_face_lid) = layer;

                // Children
                for (int k = 0; k < 4; k++) f_cid(new_face_lid, k) = -1;

                // Parent
                f_pid(new_face_lid) = parent_face_gid;

                // printf("R%d: Adding fgid %d from parent face %d\n", rank, new_face_gid, parent_face_gid);
            }

            // Get the vertex GIDs of the three new vertices created from the parent face edges
            int parent_eg0, parent_eg1, parent_eg2, parent_el0, parent_el1, parent_el2;
            int vg_mid0, vg_mid1, vg_mid2;
            parent_eg0 = f_eid(parent_face_lid, 0);
            parent_eg1 = f_eid(parent_face_lid, 1);
            parent_eg2 = f_eid(parent_face_lid, 2);
            parent_el0 = Utils::get_lid(e_gid, parent_eg0, 0, num_edges);
            assert(parent_el0 != -1);
            parent_el1 = Utils::get_lid(e_gid, parent_eg1, 0, num_edges);
            assert(parent_el1 != -1);
            parent_el2 = Utils::get_lid(e_gid, parent_eg2, 0, num_edges);
            assert(parent_el2 != -1);
            vg_mid0 = e_vid(parent_el0, 2); vg_mid1 = e_vid(parent_el1, 2); vg_mid2 = e_vid(parent_el2, 2);
            // printf("R%d: vgmid 0/1/2: %d, %d, %d\n", rank, vg_mid0, vg_mid1, vg_mid2);
            // printf("R%d: parent eg 0/1/2: %d, %d, %d\n", rank, parent_eg0, parent_eg1, parent_eg2);
            
            // Get the local and global IDs of the new edges that connect the new vertices
            int new_eg0, new_eg1, new_eg2, new_el0, new_el1, new_el2;
            // printf("R%d: vgmid0: %d, vg_mid1: %d\n", rank, vg_mid0, vg_mid1);
            new_el0 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid0, vg_mid1);
            assert(new_el0 != -1);
            new_el1 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid1, vg_mid2);
            assert(new_el1 != -1);
            new_el2 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid0, vg_mid2);
            assert(new_el2 != -1);
            new_eg0 = e_gid(new_el0); new_eg1 = e_gid(new_el1); new_eg2 = e_gid(new_el2);
            // printf("R%d: new eg 0/1/2: %d, %d, %d\n", rank, new_eg0, new_eg1, new_eg2);

            /******************************************************
             * Face 0: The face created from all new vertices and
             *  new edges without parents
             *****************************************************/
            new_face_lid = f_new_lid_start + offset;
            f_vid(new_face_lid, 0) = vg_mid0; f_vid(new_face_lid, 1) = vg_mid1; f_vid(new_face_lid, 2) = vg_mid2;
            f_eid(new_face_lid, 0) = new_eg0; f_eid(new_face_lid, 1) = new_eg1; f_eid(new_face_lid, 2) = new_eg2; 

            // The vertex GID on faces 1, 2, and 3 that is not a middle vertex
            int vg_outer;

            // The edge local IDs on faces 1, 2, and 3 that 
            // do not connect two middle vertices
            int el_outer0, el_outer1;

            /******************************************************
             * Face 1: The face that shares inner edge 0
             *  (verts vg_mid0 and vg_mid1) of Face 0
             *****************************************************/
            new_face_lid = f_new_lid_start + offset + 1;
            vg_outer = Utils::vertex_from_parent_edges(e_vid, parent_el0, parent_el1, parent_el2,
                                                       vg_mid0, vg_mid1);
            // printf("R%d: mid0: %d, mid1: %d, outer: %d\n", rank, vg_mid0, vg_mid1, vg_outer);
            // Get the edge LIDs that connect to vg_outer
            el_outer0 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid0, vg_outer);
            // printf("R%d: vgmid0: %d, vg_outer: %d\n", rank, vg_mid0, vg_outer);
            assert(el_outer0 != -1);
            el_outer1 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid1, vg_outer);
            assert(el_outer1 != -1);
            // Assign values
            f_vid(new_face_lid, 0) = vg_mid0; f_vid(new_face_lid, 1) = vg_mid1; f_vid(new_face_lid, 2) = vg_outer;
            f_eid(new_face_lid, 0) = new_eg0; f_eid(new_face_lid, 1) = e_gid(el_outer0); f_eid(new_face_lid, 2) = e_gid(el_outer1);

            /******************************************************
             * Face 2: The face that shares inner edge 1
             *  (verts vg_mid1 and vg_mid2) of Face 0
             *****************************************************/
            new_face_lid = f_new_lid_start + offset + 2;
            vg_outer = Utils::vertex_from_parent_edges(e_vid, parent_el0, parent_el1, parent_el2,
                                                       vg_mid1, vg_mid2);
            // Get the edge LIDs that connect to vg_outer
            el_outer0 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid1, vg_outer);
            assert(el_outer0 != -1);
            el_outer1 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid2, vg_outer);
            assert(el_outer1 != -1);
            // Assign values
            f_vid(new_face_lid, 0) = vg_mid1; f_vid(new_face_lid, 1) = vg_mid2; f_vid(new_face_lid, 2) = vg_outer;
            f_eid(new_face_lid, 0) = new_eg1; f_eid(new_face_lid, 1) = e_gid(el_outer0); f_eid(new_face_lid, 2) = e_gid(el_outer1);
           
            /******************************************************
             * Face 3: The face that shares inner edge 2
             *  (verts vg_mid0 and vg_mid2) of Face 0
             *****************************************************/
            new_face_lid = f_new_lid_start + offset + 3;
            vg_outer = Utils::vertex_from_parent_edges(e_vid, parent_el0, parent_el1, parent_el2,
                                                       vg_mid0, vg_mid2);
            // Get the edge LIDs that connect to vg_outer
            el_outer0 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid0, vg_outer);
            assert(el_outer0 != -1);
            el_outer1 = Utils::find_edge(e_vid, e_new_lid_start, num_edges, vg_mid2, vg_outer);
            assert(el_outer1 != -1);
            // Assign values
            f_vid(new_face_lid, 0) = vg_mid0; f_vid(new_face_lid, 1) = vg_mid2; f_vid(new_face_lid, 2) = vg_outer;
            f_eid(new_face_lid, 0) = new_eg2; f_eid(new_face_lid, 1) = e_gid(el_outer0); f_eid(new_face_lid, 2) = e_gid(el_outer1);

        });

        // if (rank == 0)
        // {
        //     printf("***************AFTER***************\n");
        //     printEdges(3, 0);
        //     // find verts 30, 103
        //     // verts 25, -1; -1, 26
        // }
        // printf("R%d: end of refine: num faces: %d\n", rank, _faces.size());

    }
  
    void _updateGlobalIDs()
    {
        // Update mesh version
        _version++;
        
        // Temporarily save old starting positions
        Kokkos::View<int*[3], memory_space> old_vef_start("old_vef_start", _comm_size);
        Kokkos::deep_copy(old_vef_start, _vef_gid_start);

        // Copy _vef_gid_start to host to modify
        auto vef_gid_start_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _vef_gid_start);

        int vef[3] = {_owned_vertices, _owned_edges, _owned_faces};
        MPI_Allgather(vef, 3, MPI_INT, _vef_gid_start.data(), 3, MPI_INT, _comm);

        // Find where each process starts its global IDs
        for (int i = 1; i < _comm_size; ++i) {
            vef_gid_start_h(i, 0) += vef_gid_start_h(i - 1, 0);
            vef_gid_start_h(i, 1) += vef_gid_start_h(i - 1, 1);
            vef_gid_start_h(i, 2) += vef_gid_start_h(i - 1, 2);
        }
        for (int i = _comm_size - 1; i > 0; --i) {
            vef_gid_start_h(i, 0) = vef_gid_start_h(i - 1, 0);
            vef_gid_start_h(i, 1) = vef_gid_start_h(i - 1, 1);
            vef_gid_start_h(i, 2) = vef_gid_start_h(i - 1, 2);
        }
        vef_gid_start_h(0, 0) = 0;
        vef_gid_start_h(0, 1) = 0;
        vef_gid_start_h(0, 2) = 0;

        if (old_vef_start(0, 0) == -1)
        {
            // Don't update global IDs when the mesh is initially formed
            return;
        }

        // Copy updated vef view to device
        auto hv_tmp1 = Kokkos::create_mirror_view(_vef_gid_start);
        Kokkos::deep_copy(hv_tmp1, vef_gid_start_h);
        Kokkos::deep_copy(_vef_gid_start, hv_tmp1);

        auto vef_gid_start = _vef_gid_start;

        // Update vertices
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        Kokkos::parallel_for("update vertex GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_vertices),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            Utils::updateGlobalID(Vertex(), &v_gid(i), vef_gid_start, old_vef_start);
            
        });

        /**
         * Edges may connect to vertices owned by another process,
         * or may be ghosted. IDs must be adjusted based off the process
         * that owns the edge.
         */
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_vid = Cabana::slice<E_VIDS>(_edges);
        auto e_cid = Cabana::slice<E_CIDS>(_edges);
        auto e_pid = Cabana::slice<E_PID>(_edges);
        Kokkos::parallel_for("update edge GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_edges),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            Utils::updateGlobalID(Edge(), &e_gid(i), vef_gid_start, old_vef_start);

            // Vertex association GIDs
            for (int j = 0; j < 3; j++)
            {
                Utils::updateGlobalID(Vertex(), &e_vid(i, j), vef_gid_start, old_vef_start);
            }
            
            // Child edge GIDs
            for (int j = 0; j < 2; j++)
            {
                Utils::updateGlobalID(Edge(), &e_cid(i, j), vef_gid_start, old_vef_start);
            }

            // Parent edge GID
            Utils::updateGlobalID(Edge(), &e_pid(i), vef_gid_start, old_vef_start);
        });

        // Update Faces
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_cid = Cabana::slice<F_CID>(_faces);
        auto f_eid = Cabana::slice<F_EIDS>(_faces);
        auto f_vid = Cabana::slice<F_VIDS>(_faces);
        auto f_pid = Cabana::slice<F_PID>(_faces);
        Kokkos::parallel_for("update face GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_faces),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            Utils::updateGlobalID(Face(), &f_gid(i), vef_gid_start, old_vef_start);

            // Child face association GIDs
            for (int j = 0; j < 4; j++)
            {   
                Utils::updateGlobalID(Face(), &f_cid(i, j), vef_gid_start, old_vef_start);
            }

            // Edge association GIDs
            for (int j = 0; j < 3; j++)
            {
                Utils::updateGlobalID(Edge(), &f_eid(i, j), vef_gid_start, old_vef_start);
            }

            // Vertex association GIDs
            for (int j = 0; j < 3; j++)
            {
                Utils::updateGlobalID(Vertex(), &f_vid(i, j), vef_gid_start, old_vef_start);
            }

            // Parent face
            Utils::updateGlobalID(Face(), &f_pid(i), vef_gid_start, old_vef_start);
        });
    }

    /**
     * Find the vertices, edges, and/or faces that must be exchanged for the halo 
     * 
     * Step 1:
     *  Use distributor to request missing vertex data for owned faces which
     *      contain an unowned vertex. At the same time, build the halo data
     *      Approach:
     *          - Iterate over boundary faces:
     *              1. Add to vert_distributor_export: Any vertex GIDs we do not own
     *              2. Add to edge_distributor_export: Any edge GIDs we do not own
     *              3. Add to halo_export_ids: All vef data on faces that contain an unowned vertex
     *              4. Add to halo_export_ranks: The owner rank of the unowned vertex
     * 
     * Step 2:
     *  - Distribute the vert/edge_distributor_export data into vert/edge_distributor_import
     *  - Add imported vertex and edge GIDs to halo_export_ids and halo_export_ranks.
     * 
     */
    void _gather_depth_one()
    {
        const int level = _halo_level, rank = _rank, tree_depth = _max_tree_level;

        // Define the hash map type
        using PairType = std::pair<int, int>;
        using KeyType = uint64_t;  // Hashable key
        using MapType = Kokkos::UnorderedMap<KeyType, int, memory_space>;

        // Hash function to combine two integers into a single key
        auto hashFunction = KOKKOS_LAMBDA(int first, int second) -> KeyType {
            return (static_cast<KeyType>(first) << 32) | (static_cast<KeyType>(second) & 0xFFFFFFFF);
        };

        auto boundary_faces = _boundary_faces;
        size_t num_boundary_faces = boundary_faces.extent(0);
        // size_t num_boundary_edges = _mesh->boundary_edges().extent(0);

        auto vef_gid_start = _vef_gid_start;

        // Vertex slices
        auto v_gid = Cabana::slice<V_GID>(_vertices);

        // Edge slices
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_vid = Cabana::slice<E_VIDS>(_edges);
        auto e_rank = Cabana::slice<E_OWNER>(_edges);
        auto e_cid = Cabana::slice<E_CIDS>(_edges);
        auto e_pid = Cabana::slice<E_PID>(_edges);
        auto e_layer = Cabana::slice<E_LAYER>(_edges);

        // Face slices
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_eid = Cabana::slice<F_EIDS>(_faces);
        auto f_vid = Cabana::slice<F_VIDS>(_faces);
        auto f_cid = Cabana::slice<F_CID>(_faces);
        auto f_pid = Cabana::slice<F_PID>(_faces);
        auto f_rank = Cabana::slice<F_OWNER>(_faces);
        auto f_layer = Cabana::slice<F_LAYER>(_faces);

        int vertex_count = _owned_vertices;
        int edge_count = _owned_edges;
        int face_count = _owned_faces;

        // Distributor data
        // (global ID, to_rank, from_rank) tuples
        using distributor_aosoa = Cabana::AoSoA<Cabana::MemberTypes<int, int, int>, memory_space, 4>;
        size_t vert_distributor_size = num_boundary_faces*(tree_depth+1)*4;
        size_t edge_distributor_size = num_boundary_faces*(tree_depth+1)*4;
        // printf("R%d: vert/edge dist sizes: %d, %d\n", rank, vert_distributor_size, edge_distributor_size);
        distributor_aosoa vert_distributor_export("vert_distributor_export", vert_distributor_size);
        distributor_aosoa edge_distributor_export("edge_distributor_export", edge_distributor_size);

        // Halo data
        // (global ID, to_rank) tuples
        size_t vhalo_size = num_boundary_faces*(tree_depth+1)*4; // Slight buffer for faces that must be sent to >1 processes
        size_t ehalo_size = num_boundary_faces*(tree_depth+1)*4; // and for refinements along boundaries
        size_t fhalo_size = num_boundary_faces*(tree_depth+1)*4;
        halo_aosoa vert_halo_export("vert_halo_export", vhalo_size);
        halo_aosoa edge_halo_export("edge_halo_export", ehalo_size);
        halo_aosoa face_halo_export("face_halo_export", fhalo_size);

        // Hash tables - used to keep duplicate entries from the distributor and halo data structures
        MapType vert_distributor_map(vert_distributor_size);
        MapType edge_distributor_map(edge_distributor_size);
        MapType vert_halo_map(vhalo_size);
        MapType edge_halo_map(ehalo_size);
        /**
         * The face_halo_map is different:
         * Used track which faces (and the vertex and edge data they hold)
         * must be sent to which remote processes
         */
        MapType face_halo_map(fhalo_size);

        // Counters
        using int_d = Kokkos::View<size_t, memory_space>;
        int_d vd_idx("vd_idx"); Kokkos::deep_copy(vd_idx, 0); // Vert distributor
        int_d ed_idx("ed_idx"); Kokkos::deep_copy(ed_idx, 0); // Edge distributor
        int_d vh_idx("vh_idx"); Kokkos::deep_copy(vh_idx, 0); // Vert halo
        int_d eh_idx("eh_idx"); Kokkos::deep_copy(eh_idx, 0); // Edge halo
        int_d fh_idx("fh_idx"); Kokkos::deep_copy(fh_idx, 0); // Face halo

        // Slices
        auto vert_distributor_export_gids = Cabana::slice<0>(vert_distributor_export);
        auto vert_distributor_export_to_ranks = Cabana::slice<1>(vert_distributor_export);
        auto vert_distributor_export_from_ranks = Cabana::slice<2>(vert_distributor_export);
        auto edge_distributor_export_gids = Cabana::slice<0>(edge_distributor_export);
        auto edge_distributor_export_to_ranks = Cabana::slice<1>(edge_distributor_export);
        auto edge_distributor_export_from_ranks = Cabana::slice<2>(edge_distributor_export);
        auto vert_halo_export_lids = Cabana::slice<0>(vert_halo_export);
        auto vert_halo_export_ranks = Cabana::slice<1>(vert_halo_export);
        auto edge_halo_export_lids = Cabana::slice<0>(edge_halo_export);
        auto edge_halo_export_ranks = Cabana::slice<1>(edge_halo_export);
        auto face_halo_export_lids = Cabana::slice<0>(face_halo_export);
        auto face_halo_export_ranks = Cabana::slice<1>(face_halo_export);

        /**
         * Iterate over boundary faces. For each boundary face, check if any vertices
         * on it are owned by another process.
         * If so, hash the (fgid, vert_owner) tuple into face_halo_map 
         * AND hash all children faces of this face into face_halo_map,
         *  to the same vert_owner.
         * ALSO add all vertices and edges of these faces to the halos and distributors
         */
        Kokkos::parallel_for("boundary face iteration", Kokkos::RangePolicy<execution_space>(0, num_boundary_faces),
            KOKKOS_LAMBDA(int face_idx) {
            
            int fgid_parent = boundary_faces(face_idx);
            int flid_parent = fgid_parent - vef_gid_start(rank, 2);
            int face_level = f_layer(flid_parent);
            // if (rank == 2) printf("R%d: boundary face gid: %d, level: %d\n", rank, fgid_parent, face_level);
            if (face_level != level) return; // Only consider elements at our level and their children

            // Consider each vertex of this face
            for (int i = 0; i < 3; i++)
            {
                const int vgid_parent = f_vid(flid_parent, i);
                int vert_owner = Utils::owner_rank(Vertex(), vgid_parent, vef_gid_start);

                // If unowned vertex, add this face and all child faces to
                // send to vert_owner
                if (vert_owner != rank)
                {
                    // Add this vert to the distributor
                    // but first check that it is not already present
                    auto hash_key = hashFunction(vgid_parent, vert_owner);
                    auto result = vert_distributor_map.insert(hash_key, 1);
                    if (result.success()) {
                        // If insertion succeeds; tuple not present; add to AoSoA
                        int dvdx = Kokkos::atomic_fetch_add(&vd_idx(), 1);
                        assert(dvdx < vert_distributor_size);
                        vert_distributor_export_gids(dvdx) = vgid_parent;
                        vert_distributor_export_from_ranks(dvdx) = rank;
                        vert_distributor_export_to_ranks(dvdx) = vert_owner;
                        // if (fgid_parent == 376) printf("R%d: adding (R%d, vgid %d) to distributor\n", rank, vert_owner, vgid_parent);
                    }

                    // Queue for children (local per thread), that will all go to the remote rank
                    const int capacity = 86;
                    int queue[capacity]; // Adjust size as needed
                    int front = 0, back = 0;

                    // Initialize queue with the parent face
                    queue[back] = fgid_parent;
                    back = (back + 1) % capacity;

                    // Traverse children iteratively
                    while (front != back)
                    {
                        // Dequeue
                        int fgid = queue[front];
                        front = (front + 1) % capacity;
                        
                        int flid = fgid - vef_gid_start(rank, 2);
                        assert(flid > -1);

                        /************************
                         * Process the face
                         ***********************/
                        
                        // Add this face to the halo to send to the given vertex owner
                        hash_key = hashFunction(fgid, vert_owner);
                        result = face_halo_map.insert(hash_key, vert_owner);
                        if (result.success()) {
                            // If insertion succeeds; tuple not present; add to AoSoA
                            int fdx = Kokkos::atomic_fetch_add(&fh_idx(), 1);
                            assert(fdx < fhalo_size);
                            face_halo_export_lids(fdx) = flid;
                            face_halo_export_ranks(fdx) = vert_owner;
                            // if (fgid == 376) printf("R%d: adding fgid %d to halo to R%d\n", rank, fgid, vert_owner);
                        }
                                                
                        // Add owned edges
                        for (int j = 0; j < 3; j++)
                        {
                            int egid = f_eid(flid, j);
                            int edge_owner = Utils::owner_rank(Edge(), egid, vef_gid_start);
                            // if (fgid == 38) printf("R%d: edge %d from face %d, owner R%d\n", rank, egid, fgid, edge_owner);
                            if (edge_owner != rank)
                            {
                                // We reference this edge but do not own it; add to AoSoA
                                hash_key = hashFunction(egid, edge_owner);
                                result = edge_distributor_map.insert(hash_key, 1);
                                if (result.success()) {
                                    // If insertion succeeds; tuple not present; add to AoSoA
                                    int dedx = Kokkos::atomic_fetch_add(&ed_idx(), 1);
                                    assert(dedx < edge_distributor_size);
                                    edge_distributor_export_gids(dedx) = egid;
                                    edge_distributor_export_from_ranks(dedx) = rank;
                                    edge_distributor_export_to_ranks(dedx) = edge_owner;
                                }
                                // fgid 102, edge 98 
                                // if (rank == 3) printf("R%d: from face %d: edge_dist(%d): %d to rank %d\n", rank, fgid, dedx, egid, edge_owner);
                                continue; 
                            }

                            // Otherwise we own the edge and need to add it to the halo
                            hash_key = hashFunction(egid, edge_owner);
                            result = edge_halo_map.insert(hash_key, 1);
                            if (result.success()) {
                                // If insertion succeeds; tuple not present; add to AoSoA
                                int edx = Kokkos::atomic_fetch_add(&eh_idx(), 1);
                                assert(edx < ehalo_size);
                                int elid = egid - vef_gid_start(rank, 1);
                                edge_halo_export_lids(edx) = elid;
                                edge_halo_export_ranks(edx) = vert_owner;
                            }
                            
                            // if (rank == 2) printf("R%d: adding edge %d from face %d to rank %d\n", rank, egid, fgid, vert_owner);
                        }

                        // Add owned verts
                        for (int k = 0; k < 3; k++)
                        {
                            int vgid1 = f_vid(flid, k);
                            int vowner = Utils::owner_rank(Vertex(), vgid1, vef_gid_start);
                            if (vowner != rank) continue; // Can't export verts we don't own
                            hash_key = hashFunction(vgid1, vert_owner);
                            result = vert_halo_map.insert(hash_key, 1);
                            if (result.success()) {
                                // If insertion succeeds; tuple not present; add to AoSoA
                                int vdx = Kokkos::atomic_fetch_add(&vh_idx(), 1);
                                assert(vdx < vhalo_size);
                                int vlid1 = vgid1 - vef_gid_start(rank, 0);
                                vert_halo_export_lids(vdx) = vlid1;
                                vert_halo_export_ranks(vdx) = vert_owner;
                            }
                            
                            // if (rank == 1) printf("R%d: sending vgid %d to %d\n", rank, vgid1, vert_owner);
                        }

                        // Enqueue child faces to be send to vert_owner rank
                        for (int j = 0; j < 4; j++)
                        {
                            int fcgid = f_cid(flid, j);
                            // Enqueue child if it exists
                            if (fcgid != -1) { // -1 indicates no child
                                queue[back] = fcgid;
                                back = (back + 1) % capacity;

                                // Handle queue overflow (optional, if queue size is too small)
                                assert(back != front);
                                // if (fgid_parent == 326) printf("R%d: from fgid_parent %d, adding child %d\n",
                                //     rank, fgid_parent, fcgid);
                            }
                        }
                    }
                }
            }
        });
                
        // Resize distributor data to correct sizes
        Kokkos::deep_copy(vert_distributor_size, vd_idx);
        Kokkos::deep_copy(edge_distributor_size, ed_idx);
        
        // printf("R%d: sizes: %d, %d; actual: %d, %d\n", rank, vert_distributor_size, edge_distributor_size,
        //     vert_distributor_export.size(), edge_distributor_export.size());

        vert_distributor_export.resize(vert_distributor_size);
        edge_distributor_export.resize(edge_distributor_size);

        // Set duplicates to -1 so they are ignored
        // _set_duplicates_neg1(vert_distributor_export);
        // _set_duplicates_neg1(edge_distributor_export);

        // Update distributor slices after resizing
        vert_distributor_export_gids = Cabana::slice<0>(vert_distributor_export);
        vert_distributor_export_to_ranks = Cabana::slice<1>(vert_distributor_export);
        vert_distributor_export_from_ranks = Cabana::slice<2>(vert_distributor_export);
        edge_distributor_export_gids = Cabana::slice<0>(edge_distributor_export);
        edge_distributor_export_to_ranks = Cabana::slice<1>(edge_distributor_export);
        edge_distributor_export_from_ranks = Cabana::slice<2>(edge_distributor_export);

        // Set duplicate edge/face + send_to_rank pairs to be ignored (sent to rank -1)
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_distributor_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("before R%d: to: R%d, vert gid: %d\n", rank,
        //         vert_distributor_export_to_ranks(i), vert_distributor_export_gids(i));

        // });
       
        
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_distributor_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 0) printf("vert distributor R%d: to: R%d, vert gid: %d\n", rank,
        //         vert_distributor_export_to_ranks(i), vert_distributor_export_gids(i));

        // });

        /**
         * Distribute vertex data, then add imported vertex data to the vertex halo
         */
        auto vert_distributor = Cabana::Distributor<memory_space>(_comm, vert_distributor_export_to_ranks);
        const int vert_distributor_total_num_import = vert_distributor.totalNumImport();
        distributor_aosoa vert_distributor_import("vert_distributor_import", vert_distributor_total_num_import); 
        Cabana::migrate(vert_distributor, vert_distributor_export, vert_distributor_import);
        
        auto edge_distributor = Cabana::Distributor<memory_space>(_comm, edge_distributor_export_to_ranks);
        const int edge_distributor_total_num_import = edge_distributor.totalNumImport();
        distributor_aosoa edge_distributor_import("edge_distributor_import", edge_distributor_total_num_import);
        Cabana::migrate(edge_distributor, edge_distributor_export, edge_distributor_import);

        // Distributor import slices
        auto vert_distributor_import_gids = Cabana::slice<0>(vert_distributor_import);
        auto vert_distributor_import_to_ranks = Cabana::slice<1>(vert_distributor_import);
        auto vert_distributor_import_from_ranks = Cabana::slice<2>(vert_distributor_import);
        auto edge_distributor_import_gids = Cabana::slice<0>(edge_distributor_import);
        auto edge_distributor_import_to_ranks = Cabana::slice<1>(edge_distributor_import);
        auto edge_distributor_import_from_ranks = Cabana::slice<2>(edge_distributor_import);

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_distributor_import.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("R%d: R%d asking for vert gid %d\n", rank,
        //         vert_distributor_import_from_ranks(i), vert_distributor_import_gids(i));

        // });
        Kokkos::parallel_for("add imported distributor vertex data",
            Kokkos::RangePolicy<execution_space>(0, vert_distributor_total_num_import),
            KOKKOS_LAMBDA(int i) {
            
            int vgid = vert_distributor_import_gids(i);
            int vlid = vgid - vef_gid_start(rank, 0);
            int from_rank = vert_distributor_import_from_ranks(i);

            // Add to vert halo
            KeyType hash_key = hashFunction(vgid, from_rank);
            auto result = vert_halo_map.insert(hash_key, 1);
            if (result.success()) {
                // If insertion succeeds; tuple not present; add to AoSoA
                int vdx = Kokkos::atomic_fetch_add(&vh_idx(), 1);
                assert(vdx < vhalo_size);
                vert_halo_export_lids(vdx) = vlid;
                vert_halo_export_ranks(vdx) = from_rank;
            }
            // if (vgid == 16 && from_rank == 3) printf("R%d: adding vgid %d to R%d to halo at vdx %d\n", rank, vgid, from_rank, vdx);
            
        });

        // Add all edges and their vertices, and their children to the halo
        Kokkos::parallel_for("add imported distributor edge data",
            Kokkos::RangePolicy<execution_space>(0, edge_distributor_total_num_import),
            KOKKOS_LAMBDA(int i) {
            
            int egid_parent = edge_distributor_import_gids(i);
            int from_rank = edge_distributor_import_from_ranks(i);

            // Queue for children (local per thread), that will all go to the remote rank
            const int capacity = 32;
            int queue[capacity]; // Adjust size as needed
            int front = 0, back = 0;

            // Initialize queue with the parent face
            queue[back] = egid_parent;
            back = (back + 1) % capacity;

            // Traverse children iteratively
            while (front != back)
            {
                // Dequeue
                int egid = queue[front];
                front = (front + 1) % capacity;
                
                int elid = egid - vef_gid_start(rank, 1);
                assert(elid > -1);
                
                // Add this edge to the halo to send to from_rank
                KeyType hash_key = hashFunction(egid, from_rank);
                auto result = edge_halo_map.insert(hash_key, 1);
                if (result.success()) {
                    // If insertion succeeds; tuple not present; add to AoSoA
                    int edx = Kokkos::atomic_fetch_add(&eh_idx(), 1);
                    assert(edx < ehalo_size);
                    edge_halo_export_lids(edx) = elid;
                    edge_halo_export_ranks(edx) = from_rank;
                }

                // Add vertices on this edge to the halo to send to from_rank
                for (int i = 0; i < 3; i++)
                {
                    int vgid = e_vid(elid, i);
                    int vowner = Utils::owner_rank(Vertex(), vgid, vef_gid_start);
                    if (vowner != rank) continue; // Can't export verts we don't own
                    hash_key = hashFunction(vgid, from_rank);
                    result = vert_halo_map.insert(hash_key, 1);
                    if (result.success()) {
                        // If insertion succeeds; tuple not present; add to AoSoA
                        int vdx = Kokkos::atomic_fetch_add(&vh_idx(), 1);
                        assert(vdx < vhalo_size);
                        int vlid = vgid - vef_gid_start(rank, 0);
                        vert_halo_export_lids(vdx) = vlid;
                        vert_halo_export_ranks(vdx) = from_rank;
                    }
                }

                // Enqueue child edges to be sent to vert_owner rank
                for (int j = 0; j < 1; j++)
                {
                    int ecgid = e_cid(elid, j);
                    // Enqueue child if it exists
                    if (ecgid != -1) { // -1 indicates no child
                        queue[back] = ecgid;
                        back = (back + 1) % capacity;

                        // Handle queue overflow (optional, if queue size is too small)
                        assert(back != front);
                    }
                }
            }

            // if (edx > 85) printf("R%d: edx: %d\n", rank, edx);
            // if (rank == 2) printf("R%d: adding egid %d to halo to rank %d: el/gid: %d, %d\n", rank, egid, from_rank, elid, e_gid(elid));
            // if (vgid == 16 && from_rank == 3) printf("R%d: adding vgid %d to R%d to halo at vdx %d\n", rank, vgid, from_rank, vdx);
            
        });

        // Finalize halo sizes
        Kokkos::deep_copy(vhalo_size, vh_idx);
        Kokkos::deep_copy(ehalo_size, eh_idx);
        Kokkos::deep_copy(fhalo_size, fh_idx);
        vert_halo_export.resize(vhalo_size);
        edge_halo_export.resize(ehalo_size);
        face_halo_export.resize(fhalo_size);

        // Add this halo data to the halo vectors
        _vert_halo_export.push_back(std::make_shared<halo_aosoa>(vert_halo_export));
        _edge_halo_export.push_back(std::make_shared<halo_aosoa>(edge_halo_export));
        _face_halo_export.push_back(std::make_shared<halo_aosoa>(face_halo_export));

        // Update slices
        vert_halo_export_lids = Cabana::slice<0>(vert_halo_export);
        vert_halo_export_ranks = Cabana::slice<1>(vert_halo_export);
        edge_halo_export_lids = Cabana::slice<0>(edge_halo_export);
        edge_halo_export_ranks = Cabana::slice<1>(edge_halo_export);
        face_halo_export_lids = Cabana::slice<0>(face_halo_export);
        face_halo_export_ranks = Cabana::slice<1>(face_halo_export);

        // printf("R%d: vef halo sizes: %d, %d, %d\n", rank, vhalo_size, ehalo_size, fhalo_size);

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {

            // if (rank == 1) printf("R%d: to: R%d, vert lid: %d\n", rank,
            //     vert_halo_export_ranks(i), vert_halo_export_lids(i));
            // int export_lid = vert_halo_export_lids(i);
            // if (export_lid >= vertex_count) printf("R%d: max %d verts, has vlid %d\n", rank, vertex_count, export_lid);

        // });
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, edge_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     printf("R%d: to: R%d, edge lid: %d\n", rank,
        //         edge_halo_export_ranks(i), edge_halo_export_lids(i));

        // });
        
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, face_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("R%d: to: R%d, face lid: %d\n", rank,
        //         face_halo_export_ranks(i), face_halo_export_lids(i));

        // });

        // _set_duplicates_neg1(vert_halo_export);
        // _set_duplicates_neg1(edge_halo_export);
        // _set_duplicates_neg1(face_halo_export);

        // Create halos
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     printf("R%d: to: R%d, vert lid: %d\n", rank,
        //         vert_halo_export_ranks(i), vert_halo_export_lids(i));
        //     // int export_lid = vert_halo_export_lids(i);
        //     // if (export_lid >= vertex_count) printf("R%d: max %d verts, has vlid %d\n", rank, vertex_count, export_lid);

        // });
        // printf("R%d: vert count: %d, export size: %d\n", rank, vertex_count, vert_halo_export.size());
        // halo_aosoa vert_halo_export_test("vert_halo_export", 10);
        // auto vert_halo_export_lids_test = Cabana::slice<0>(vert_halo_export_test);
        // auto vert_halo_export_ranks_test = Cabana::slice<1>(vert_halo_export_test);
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_halo_export_test.size()),
        //     KOKKOS_LAMBDA(int i) {

            
        //     vert_halo_export_ranks_test(i) = 1;
        //     vert_halo_export_lids_test(i) = 1;
        //     // int export_lid = vert_halo_export_lids(i);
        //     // if (export_lid >= vertex_count) printf("R%d: max %d verts, has vlid %d\n", rank, vertex_count, export_lid);

        // });

        Cabana::Halo<memory_space> vhalo( _comm, vertex_count, vert_halo_export_lids, vert_halo_export_ranks);
        Cabana::Halo<memory_space> ehalo( _comm, edge_count, edge_halo_export_lids, edge_halo_export_ranks);
        Cabana::Halo<memory_space> fhalo( _comm, face_count, face_halo_export_lids, face_halo_export_ranks);

        // Resize and gather
        _vertices.resize(vhalo.numLocal() + vhalo.numGhost());
        _edges.resize(ehalo.numLocal() + ehalo.numGhost());
        _faces.resize(fhalo.numLocal() + fhalo.numGhost());
        
        Cabana::gather(vhalo, _vertices);
        Cabana::gather(ehalo, _edges);
        Cabana::gather(fhalo, _faces);
        
        // Update ghost counts in mesh
        _owned_vertices = vhalo.numLocal();
        _owned_edges = ehalo.numLocal();
        _owned_faces = fhalo.numLocal();
        _ghost_vertices = vhalo.numGhost();
        _ghost_edges = ehalo.numGhost();
        _ghost_faces = fhalo.numGhost();
    }

    /**
     * Make a sparse matrix and square it
     */
    void make_sparse()
    {
        /**
         * A = hypre_ParCSRMatrixCreate(...)
         * row_starts and col_starts: these will be the same. Arrays of size 2 where
         *      0 = the global row/col I start
         *      1 = the global row/col I end, non-inclusive (Add number of row/col I own to index 0)
         * num_cols_offd: number of non-zero columns in the rows I own.
         *      Number of distinct vertices that I am connected to that I do not own.
         * num_nonzeros_diag: The number of non-zeroes in the part of the matrix that own (the fully local bit)
         *      Number of edges fully local between two vertices I own
         * num_nonzeroes_offd: total number of non-zero values in the rows but not the columns I own.
         *      Number of edges that go to any other node on any other process
         * returns a matrix of type hypre_ParCSRMatrix *A.
         * 
         * https://github.com/hypre-space/hypre/blob/master/src/parcsr_ls/par_laplace_27pt.c
         * 
         * Hypre uses structs to store all the parts of the matrix
         * Line 1667: you make your own CSR 
         * Use HYPRE_MEMORY_DEVICE
         * col_map_offd = size = num_cols_offd; this holds, for each non-zero offd column the global ID of the column, 
         *      ordered from lowest to highest.
         * 
         * Use HYPRE_MEMORY_HOST even if you want it on the GPU, 
         * at the end call hypre_ParCSRMatrixMigrate with HYPRE_MEMORY_DEVICE as second arg.
         * 
         * How to print out matrix:
         *      hypre_ParCRSMatrixPrint(). Might have to call upper-case version
         * 
         * TO get these functions, migh tneed to just go to the files Amanda sent and include those includes.
         * 
         * To multiply matrix:
         *      See Amanada notes
         * 
         * How to extract info from matrix:
         *      Call the set diags/values functions in reverse.
         */

        hypre_ParCSRMatrix *A;
        hypre_CSRMatrix *diag;
        hypre_CSRMatrix *offd;

        HYPRE_Int    *diag_i;
        HYPRE_Int    *diag_j;
        HYPRE_Real *diag_data;

        HYPRE_Int    *offd_i;
        HYPRE_Int    *offd_j = NULL;
        HYPRE_BigInt *big_offd_j = NULL;
        HYPRE_Real *offd_data = NULL;

        HYPRE_BigInt global_part[2];
        HYPRE_BigInt ix, iy, iz;
        HYPRE_Int cnt, o_cnt;
        HYPRE_Int local_num_rows;
        HYPRE_BigInt *col_map_offd;
        HYPRE_Int row_index;
        HYPRE_Int i;

        HYPRE_Int nx_local, ny_local, nz_local;
        HYPRE_Int num_cols_offd;
        HYPRE_BigInt grid_size;

        // diag_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);
        // offd_i = hypre_CTAlloc(HYPRE_Int,  local_num_rows + 1, HYPRE_MEMORY_HOST);


        // A = hypre_ParCSRMatrixCreate(comm, grid_size, grid_size,
        //                         global_part, global_part, num_cols_offd,
        //                         diag_i[local_num_rows],
        //                         offd_i[local_num_rows]);
        
    }

    /**
     * Turn a 2D array into an unstructured mesh of vertices, edges, and faces
     * by turning each (i, j) index into a vertex and creating edges between 
     * all eight neighbor vertices in the grid.
     * 
     * The values in the array are not used
     */
    template <class CabanaArray>
    void initializeFromArray( CabanaArray& array )
    {
        static_assert( Cabana::Grid::is_array<CabanaArray>::value, "NuMesh::Mesh::initializeFromArray: Cabana::Grid::Array required" );
        
        if (!Utils::isPerfectSquare(_comm_size))
        {
            std::cerr << "NuMesh::initializeFromArray only supports communicator sizes that are square numbers\n";
        }

        auto local_grid = array.layout()->localGrid();
        auto node_space = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                Cabana::Grid::Local() );
        
        /* Iterate over the 2D position array to populate AoSoAs in the unstructured mesh*/
        int istart = node_space.min(0), jstart = node_space.min(1);
        int iend = node_space.max(0), jend = node_space.max(1);

        // Create the AoSoA
        int ov = (iend - istart) * (jend - jstart);
        int oe = ov * 3;
        int of =  ov * 2;
        _vertices.resize(ov);
        _edges.resize(oe);
        _faces.resize(of);
        _owned_vertices = ov; _owned_edges = oe; _owned_faces = of;

        _updateGlobalIDs();

        auto vef_gid_start = _vef_gid_start;

        // We should convert the following loops to a Cabana::simd_parallel_for at some point to get better write behavior

        // Initialize the vertices, edges, and faces
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        auto v_owner = Cabana::slice<V_OWNER>(_vertices);

        auto e_vid = Cabana::slice<E_VIDS>(_edges); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_cids = Cabana::slice<E_CIDS>(_edges);
        auto e_pid = Cabana::slice<E_PID>(_edges);
        auto e_owner = Cabana::slice<E_OWNER>(_edges);
        auto e_layer = Cabana::slice<E_LAYER>(_edges);
        int rank = _rank;
        auto topology = Cabana::Grid::getTopology( *local_grid );
        auto device_topology = Utils::vectorToArray<9>( topology );
        /* 0 = (-1, -1)
         * 1 = (0, -1)
         * 2 = (1, -1)
         * 3 = (-1, 0)
         * 4 = (0, 0)
         * 5 = (1, 0)
         * 6 = (-1, 1)
         * 7 = (0, 1)
         * 8 = (1, 1) 
         */
        

        // Initialize values that are shared for all edges
        Kokkos::parallel_for("init edge shared values", Kokkos::RangePolicy<execution_space>(0, _edges.size()),
            KOKKOS_LAMBDA(int i) {

            e_vid(i, 2) = -1; // No edge has been split
            e_pid(i) = -1;
            e_cids(i, 0) = -1; e_cids(i, 1) = -1;
            e_owner(i) = rank;
            e_layer(i) = 0;
        });

        auto z = array.view();
        Kokkos::parallel_for("populate_ve", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
            KOKKOS_LAMBDA(int i, int j) {

            // Initialize vertices
            int v_lid = (i - istart) * (jend - jstart) + (j - jstart);
            int v_gid_ = vef_gid_start(rank, 0) + v_lid;
            //printf("i/j/vid: %d, %d, %d\n", i, j, v_lid);
            v_gid(v_lid) = v_gid_;
            v_owner(v_lid) = rank;

            /* Initialize edges
             * Edges between vertices for their:
             *  1. North and south neighbors
             *  2. East and west neighbors
             *  3. Northeast and southwest neighbors
             * Populate edges from west to east and clockwise
             */
            int v_gid_other, e_lid, neighbor_rank, offset;
            if ((i+1 < iend) && (j+1 < jend))
            {
                // Edge 0: north
                v_gid_other = vef_gid_start(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                //printf("R%d: e_gid: %d, e_lid: %d, v_lid: %d\n", rank, vef_gid_start(rank, 1) + e_lid, e_lid, v_lid);
                e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                

                // Edge 1: northeast
                v_gid_other = vef_gid_start(rank, 0) + (i+1 - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3 + 1;
                e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                // Edge 2: east
                v_gid_other = vef_gid_start(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                //printf("ij: %d, %d: e3: vid: %d, vo: %d\n", i, j, v_lid, v_lid_other);
            }
            // Boundary edges on east boundary
            else if ((i == iend-1) && (j < jend-1))
            {   
                // Edge 0: north
                v_gid_other = vef_gid_start(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                // Edges 1 and 2
                neighbor_rank = device_topology[5];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 1
                    offset = v_lid % (jend-jstart);
                    v_gid_other = vef_gid_start(neighbor_rank, 0) + offset + 1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                    // Edge 2
                    v_gid_other = vef_gid_start(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                }
            }
            // Boundary edges on north boundary
            else if ((j == jend-1) && (i < iend-1))
            {
                // Edge 2: east
                v_gid_other = vef_gid_start(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                // Edges 0 and 1
                neighbor_rank = device_topology[7];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 0
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                    // Edge 1
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start(neighbor_rank, 0) + (offset+1) * (iend-istart);
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                }
            }

            // Edges crosses 2 MPI boundaries.
            else
            {
                // Edge 0
                neighbor_rank = device_topology[7];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                }

                // Edge 1
                neighbor_rank = device_topology[8];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    v_gid_other = vef_gid_start(neighbor_rank, 0);
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                }

                // Edge 2
                neighbor_rank = device_topology[5];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    offset = v_lid % (jend-jstart);
                    v_gid_other = vef_gid_start(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                }
            }
        });
        Kokkos::fence();

        // All initialized faces are on the same level
        _max_tree_level = 0;

        _finializeInit();
    }
        

    /**
     * Refine all faces specified in the fids vector 
     * Calling this function increments the version of the mesh
     * and updates global IDS
     * 
     * @param fids: local IDs of faces to be refined
     */
    template <class View>
    void refine(View& fgids)
    {
        // Refining outdates halo data
        _halo_level = 0; _halo_depth = 0;
        _vert_halo_export.clear();
        _edge_halo_export.clear();
        _face_halo_export.clear();

        _refineFaces(fgids);

        // Increase max tree depth by 1
        _max_tree_level++;

        // _sort_by_layer();
        _populate_boundary_elements();
        _version++;
    }

    /**
     * Gather vertices, edges, and faces at the given
     * level of the tree and any levels higher (i.e.
     * their children) and for the given depth into
     * remotely owned sections of the mesh
     * 
     * Populates the vhalo, ehalo, fhalo objects
     */
    void gather(int level, int depth)
    {
        if (depth < 1)
            throw std::runtime_error(
                    "NuMesh::Mesh: halo depth of gather must be at least 1." );
        if (level < 0)
            throw std::runtime_error(
                    "NuMesh::Mesh: level of gather must be at least 1." );
        if (level > _max_tree_level)
            throw std::runtime_error(
                    "NuMesh::Mesh: level of gather must be at <= max tree level." );
        
        _halo_level = level; _halo_depth = depth;
        _gather_depth_one();
        // printf("R%d: after gather: %d verts\n", _rank, _vertices.size());
        // if (_rank == 1)
        // {
        //     printf("R%d: total verts: %d\n", _rank, _mesh->vertices().size());
        //     _mesh->printVertices();
        // }
        _version++;
    }
    
    v_array_type& vertices() {return _vertices;}
    e_array_type& edges() {return _edges;}
    f_array_type& faces() {return _faces;}
    std::vector<std::shared_ptr<halo_aosoa>> halo_export(Vertex) const {return _vert_halo_export;}
    std::vector<halo_aosoa> halo_export(Edge) const {return _edge_halo_export;}
    std::vector<halo_aosoa> halo_export(Face) const {return _face_halo_export;}
    int halo_level() const {return _halo_level;}
    int halo_depth() const {return _halo_depth;}

    int depth() const {return _max_tree_level;}
    int count(Own, Vertex) const {return _owned_vertices;}
    int count(Own, Edge) const {return _owned_edges;}
    int count(Own, Face) const {return _owned_faces;}
    int count(Ghost, Vertex) const {return _ghost_vertices;}
    int count(Ghost, Edge) const {return _ghost_edges;}
    int count(Ghost, Face) const {return _ghost_faces;}

    MPI_Comm comm() {return _comm;}
    int version() const {return _version;}
    int rank() const {return _rank;}

    auto vef_gid_start() {return _vef_gid_start;}
    auto neighbors() {return _neighbors;}
    auto boundary_edges() {return _boundary_edges;}
    auto boundary_faces() {return _boundary_faces;}

    void printVertices()
    {
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        auto v_owner = Cabana::slice<V_OWNER>(_vertices);
        for (int i = 0; i < (int) _vertices.size(); i++)
        {
            printf("R%d: %d, %d\n", _rank,
                v_gid(i), 
                v_owner(i));
        }
    }
    /**
     * opt: 1 = specific edge, 2 = owned, 3 = own+ghost
     */
    void printEdges(int opt, int egid)
    {
        auto e_vid = Cabana::slice<E_VIDS>(_edges);
        auto e_children = Cabana::slice<E_CIDS>(_edges);
        auto e_parent = Cabana::slice<E_PID>(_edges);
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_owner = Cabana::slice<E_OWNER>(_edges);
        auto e_layer = Cabana::slice<E_LAYER>(_edges);
        int rank = _rank;
        if (opt == 1)
        {
            int e_gid_start = _vef_gid_start(_rank, 1);
            int owned_edges = _owned_edges;
            Kokkos::parallel_for("print edges", Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(int x) {
            
            if (x == 0) {
            int elid = egid - e_gid_start;
            if ((elid >= 0) && (elid < owned_edges))
            {
            int i = elid;
            printf("i%d, e%d, v(%d, %d, %d), c(%d, %d), p(%d), L%d, %d, %d\n", i,
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1), e_vid(i, 2),
                e_children(i, 0), e_children(i, 1),
                e_parent(i), e_layer(i),
                e_owner(i), rank);
            }
            }
            });
            return;
        }
        int start = 0, end = _edges.size();
        if (opt == 2) end = _owned_edges;
        Kokkos::parallel_for("print edges", Kokkos::RangePolicy<execution_space>(start, end),
            KOKKOS_LAMBDA(int i) {
            
            printf("i%d, e%d, v(%d, %d, %d), c(%d, %d), p(%d), L%d, %d, %d\n", i,
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1), e_vid(i, 2),
                e_children(i, 0), e_children(i, 1),
                e_parent(i), e_layer(i),
                e_owner(i), rank);

        });
    }

    /**
     * opt = 0: print all faces
     * opt = 1: print fgid face
     */
    void printFaces(int opt, int fgid)
    {
        auto f_egids = Cabana::slice<F_EIDS>(_faces);
        auto f_vgids = Cabana::slice<F_VIDS>(_faces);
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_parent = Cabana::slice<F_PID>(_faces);
        auto f_child = Cabana::slice<F_CID>(_faces);
        auto f_owner = Cabana::slice<F_OWNER>(_faces);
        auto f_layer = Cabana::slice<F_LAYER>(_faces);
        if (opt == 0)
        {
            // Print all faces
            for (int i = 0; i < (int) _faces.size(); i++)
            {
                printf("R%d: i%d, f%d, v(%d, %d, %d), e(%d, %d, %d), c(%d, %d, %d, %d), p(%d), L%d, %d\n", _rank, i,
                    f_gid(i),
                    f_vgids(i, 0), f_vgids(i, 1), f_vgids(i, 2),
                    f_egids(i, 0), f_egids(i, 1), f_egids(i, 2),
                    f_child(i, 0), f_child(i, 1), f_child(i, 2), f_child(i, 3),
                    f_parent(i),
                    f_layer(i),
                    f_owner(i));
            }
        }
        else if (opt == 1)
        {
            // Print one face
            int flid = fgid - _vef_gid_start(_rank, 2);
            if ((flid >= 0) && (flid < _owned_faces))
            {
                int i = flid;
                printf("R%d: i%d, f%d, v(%d, %d, %d), e(%d, %d, %d), c(%d, %d, %d, %d), p(%d), L%d, %d\n", _rank, i,
                    f_gid(i),
                    f_vgids(i, 0), f_vgids(i, 1), f_vgids(i, 2),
                    f_egids(i, 0), f_egids(i, 1), f_egids(i, 2),
                    f_child(i, 0), f_child(i, 1), f_child(i, 2), f_child(i, 3),
                    f_parent(i),
                    f_layer(i),
                    f_owner(i));
            }
        }
    }

  private:
    MPI_Comm _comm;
    int _rank, _comm_size;

    //MPIX_Comm* _xcomm;
    //MPIX_Info* _xinfo;

    // AoSoAs for the mesh
    v_array_type _vertices;
    e_array_type _edges;
    f_array_type _faces;
    int _owned_vertices, _owned_edges, _owned_faces, _ghost_vertices, _ghost_edges, _ghost_faces;

    // Neighbor ranks and boundary data
    Kokkos::View<int*, memory_space> _neighbors;      // Ranks that own a vertex of an owned edge or face
    Kokkos::View<int*, memory_space> _boundary_edges; // GIDs of edges on an MPI boundary
    Kokkos::View<int*, memory_space> _boundary_faces; // GIDs of faces on an MPI boundary

    // How many vertices, edges, and faces each process owns
    // Index = rank
    Kokkos::View<int*[3], memory_space> _vef_gid_start;

    // Version number to keep mesh in sync with other objects. Updates on mesh refinement
    int _version;

    // Halos associated with this mesh version
    std::vector<std::shared_ptr<halo_aosoa>> _vert_halo_export;
    std::vector<std::shared_ptr<halo_aosoa>> _edge_halo_export;
    std::vector<std::shared_ptr<halo_aosoa>> _face_halo_export;
    int _halo_level, _halo_depth;

    // Max depth of tree
    int _max_tree_level;
};
//---------------------------------------------------------------------------//

// Static type checkers
template <typename T>
struct is_numesh_mesh : std::false_type {};
template <typename ExecutionSpace, typename MemSpace>
struct is_numesh_mesh<NuMesh::Mesh<ExecutionSpace, MemSpace>> : std::true_type {};

/**
 *  Returns a mesh with no vertices, edges, or faces.
 */
template <class ExecutionSpace, class MemorySpace>
auto createEmptyMesh( MPI_Comm comm )
{
    return std::make_shared<Mesh<ExecutionSpace, MemorySpace>>(comm);
}

} // end namespace NuMesh


#endif // NUMESH_MESH_HPP