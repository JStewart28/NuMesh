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
    {
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        // XXX - Optimize these approximations
        _num_vertices = guess;
        _num_edges = _num_vertices*3; // There will be about thrice as many vertices as edges,
                                      // but a split edge adds two edges and one vertex
        _num_faces = _num_vertices;   // Each face contains three edges

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

        gather_depth_one();
    };

    ~Halo()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
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
        
    }

    /**
     * Helper function. Given sorted AoSoA slices of ID and send_to_ranks,
     * if any ID+send_to_rank appears more than once, set its
     * send_to_rank to -1 so Cabana ignores it.
     * 
     * Returns the number of duplicates
     */
    template <class SliceType>
    int _set_duplicates_neg1(SliceType& ids, SliceType& to_ranks)
    {
        if ( ids.size() != to_ranks.size() )
            throw std::runtime_error( "NuMesh::Halo::_set_duplicates_neg1: gid and to_ranks sizes do not match!" );

        int_d counter("counter");
        Kokkos::deep_copy(counter, 0);
        Kokkos::parallel_for("set duplicates to -1", Kokkos::RangePolicy<execution_space>(1, ids.size()),
            KOKKOS_LAMBDA(int i) {

            int current, prev, current_to, prev_to;
            current = ids(i);
            prev = ids(i-1);
            current_to = to_ranks(i);
            prev_to = to_ranks(i-1);
            if ((current == prev) && (current_to == prev_to))
            {
                to_ranks(i) = -1;
                Kokkos::atomic_increment(&counter());
            }
        });
        int num_dups;
        Kokkos::deep_copy(num_dups, counter);

        return num_dups;
    }

    /**
     * Find the vertices, edges, and/or faces that must be exchanged for the halo 
     * 
     * Step 1:
     *  Use distributor to request missing vertex data for owned faces which
     *      contain an unowned vertex. At the same time, build the halo data
     *      Approach:
     *          - Iterate over boundary faces:
     *              1. Add to distributor_export: Any vertex GIDs we do not own
     *              2. Add to halo_export_ids: All vef data on faces that contain an unowned vertex
     *              3. Add to halo_export_ranks: The owner rank of the unowned vertex
     * 
     * Step 2:
     *  - Distribute the distributor_export data into distributor_import
     *  - Add imported vertex GIDs to halo_export_ids and halo_export_ranks.
     * 
     */
    void gather_depth_one()
    {
        const int level = _level, rank = _rank, comm_size = _comm_size;

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();
        auto faces = _mesh->faces();
        auto boundary_faces = _mesh->boundary_faces();
        size_t num_boundary_faces = boundary_faces.extent(0);
        size_t num_boundary_edges = _mesh->boundary_edges().extent(0);

        // Get vef_gid_start and copy to device
        auto vef_gid_start = _mesh->vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        // Vertex slices
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Edge slices
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_rank = Cabana::slice<E_OWNER>(edges);
        auto e_cid = Cabana::slice<E_CIDS>(edges);
        auto e_pid = Cabana::slice<E_PID>(edges);
        auto e_layer = Cabana::slice<E_LAYER>(edges);

        // Face slices
        auto f_gid = Cabana::slice<F_GID>(faces);
        auto f_eid = Cabana::slice<F_EIDS>(faces);
        auto f_vid = Cabana::slice<F_VIDS>(faces);
        auto f_rank = Cabana::slice<F_OWNER>(faces);
        auto f_layer = Cabana::slice<F_LAYER>(faces);

        int vertex_count = _mesh->count(Own(), Vertex());
        int edge_count = _mesh->count(Own(), Edge());
        int face_count = _mesh->count(Own(), Face());

        // Distributor data
        // (global ID, from_rank, to_rank) tuples
        Cabana::AoSoA<Cabana::MemberTypes<int, int, int>, memory_space, 4>
            distributor_export("distributor_export", num_boundary_faces);
        auto distributor_export_gids = Cabana::slice<0>(distributor_export);
        auto distributor_export_from_ranks = Cabana::slice<1>(distributor_export);
        auto distributor_export_to_ranks = Cabana::slice<2>(distributor_export);
        int_d de_idx("de_idx");
        Kokkos::deep_copy(de_idx, 0);

        // Halo data
        // (global ID, to_rank) tuples
        using halo_aosoa = Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4>;
        halo_aosoa vert_halo_export("vert_halo_export", num_boundary_faces*2 + 10); // Slight buffer for faces that must be sent to >1 processes
        halo_aosoa edge_halo_export("edge_halo_export", num_boundary_edges*2 + 10); 
        halo_aosoa face_halo_export("face_halo_export", num_boundary_faces + 10);
        auto vert_halo_export_gids = Cabana::slice<0>(vert_halo_export);
        auto vert_halo_export_ranks = Cabana::slice<1>(vert_halo_export);
        auto edge_halo_export_gids = Cabana::slice<0>(edge_halo_export);
        auto edge_halo_export_ranks = Cabana::slice<1>(edge_halo_export);
        auto face_halo_export_gids = Cabana::slice<0>(face_halo_export);
        auto face_halo_export_ranks = Cabana::slice<1>(face_halo_export);
        int_d vh_idx("vh_idx"); Kokkos::deep_copy(vh_idx, 0);
        int_d eh_idx("eh_idx"); Kokkos::deep_copy(eh_idx, 0);
        int_d fh_idx("fh_idx"); Kokkos::deep_copy(fh_idx, 0);

        // Iterate over boundary faces
        Kokkos::parallel_for("boundary face iteration", Kokkos::RangePolicy<execution_space>(0, num_boundary_faces),
            KOKKOS_LAMBDA(int face_idx) {
        
            int fgid = boundary_faces(face_idx);
            int flid = fgid - vef_gid_start_d(rank, 2);
            for (int i = 0; i < 3; i++)
            {
                int vgid = f_vid(flid, i);
                int vert_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start_d);
                if (vert_owner != rank)
                {
                    // Add this vert to the distributor
                    int vdx = Kokkos::atomic_fetch_add(&de_idx(), 1);
                    distributor_export_gids(i) = vgid;
                    distributor_export_from_ranks(i) = rank;
                    distributor_export_to_ranks(i) = vert_owner;

                    // Add this face to the halo to send to the given vertex owner
                    int fdx = Kokkos::atomic_fetch_add(&fh_idx(), 1);
                    face_halo_export_gids(fdx) = fgid;
                    face_halo_export_ranks(fdx) = vert_owner;
                    
                    // Add edges
                    for (int j = 0; j < 3; j++)
                    {
                        int egid = f_eid(flid, j);
                        int edx = Kokkos::atomic_fetch_add(&eh_idx(), 1);
                        edge_halo_export_gids(edx) = egid;
                        edge_halo_export_ranks(edx) = vert_owner;
                    }

                    // Add all owned verts
                    for (int k = 0; k < 3; k++)
                    {
                        if (k == i) continue;
                        int vgid = f_vid(flid, k);
                        int vdx = Kokkos::atomic_fetch_add(&vh_idx(), 1);
                        vert_halo_export_gids(edx) = vgid;
                        vert_halo_export_ranks(edx) = vert_owner;
                    }

                    // Can't break loop in case a face has two unowned vertices
                }
            }
        });

        // Resize data to correct sizes, except verts because we will get more from the distributor
        int distributor_size, vhalo_size, ehalo_size, fhalo_size;
        Kokkos::deep_copy(distributor_size, de_idx);
        Kokkos::deep_copy(ehalo_size, eh_idx);
        Kokkos::deep_copy(fhalo_size, fh_idx);
        distributor_export.resize(distributor_size);
        edge_halo_export.resize(ehalo_size);
        face_halo_export.resize(fhalo_size);


        // Set duplicate edge/face + send_to_rank pairs to be ignored (sent to rank -1)
        // 1. Sort the distributor data by increasing VGID to filter duplicates
        auto sort_export_gids = Cabana::sortByKey( distributor_export_gids );
        Cabana::permute( sort_export_gids, distributor_export );
        auto sort_halo_edges = Cabana::sortByKey( edge_halo_export_gids );
        Cabana::permute( sort_halo_edges, edge_halo_export );
        auto sort_halo_faces = Cabana::sortByKey( face_halo_export_gids );
        Cabana::permute( sort_halo_faces, face_halo_export );
        

        // 2. Set duplicate tuples to -1
        int distributor_dups = _set_duplicates_neg1(distributor_export_gids, distributor_export_ranks);
        int edge_dups = _set_duplicates_neg1(edge_halo_export_gids, edge_halo_export_ranks);
        int face_dups = _set_duplicates_neg1(face_halo_export_gids, face_halo_export_ranks);



        /***********
         * Step 1
         **********/
        // {
        // auto distributor_export_gids = Cabana::slice<0>(distributor_export);
        // auto distributor_export_from_ranks = Cabana::slice<1>(distributor_export);
        // auto distributor_export_to_ranks = Cabana::slice<2>(distributor_export);
        
        // Kokkos::parallel_for("fill distributor data", Kokkos::RangePolicy<execution_space>(0, boundary_edges.extent(0)),
        //     KOKKOS_LAMBDA(int edge_idx) {
            
        //     int egid = boundary_edges(edge_idx);
        //     int elid = Utils::get_lid(e_gid, egid, 0, edge_count);
        //     // if (rank == 3) printf("R%d: elg: %d, %d\n", rank, elid, egid);

        //     for (int v = 0; v < 2; v++)
        //     {
        //         int vgid = e_vids(elid, v);
        //         int vertex_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start_d);
        //         // if (rank == 3) printf("R%d: elg: %d, %d, vg: %d, vowner: %d, neighbor rank: %d\n", rank, elid, egid,
        //         //     vgid, vertex_owner, neighbor_rank);
        //         if (vertex_owner != rank)
        //         {
        //             int idx = Kokkos::atomic_fetch_add(&counter(), 1);
        //             distributor_export_to_ranks(idx) = vertex_owner;
        //             distributor_export_gids(idx) = vgid;
        //             distributor_export_from_ranks(idx) = rank;
        //         }
        //     }
        // });
        // }

        // // Resize distributor data to correct sizes
        // int distributor_total_num_export;
        // Kokkos::deep_copy(distributor_total_num_export, counter);
        // distributor_export.resize(distributor_total_num_export);

        // // Sort the distributor data by increasing VGID to filter duplicates
        // auto distributor_export_gids = Cabana::slice<0>(distributor_export);
        // auto distributor_export_from_ranks = Cabana::slice<1>(distributor_export);
        // auto distributor_export_to_ranks = Cabana::slice<2>(distributor_export);
        // // Sort by GID
        // auto sort_export_gids = Cabana::sortByKey( distributor_export_gids );
        // Cabana::permute( sort_export_gids, distributor_export );
        

        // // Iterate over the distributor data to remove repeated vertices
        // // i.e. set their to_rank to -1 so the distributor ignores them
        // // Also count repeats for sizing vertex seeds appropiately
        // Kokkos::deep_copy(counter, 0);
        // Kokkos::parallel_for("set duplicates to -1", Kokkos::RangePolicy<execution_space>(1, distributor_total_num_export),
        //     KOKKOS_LAMBDA(int i) {

        //     int current, prev;
        //     current = distributor_export_gids(i);
        //     prev = distributor_export_gids(i-1);
        //     if (current == prev)
        //     {
        //         distributor_export_to_ranks(i) = -1;
        //         Kokkos::atomic_increment(&counter());
        //     }
        // });
        // int num_dups;
        // Kokkos::deep_copy(num_dups, counter);

        // /***********
        //  * Step 2
        //  **********/
        // auto distributor = Cabana::Distributor<memory_space>(_comm, distributor_export_to_ranks);
        // const int distributor_total_num_import = distributor.totalNumImport();
        // distributor_data_aosoa_t distributor_import("distributor_import", distributor_total_num_import);
        // Cabana::migrate(distributor, distributor_export, distributor_import);
        // auto distributor_import_gids = Cabana::slice<0>(distributor_import);
        // auto distributor_import_from_ranks = Cabana::slice<1>(distributor_import);
        // auto distributor_import_to_ranks = Cabana::slice<2>(distributor_import);

        // // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_export),
        // //     KOKKOS_LAMBDA(int i) {

        // //     if (rank == 1) printf("R%d: to: R%d, data: (%d, %d)\n", rank,
        // //         distributor_export_to_ranks(i), distributor_export_gids(i), distributor_export_from_ranks(i));

        // // });
        
        // // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_import),
        // //     KOKKOS_LAMBDA(int i) {

        // //     if (rank == 1) printf("R%d: from: R%d, data: (%d, %d)\n", rank,
        // //         distributor_import_from_ranks(i), distributor_import_gids(i), distributor_import_from_ranks(i));

        // // });
        // // printf("*****\n");

        // /***********
        //  * Step 3
        //  **********/
    
        // // Reset index counters
        // _vdx = 0; _edx = 0; _fdx = 0;

        // auto neighbor_ranks = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _mesh->neighbors());

        //  // View for ID halo data
        // auto vertex_send_ids = _vertex_send_ids;

        // // We must add data one rank at a time to keep the data structures packed
        // for (size_t i = 0; i < neighbor_ranks.extent(0); i++)
        // {
        //     const int neighbor_rank = neighbor_ranks(i);
        //     // if (rank == 1) printf("R%d: neighbor_rank: %d\n", rank, neighbor_rank);

        //     // Store true if added to the halo to avoid uniqueness searches
        //     Kokkos::View<bool*, memory_space> vb("vb", vertex_count);
        //     Kokkos::View<bool*, memory_space> eb("eb", edge_count);
        //     Kokkos::View<bool*, memory_space> fb("fb", face_count);
        //     Kokkos::deep_copy(vb, false);
        //     Kokkos::deep_copy(eb, false);
        //     Kokkos::deep_copy(fb, false);

        //     // Set the offsets where the data for this rank will start in the indices views
        //     auto v_subview = Kokkos::subview(_vertex_send_offsets, neighbor_rank);
        //     Kokkos::deep_copy(v_subview, _vdx);
        //     auto e_subview = Kokkos::subview(_edge_send_offsets, neighbor_rank);
        //     Kokkos::deep_copy(e_subview, _edx);
        //     auto f_subview = Kokkos::subview(_face_send_offsets, neighbor_rank);
        //     Kokkos::deep_copy(f_subview, _fdx);

        //     int_d vdx_d("vdx_d");
        //     //int_d edx_d("edx_d");
        //     // int_d fdx_d("fdx_d");
        //     Kokkos::deep_copy(vdx_d, _vdx);
        //     //Kokkos::deep_copy(edx_d, _edx);
        //     // Kokkos::deep_copy(fdx_d, _fdx);

        //     // Add imported VGIDs to seeds and to IDs halo data
        //     integer_view vgid_seeds("vgid_seeds", distributor_total_num_import + distributor_total_num_export - num_dups);
        //     Kokkos::deep_copy(vgid_seeds, -1);
        //     int_d vgid_seeds_idx("vgid_seeds_idx");
        //     Kokkos::deep_copy(vgid_seeds_idx, 0);
        //     Kokkos::parallel_for("add import VGIDS", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_import),
        //         KOKKOS_LAMBDA(int j) {

        //         int from_rank = distributor_import_from_ranks(j);
        //         if (from_rank != neighbor_rank) return;

        //         int vgid = distributor_import_gids(j);
        //         int vdx = Kokkos::atomic_fetch_add(&vdx_d(), 1);
        //         int svdx = Kokkos::atomic_fetch_add(&vgid_seeds_idx(), 1);
        //         // if (rank == 1) printf("R%d: adding import VGID %d at %d\n", rank, vgid, vdx);
        //         vgid_seeds(svdx) = vgid;
        //         vertex_send_ids(vdx) = vgid;

        //         // Set these vgids as added to the halo
        //         // Don't need atomics because the vlid will be unique for the kernel
        //         int vlid = vgid - vef_gid_start_d(rank, 0);
        //         vb(vlid) = true;
        //         // if (rank == 1) printf("R%d: setting vb(%d) = %d\n", rank, vlid, vb(vlid));

        //     });

        //     // Update offset counter
        //     Kokkos::deep_copy(_vdx, vdx_d);

        //     // Add exported VGIDs to seeds
        //     Kokkos::parallel_for("add export VGIDS", Kokkos::RangePolicy<execution_space>(0, distributor_total_num_export),
        //         KOKKOS_LAMBDA(int j) {

        //         int to_rank = distributor_export_to_ranks(j);
        //         if (to_rank != neighbor_rank) return;

        //         int vgid = distributor_export_gids(j);
        //         int svdx = Kokkos::atomic_fetch_add(&vgid_seeds_idx(), 1);
        //         vgid_seeds(svdx) = vgid;

        //     });

        //     // Shrink seed view to appropriate num_vert_seeds
        //     int vgid_seeds_size;
        //     Kokkos::deep_copy(vgid_seeds_size, vgid_seeds_idx);
        //     Kokkos::resize(vgid_seeds, vgid_seeds_size);

            // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vgid_seeds_size),
            // KOKKOS_LAMBDA(int i) {

            //     if (rank == 1) printf("R%d: vgid_seeds(%d): %d\n", rank, i, vgid_seeds(i));

            // });

            // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vb.extent(0)),
            // KOKKOS_LAMBDA(int i) {

            //     if (rank == 1) printf("R%d: vb(%d): %d\n", rank, i, vb(i));

            // });
            // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, eb.extent(0)),
            // KOKKOS_LAMBDA(int i) {

            //     if (rank == 1) printf("R%d: eb(%d): %d\n", rank, i, eb(i));

            // });

            // Build halo data for this rank
            // collect_entities(vgid_seeds, vb, eb, fb);

            // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vgid_seeds.extent(0)),
            //     KOKKOS_LAMBDA(int i) {

            //     if (rank == 1) printf("R%d: to: %d, seed%d: %d\n", rank, neighbor_rank, i, vgid_seeds(i));

            // });
            // if (i == 1) break;
        

        // auto vertex_send_offsets = _vertex_send_offsets;
        // // auto vertex_send_ids = _vertex_send_ids;
        // int d_rank = 1;
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, neighbor_ranks.extent(0)),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == d_rank) printf("R%d: rank: %d, V offset: %d\n", rank, neighbor_ranks(i), vertex_send_offsets(neighbor_ranks(i)));

        // });

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, _vdx),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == d_rank) printf("R%d: offset %d: VGID: %d\n", rank, i, vertex_send_ids(i));

        // });

        // auto edge_send_offsets = _edge_send_offsets;
        // auto edge_send_ids = _edge_send_ids;
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, neighbor_ranks.extent(0)),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == d_rank) printf("R%d: rank: %d, E offset: %d\n", rank, neighbor_ranks(i), edge_send_offsets(neighbor_ranks(i)));

        // });

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, _edx),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == d_rank) printf("R%d: offset %d: EGID: %d\n", rank, i, edge_send_ids(i));

        // });

        // auto face_send_offsets = _face_send_offsets;
        // auto face_send_ids = _face_send_ids;
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, neighbor_ranks.extent(0)),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == d_rank) printf("R%d: rank: %d, F offset: %d\n", rank, neighbor_ranks(i), face_send_offsets(neighbor_ranks(i)));

        // });

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, _fdx),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == d_rank) printf("R%d: offset %d: FGID: %d\n", rank, i, face_send_ids(i));

        // });


        // Update halo version
        _halo_version = _mesh->version();

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
    void collect_entities(const VertexView& v, bool_view& vb, bool_view& eb, bool_view& fb)
    {
        if (_depth > 5)
        {
            throw std::runtime_error(
                "NuMesh::Halo::collect_entities: a maxmimum halo depth of 5 is supported" );
        }

        if (_depth >= 1)
        {
            // if (_rank == 1) printf("R%d: start 1: vdx: %d, edx: %d, v extent: %d\n", 
            //     _rank, _vdx, _edx, (int)v.extent(0));
            auto out1 = gather_local_neighbors(v, vb, eb, fb);
            // End depth 1
            if (_depth >= 2)
            {
                // if (_rank == 1) printf("R%d: start 2: vdx: %d, edx: %d, out1 extent: %d\n", 
                //     _rank, _vdx, _edx, (int)out1.extent(0));
                // int rank = _rank;
                // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, out1.extent(0)),
                //     KOKKOS_LAMBDA(int i) {

                //     if (rank == 1) printf("R%d: depth2: seed(%d): %d\n", rank, i, out1(i));

                // });
                auto out2 = gather_local_neighbors(out1, vb, eb, fb);
                // End depth 2
                if (_depth >= 3)
                {
                    auto out3 = gather_local_neighbors(out2, vb, eb, fb);
                    // End depth 3
                    if (_depth >= 4)
                    {
                        auto out4 = gather_local_neighbors(out3, vb, eb, fb);
                        // End depth 4
                        if (_depth >= 5)
                        {
                            auto out5 = gather_local_neighbors(out4, vb, eb, fb);
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
    integer_view gather_local_neighbors(const integer_view& verts, bool_view& vb, bool_view& eb, bool_view& fb)
    {
        if (_halo_version != _mesh->version())
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: mesh and halo version do not match" );
        }

        const int level = _level, rank = _rank, comm_size = _comm_size;

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();
        auto faces = _mesh->faces();
        auto boundary_faces = _mesh->boundary_faces();
        size_t num_boundary_faces = boundary_faces.extent(0);

        // Get vef_gid_start and copy to device
        auto vef_gid_start = _mesh->vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        // auto vertex_face_offsets = _v2f.offsets();
        // auto vertex_face_indices = _v2f.indices();

        // Vertex slices
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Edge slices
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_rank = Cabana::slice<E_OWNER>(edges);
        auto e_cid = Cabana::slice<E_CIDS>(edges);
        auto e_pid = Cabana::slice<E_PID>(edges);
        auto e_layer = Cabana::slice<E_LAYER>(edges);

        // Face slices
        auto f_gid = Cabana::slice<F_GID>(faces);
        auto f_eid = Cabana::slice<F_EIDS>(faces);
        auto f_vid = Cabana::slice<F_VIDS>(faces);
        auto f_rank = Cabana::slice<F_OWNER>(faces);
        auto f_layer = Cabana::slice<F_LAYER>(faces);

        /**
         * Allocate views. Each vertex can have at most six edges and six vertices
         * extending from it.
         * 
         * XXX - optimize this num_vert_seeds estimate
         */
        size_t num_vert_seeds = verts.extent(0);
        integer_view vids_view("vids_view", num_vert_seeds*6);
        integer_view eids_view("eids_view", num_vert_seeds*6);
        integer_view fids_view("fids_view", num_vert_seeds*6);
        int_d vertex_idx("vertex_idx");
        int_d edge_idx("edge_idx");
        int_d face_idx("face_idx");
        Kokkos::deep_copy(vertex_idx, 0);
        Kokkos::deep_copy(edge_idx, 0);
        Kokkos::deep_copy(face_idx, 0);
        // printf("R%d: vef view sizes: %d, %d, %d\n", rank, vids_view.extent(0), eids_view.extent(0), fids_view.extent(0));
        int owned_vertices = _mesh->count(Own(), Vertex());
        int owned_faces = _mesh->count(Own(), Face());
        Kokkos::parallel_for("gather neighboring vertex and edge IDs", Kokkos::RangePolicy<execution_space>(0, num_vert_seeds),
            KOKKOS_LAMBDA(int i) {
            
            int vgid = verts(i);
            // if (rank == 1) printf("R%d: seed VGID: %d\n", rank, vgid);
            int vlid = vgid - vef_gid_start_d(rank, 0);
            if ((vlid >= 0) && (vlid < owned_vertices)) // If we own the vertex
            {
                int offset = vertex_face_offsets(vlid);
                int next_offset = (vlid + 1 < (int) vertex_face_offsets.extent(0)) ? 
                          vertex_face_offsets(vlid + 1) : 
                          (int) vertex_face_indices.extent(0);

                // Loop over faces connected to each vertex
                for (int f = offset; f < next_offset; f++)
                {
                    int flid = vertex_face_indices(f); // Local face ID
                    int fgid = f_gid(flid);
                    // if (rank == 1) printf("R%d: seed vgid (owned): %d, f l/g: %d, %d\n", rank, vgid, flid, fgid);
                    
                    int flayer = f_layer(flid);
                    int fowner = Utils::owner_rank(Face(), fgid, vef_gid_start_d);
                    if ((flayer == level) && (fowner == rank))
                    {
                        // Add face to list of faces if it's on our layer
                        // AND it is not ghosted
                        // AND it is not already added
                        bool fval = Kokkos::atomic_compare_exchange(&fb(flid), false, true);
                        if (!fval)
                        {
                            int fdx = Kokkos::atomic_fetch_add(&face_idx(), 1);
                            fids_view(fdx) = fgid;

                            // Loop over edges that make up this face and add to the
                            // halo, following the same conditions above
                            for (int e = 0; e < 3; e++)
                            {
                                int egid = f_eid(flid, e);
                                int elid = egid - vef_gid_start_d(rank, 1);
                                int eowner = Utils::owner_rank(Edge(), egid, vef_gid_start_d);
                                // The edge will always be on the same level as the face
                                if ((elid > -1) && (eowner == rank))
                                {
                                    bool eval = Kokkos::atomic_compare_exchange(&eb(elid), false, true);
                                    if (!eval)
                                    {
                                        int edx = Kokkos::atomic_fetch_add(&edge_idx(), 1);
                                        eids_view(edx) = egid;
                                        // if (rank == 1) printf("R%d: for VGID %d, adding EGID %d\n", rank, vgid, egid);

                                        // Add the vertices connected by this edge
                                        for (int v = 0; v < 2; v++)
                                        {
                                            int edge_vgid = e_vid(elid, v);
                                            // if (rank == 1) printf("R%d: egid %d, checking neighbor_vgid %d: %d\n", rank, egid, v, neighbor_vgid);
                                            int edge_vlid = edge_vgid - vef_gid_start_d(rank, 0);
                                            int vertex_owner = Utils::owner_rank(Vertex(), edge_vgid, vef_gid_start_d);
                                            // printf("R%d: EGID: %d, neighbor_vgid: %d, neighbor_vlid")
                                            if (vertex_owner == rank)
                                            {
                                                // if (rank == 1) printf("R%d: egid %d, neighbor_vgid %d: %d\n", rank, egid, v, neighbor_vgid);
                                                bool vval = Kokkos::atomic_compare_exchange(&vb(edge_vlid), false, true);
                                                if (!vval)
                                                {
                                                    int vdx = Kokkos::atomic_fetch_add(&vertex_idx(), 1);
                                                    vids_view(vdx) = edge_vgid;
                                                    // if (rank == 1) printf("R%d: from edge %d and VGID %d, adding VGID %d\n", rank, egid, vgid, neighbor_vgid);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                }
            }

            /**
             * If these are the initial seeds, we may not own the vertex, but the vertex
             * may be connected by a face we own. Find this face with a linear search through
             * the boundary faces
             * XXX - Make this search more efficient
             */
            else
            {
                for (size_t f = 0; f < num_boundary_faces; f++)
                {
                    int fgid = boundary_faces(f);
                    int flid = fgid - vef_gid_start_d(rank, 2);
                    if ((flid >= 0) && (flid < owned_faces))
                    {
                        // Ensure this face is on our level
                        if (f_layer(flid) != level) continue;

                        // Check if this face holds the vertex in question
                        for (int j = 0; j < 3; j++)
                        {
                            int fxvgid = f_vid(flid, j);
                            // if (vgid == 12) printf("R%d: vgid: %d, exvgid: %d\n", rank, vgid, exvgid);
                            if (fxvgid == vgid)
                            {
                                // Add this face, its edges, and its vertices to the halo
                                // following the same procedure as above
                                bool fval = Kokkos::atomic_compare_exchange(&fb(flid), false, true);
                                if (!fval)
                                {
                                    int fdx = Kokkos::atomic_fetch_add(&face_idx(), 1);
                                    fids_view(fdx) = fgid;
                                    // if (rank == 1) printf("R%d: seed vgid (remote): %d, f l/g: %d, %d, fdx: %d\n", rank, vgid, flid, fgid, fdx);

                                    // Loop over edges that make up this face and add to the
                                    // halo, following the same conditions above
                                    for (int e = 0; e < 3; e++)
                                    {
                                        int egid = f_eid(flid, e);
                                        int elid = egid - vef_gid_start_d(rank, 1);
                                        int eowner = Utils::owner_rank(Edge(), egid, vef_gid_start_d);
                                        // The edge will always be on the same level as the face
                                        if ((elid > -1) && (eowner == rank))
                                        {
                                            bool eval = Kokkos::atomic_compare_exchange(&eb(elid), false, true);
                                            if (!eval)
                                            {
                                                int edx = Kokkos::atomic_fetch_add(&edge_idx(), 1);
                                                eids_view(edx) = egid;
                                                // if (rank == 1) printf("R%d: for VGID %d, adding EGID %d\n", rank, vgid, egid);

                                                // Add the vertices connected by this edge
                                                for (int v = 0; v < 2; v++)
                                                {
                                                    int edge_vgid = e_vid(elid, v);
                                                    // if (rank == 1) printf("R%d: egid %d, checking neighbor_vgid %d: %d\n", rank, egid, v, neighbor_vgid);
                                                    int edge_vlid = edge_vgid - vef_gid_start_d(rank, 0);
                                                    int vertex_owner = Utils::owner_rank(Vertex(), edge_vgid, vef_gid_start_d);
                                                    // printf("R%d: EGID: %d, neighbor_vgid: %d, neighbor_vlid")
                                                    if (vertex_owner == rank)
                                                    {
                                                        // if (rank == 1) printf("R%d: egid %d, neighbor_vgid %d: %d\n", rank, egid, v, neighbor_vgid);
                                                        bool vval = Kokkos::atomic_compare_exchange(&vb(edge_vlid), false, true);
                                                        if (!vval)
                                                        {
                                                            int vdx = Kokkos::atomic_fetch_add(&vertex_idx(), 1);
                                                            vids_view(vdx) = edge_vgid;
                                                            // if (rank == 1) printf("R%d: from edge %d and VGID %d, adding VGID %d\n", rank, egid, vgid, neighbor_vgid);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });
        Kokkos::fence();
        int num_verts, num_edges, num_faces;
        Kokkos::deep_copy(num_verts, vertex_idx);
        Kokkos::deep_copy(num_edges, edge_idx);
        Kokkos::deep_copy(num_faces, face_idx);

        // For each edge, add all child edges and vertices connected to those child edges to the halo
        Kokkos::parallel_for("gather child vertex and edge IDs", Kokkos::RangePolicy<execution_space>(0, num_edges),
            KOKKOS_LAMBDA(int i) {
        
            int egid = eids_view(i);
            int elid = egid - vef_gid_start_d(rank, 1);
            for (int j = 0; j < 2; j++)
            {
                int ecid = e_cid(elid, j);
                while (ecid != -1)
                {
                    int eclid = ecid - vef_gid_start_d(rank, 1);

                    // Add the third vertex of the parent edge to the halo
                    int eplid = e_pid(eclid) - vef_gid_start_d(rank, 1);
                    int vdx = Kokkos::atomic_fetch_add(&vertex_idx(), 1);
                    vids_view(vdx) = e_vid(eplid, 2);

                    // Add the child edge to the halo
                    int edx = Kokkos::atomic_fetch_add(&edge_idx(), 1);
                    eids_view(edx) = ecid;

                    // Update ecid for for this edge's child
                    ecid = e_cid(eclid, j);
                }
            }
        
        });
        Kokkos::deep_copy(num_verts, vertex_idx);
        Kokkos::deep_copy(num_edges, edge_idx);
        // printf("R%d: num verts, edges: %d, %d\n", _rank, num_verts, num_edges);

        // Resize views and remove unique values
        Kokkos::resize(vids_view, num_verts);
        Kokkos::resize(eids_view, num_edges);
        Kokkos::resize(fids_view, num_faces);

        // auto unique_vids = Utils::filter_unique(vids_view);
        // auto unique_eids = Utils::filter_unique(eids_view);

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, eids_view.extent(0)),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("R%d: eids_view(%d): %d\n", rank, i, eids_view(i));

        // });

        /**
         * Add IDs to offset views starting at offsets
         * _vdx and _edx
         */
        if (_vdx+num_verts > (int)_vertex_send_ids.extent(0))
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: _vertex_send_ids not large enough to hold new vertices" );
        }
        if (_edx+num_edges > (int)_edge_send_ids.extent(0))
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: _edge_send_ids not large enough to hold new edges" );
        }
        if (_fdx+num_faces > (int)_face_send_ids.extent(0))
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: _face_send_ids not large enough to hold new faces" );
        }
        auto v_subview = Kokkos::subview(_vertex_send_ids, std::make_pair(_vdx, _vdx + num_verts));
        Kokkos::deep_copy(v_subview, vids_view);
        auto e_subview = Kokkos::subview(_edge_send_ids, std::make_pair(_edx, _edx + num_edges));
        Kokkos::deep_copy(e_subview, eids_view);
        auto f_subview = Kokkos::subview(_face_send_ids, std::make_pair(_fdx, _fdx + num_faces));
        Kokkos::deep_copy(f_subview, fids_view);

        // Update values of _vdx and _edx
        // if (rank == 1) printf("R%d: old edx: %d, new: %d\n", rank, _edx, _edx+num_edges);
        _vdx += num_verts; _edx += num_edges; _fdx += num_faces;

        return vids_view;
    }

    /**
     * Returns the version of the mesh the halo is built from
     */
    int version() {return _halo_version;}

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
auto createHalo(std::shared_ptr<Mesh> mesh, Edge, const int level, const int depth)
{
    return Halo(mesh, 1, level, depth);
}
template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, Edge, const int level, const int depth, const int guess)
{
    return Halo(mesh, 1, level, depth, guess);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP