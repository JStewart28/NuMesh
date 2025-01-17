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
    Halo( std::shared_ptr<Mesh> mesh, const int level, const int depth )
        : _mesh ( mesh )
        , _level ( level )
        , _depth ( depth )
        , _comm ( mesh->comm() )
        , _halo_version ( mesh->version() )
    {
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required" );

        if (_depth < 1)
        {
            throw std::runtime_error(
                    "NuMesh::Halo must be initialized with a halo depth of at least 1." );
        }

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
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
     * Helper function. Given an AoSoA slices of ID and send_to_ranks,
     * if any ID+send_to_rank appears more than once, set its
     * send_to_rank to -1 so Cabana ignores it.
     * 
     * slice<0> must be IDs
     * slice<1> must be to_ranks
     * 
     * Returns the number of duplicates
     */
    template <class AoSoAType>
    void _set_duplicates_neg1(AoSoAType& aosoa)
    {
        // Number of tuples
        const std::size_t num_tuples = aosoa.size();

        // Slices
        auto ids = Cabana::slice<0>(aosoa);
        auto to_ranks = Cabana::slice<1>(aosoa);

        // Define the hash map type
        using PairType = std::pair<int, int>;
        using KeyType = uint64_t;  // Hashable key
        using MapType = Kokkos::UnorderedMap<KeyType, int, memory_space>;

        // Hash function to combine two integers into a single key
        auto hashFunction = KOKKOS_LAMBDA(int first, int second) -> KeyType {
            return (static_cast<KeyType>(first) << 32) | (static_cast<KeyType>(second) & 0xFFFFFFFF);
        };

        // Estimate the capacity of the hash map (somewhat larger than `num_tuples` for performance)
        const size_t hash_capacity = 2 * num_tuples;

        // Create the hash map
        MapType hash_map(hash_capacity);

        // Mark duplicates
        Kokkos::parallel_for(
            "MarkDuplicates", Kokkos::RangePolicy<execution_space>(0, num_tuples),
            KOKKOS_LAMBDA(const std::size_t i) {
                int id = ids(i);
                int to_rank = to_ranks(i);

                // Compute the hash for the tuple
                KeyType hash_key = hashFunction(id, to_rank);

                // Try to insert into the hash map
                auto result = hash_map.insert(hash_key, 1);
                if (!result.success()) {
                    // If insertion fails, it's a duplicate
                    to_ranks(i) = -1;
                }
            });

        // Ensure all parallel operations are completed
        Kokkos::fence();
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
    void _gather_depth_one()
    {
        const int level = _level, rank = _rank;

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
        // (global ID, to_rank, from_rank) tuples
        Cabana::AoSoA<Cabana::MemberTypes<int, int, int>, memory_space, 4>
            distributor_export("distributor_export", num_boundary_faces*2);

        // Halo data
        // (global ID, to_rank) tuples
        using halo_aosoa = Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4>;
        halo_aosoa vert_halo_export("vert_halo_export", num_boundary_faces*4 + 10); // Slight buffer for faces that must be sent to >1 processes
        halo_aosoa edge_halo_export("edge_halo_export", num_boundary_edges*4 + 10); 
        halo_aosoa face_halo_export("face_halo_export", num_boundary_faces*2 + 10);

        // Counters
        int_d de_idx("de_idx"); Kokkos::deep_copy(de_idx, 0);
        int_d vh_idx("vh_idx"); Kokkos::deep_copy(vh_idx, 0);
        int_d eh_idx("eh_idx"); Kokkos::deep_copy(eh_idx, 0);
        int_d fh_idx("fh_idx"); Kokkos::deep_copy(fh_idx, 0);

        // Slices
        auto distributor_export_gids = Cabana::slice<0>(distributor_export);
        auto distributor_export_to_ranks = Cabana::slice<1>(distributor_export);
        auto distributor_export_from_ranks = Cabana::slice<2>(distributor_export);
        auto vert_halo_export_lids = Cabana::slice<0>(vert_halo_export);
        auto vert_halo_export_ranks = Cabana::slice<1>(vert_halo_export);
        auto edge_halo_export_lids = Cabana::slice<0>(edge_halo_export);
        auto edge_halo_export_ranks = Cabana::slice<1>(edge_halo_export);
        auto face_halo_export_lids = Cabana::slice<0>(face_halo_export);
        auto face_halo_export_ranks = Cabana::slice<1>(face_halo_export);

        // _mesh->printFaces(1, 46);
        // Iterate over boundary faces
        Kokkos::parallel_for("boundary face iteration", Kokkos::RangePolicy<execution_space>(0, num_boundary_faces),
            KOKKOS_LAMBDA(int face_idx) {
        
            int fgid = boundary_faces(face_idx);
            // if (rank == 1) printf("R%d: boundary face gid: %d\n", rank, fgid);
            int flid = fgid - vef_gid_start_d(rank, 2);
            int face_level = f_layer(flid);
            if (face_level != level) return; // Only consider elements at our level and their children
            for (int i = 0; i < 3; i++)
            {
                int vgid = f_vid(flid, i);
                int vert_owner = Utils::owner_rank(Vertex(), vgid, vef_gid_start_d);
                // if (fgid == 121) printf("R%d, fgid %d, i%d: vert %d, owner: %d\n", rank, fgid, i, vgid, vert_owner);
                if (vert_owner != rank)
                {
                    // Add this vert to the distributor
                    int dvdx = Kokkos::atomic_fetch_add(&de_idx(), 1);
                    distributor_export_gids(dvdx) = vgid;
                    distributor_export_from_ranks(dvdx) = rank;
                    distributor_export_to_ranks(dvdx) = vert_owner;
                    // if (fgid == 121) printf("R%d: from fgid %d adding to dist vgid %d: (to R%d)\n", rank, fgid, vgid, vert_owner);

                    // Add this face to the halo to send to the given vertex owner
                    int fdx = Kokkos::atomic_fetch_add(&fh_idx(), 1);
                    face_halo_export_lids(fdx) = flid;
                    face_halo_export_ranks(fdx) = vert_owner;
                    
                    // Add owned edges
                    for (int j = 0; j < 3; j++)
                    {
                        int egid = f_eid(flid, j);
                        int edge_owner = Utils::owner_rank(Edge(), egid, vef_gid_start_d);
                        if (edge_owner != rank) continue; // Only add edges we own
                        int edx = Kokkos::atomic_fetch_add(&eh_idx(), 1);
                        int elid = egid - vef_gid_start_d(rank, 1);
                        edge_halo_export_lids(edx) = elid;
                        edge_halo_export_ranks(edx) = vert_owner;
                    }

                    // Add owned verts
                    for (int k = 0; k < 3; k++)
                    {
                        int vgid1 = f_vid(flid, k);
                        int vowner = Utils::owner_rank(Vertex(), vgid1, vef_gid_start_d);
                        if (vowner != rank) continue; // Can't export verts we don't own
                        int vdx = Kokkos::atomic_fetch_add(&vh_idx(), 1);
                        int vlid1 = vgid1 - vef_gid_start_d(rank, 0);
                        vert_halo_export_lids(vdx) = vlid1;
                        vert_halo_export_ranks(vdx) = vert_owner;
                        // if (rank == 1) printf("R%d: sending vgid %d to %d\n", rank, vgid1, vert_owner);
                    }

                    // Can't break loop in case a face has two unowned vertices
                }
            }
        });
                
        // Resize data to correct sizes, except verts because we will get more from the distributor
        int distributor_size, ehalo_size, fhalo_size;
        Kokkos::deep_copy(distributor_size, de_idx);
        Kokkos::deep_copy(ehalo_size, eh_idx);
        Kokkos::deep_copy(fhalo_size, fh_idx);
        // printf("R%d: sizes: %d, %d, %d; actual: %d, %d, %d\n", rank, distributor_size, ehalo_size, fhalo_size,
        //     distributor_export.size(), edge_halo_export.size(), face_halo_export.size());
        distributor_export.resize(distributor_size);
        edge_halo_export.resize(ehalo_size);
        face_halo_export.resize(fhalo_size);

        // Update slices after resizing
        distributor_export_gids = Cabana::slice<0>(distributor_export);
        distributor_export_to_ranks = Cabana::slice<1>(distributor_export);
        distributor_export_from_ranks = Cabana::slice<2>(distributor_export);
        vert_halo_export_lids = Cabana::slice<0>(vert_halo_export);
        vert_halo_export_ranks = Cabana::slice<1>(vert_halo_export);
        edge_halo_export_lids = Cabana::slice<0>(edge_halo_export);
        edge_halo_export_ranks = Cabana::slice<1>(edge_halo_export);
        face_halo_export_lids = Cabana::slice<0>(face_halo_export);
        face_halo_export_ranks = Cabana::slice<1>(face_halo_export);


        // Set duplicate edge/face + send_to_rank pairs to be ignored (sent to rank -1)
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, distributor_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("before R%d: to: R%d, vert gid: %d\n", rank,
        //         distributor_export_to_ranks(i), distributor_export_gids(i));

        // });
        _set_duplicates_neg1(distributor_export);
        _set_duplicates_neg1(edge_halo_export);
        _set_duplicates_neg1(face_halo_export);
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, distributor_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("after R%d: to: R%d, vert gid: %d\n", rank,
        //         distributor_export_to_ranks(i), distributor_export_gids(i));

        // });
        // printf("R%d: dups: %d, %d, %d\n", rank, distributor_dups, edge_dups, face_dups);
        /**
         * Distribute vertex data, then add imported vertex data to the vertex halo
         */
        auto distributor = Cabana::Distributor<memory_space>(_comm, distributor_export_to_ranks);
        const int distributor_total_num_import = distributor.totalNumImport();
        Cabana::AoSoA<Cabana::MemberTypes<int, int, int>, memory_space, 4>
            distributor_import("distributor_import", distributor_total_num_import);
        Cabana::migrate(distributor, distributor_export, distributor_import);
        
        // Distributor import slices
        auto distributor_import_gids = Cabana::slice<0>(distributor_import);
        auto distributor_import_to_ranks = Cabana::slice<1>(distributor_import);
        auto distributor_import_from_ranks = Cabana::slice<2>(distributor_import);

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, distributor_import.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (rank == 1) printf("R%d: R%d asking for vert gid %d\n", rank,
        //         distributor_import_from_ranks(i), distributor_import_gids(i));

        // });

        Kokkos::parallel_for("add imported distributor vertex data",
            Kokkos::RangePolicy<execution_space>(0, distributor_total_num_import),
            KOKKOS_LAMBDA(int i) {
            
            int vgid = distributor_import_gids(i);
            int vlid = vgid - vef_gid_start_d(rank, 0);
            int from_rank = distributor_import_from_ranks(i);

            // Add to vert halo
            int vdx = Kokkos::atomic_fetch_add(&vh_idx(), 1);
            vert_halo_export_lids(vdx) = vlid;
            vert_halo_export_ranks(vdx) = from_rank;
            // if (vgid == 16 && from_rank == 3) printf("R%d: adding vgid %d to R%d to halo at vdx %d\n", rank, vgid, from_rank, vdx);
            
        });

        // Resize vert halo
        int vhalo_size;
        Kokkos::deep_copy(vhalo_size, vh_idx);
        vert_halo_export.resize(vhalo_size);

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     if (vert_halo_export_ranks(i) == 3 && vert_halo_export_gids(i) == 16) printf("R%d: to: R%d, vert gid: %d\n", rank,
        //         vert_halo_export_ranks(i), vert_halo_export_gids(i));

        // });

        _set_duplicates_neg1(vert_halo_export);

        // Set halo version
        _halo_version = _mesh->version();

        // Create halos
        Cabana::Halo<memory_space> vhalo( _comm, vertex_count, vert_halo_export_lids, vert_halo_export_ranks);
        Cabana::Halo<memory_space> ehalo( _comm, edge_count, edge_halo_export_lids, edge_halo_export_ranks);
        Cabana::Halo<memory_space> fhalo( _comm, face_count, face_halo_export_lids, face_halo_export_ranks);

        // Resize and gather
        vertices.resize(vhalo.numLocal() + vhalo.numGhost());
        edges.resize(ehalo.numLocal() + ehalo.numGhost());
        faces.resize(fhalo.numLocal() + fhalo.numGhost());
        Cabana::gather(vhalo, vertices);
        Cabana::gather(ehalo, edges);
        Cabana::gather(fhalo, faces);

        // Update ghost counts in mesh
        

        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0, vert_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {
            
        //     int id = vert_halo_export_gids(i);
        //     int owner = Utils::owner_rank(Vertex(), id, vef_gid_start_d);
        //     if (owner != rank) printf("R%d: vert %d not owned!\n", rank, id);
        //     // if (rank == 1 && vert_halo_export_ranks(i) != -1) printf("R%d: to: R%d, vert gid: %d\n", rank,
        //     //     vert_halo_export_ranks(i), vert_halo_export_gids(i));

        // });
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0,edge_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {

        //     int id = edge_halo_export_gids(i);
        //     int owner = Utils::owner_rank(Edge(), id, vef_gid_start_d);
        //     if (owner != rank) printf("R%d: edge %d not owned!\n", rank, id);
        //     // if (rank == 1 && edge_halo_export_ranks(i) != -1) printf("R%d: to: R%d, edge gid: %d\n", rank,
        //     //     edge_halo_export_ranks(i),edge_halo_export_gids(i));

        // });
        // Kokkos::parallel_for("print", Kokkos::RangePolicy<execution_space>(0,face_halo_export.size()),
        //     KOKKOS_LAMBDA(int i) {
            
        //     int id = face_halo_export_gids(i);
        //     int owner = Utils::owner_rank(Face(), id, vef_gid_start_d);
        //     if (owner != rank) printf("R%d: face %d not owned!\n", rank, id);
        //     // if (rank == 1 && face_halo_export_ranks(i) != -1) printf("R%d: to: R%d, face gid: %d\n", rank,
        //     //     face_halo_export_ranks(i),face_halo_export_gids(i));

        // });

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
    }

    void gather()
    {
        _gather_depth_one();
    }

    /**
     * Returns the version of the mesh the halo is built from
     */
    int version() {return _halo_version;}

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    int _rank, _comm_size;   

    // Level of tree to halo at
    const int _level;

    // Halo depth into neighboring processes
    const int _depth;

    // Mesh version this halo is built from
    int _halo_version; 
};

template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, const int level, const int depth)
{
    return Halo(mesh, level, depth);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP