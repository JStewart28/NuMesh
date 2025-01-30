#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
#include <NuMesh_Mesh.hpp>
#include <NuMesh_Maps.hpp>

#include "_hypre_parcsr_ls.h"

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
    using int_d = Kokkos::View<size_t, memory_space>;

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
        const int level = _level, rank = _rank, tree_depth = _mesh->depth();

        // Define the hash map type
        using PairType = std::pair<int, int>;
        using KeyType = uint64_t;  // Hashable key
        using MapType = Kokkos::UnorderedMap<KeyType, int, memory_space>;

        // Hash function to combine two integers into a single key
        auto hashFunction = KOKKOS_LAMBDA(int first, int second) -> KeyType {
            return (static_cast<KeyType>(first) << 32) | (static_cast<KeyType>(second) & 0xFFFFFFFF);
        };

        auto& vertices = _mesh->vertices();
        auto& edges = _mesh->edges();
        auto& faces = _mesh->faces();
        auto boundary_faces = _mesh->boundary_faces();
        size_t num_boundary_faces = boundary_faces.extent(0);
        // size_t num_boundary_edges = _mesh->boundary_edges().extent(0);

        auto vef_gid_start = _mesh->vef_gid_start();

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
        auto f_cid = Cabana::slice<F_CID>(faces);
        auto f_pid = Cabana::slice<F_PID>(faces);
        auto f_rank = Cabana::slice<F_OWNER>(faces);
        auto f_layer = Cabana::slice<F_LAYER>(faces);

        int vertex_count = _mesh->count(Own(), Vertex());
        int edge_count = _mesh->count(Own(), Edge());
        int face_count = _mesh->count(Own(), Face());

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
        using halo_aosoa = Cabana::AoSoA<Cabana::MemberTypes<int, int>, memory_space, 4>;
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

        // Finalize halo sizes and set duplicates to -1
        Kokkos::deep_copy(vhalo_size, vh_idx);
        Kokkos::deep_copy(ehalo_size, eh_idx);
        Kokkos::deep_copy(fhalo_size, fh_idx);
        vert_halo_export.resize(vhalo_size);
        edge_halo_export.resize(ehalo_size);
        face_halo_export.resize(fhalo_size);

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

        // Set halo version
        _halo_version = _mesh->version();

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
        vertices.resize(vhalo.numLocal() + vhalo.numGhost());
        edges.resize(ehalo.numLocal() + ehalo.numGhost());
        faces.resize(fhalo.numLocal() + fhalo.numGhost());
        
        Cabana::gather(vhalo, vertices);
        Cabana::gather(ehalo, edges);
        Cabana::gather(fhalo, faces);
        
        // Update ghost counts in mesh
        _mesh->set(Own(), Vertex(), vhalo.numLocal());
        _mesh->set(Own(), Edge(), ehalo.numLocal());
        _mesh->set(Own(), Face(), fhalo.numLocal());
        _mesh->set(Ghost(), Vertex(), vhalo.numGhost());
        _mesh->set(Ghost(), Edge(), ehalo.numGhost());
        _mesh->set(Ghost(), Face(), fhalo.numGhost());
        
    }

    void gather()
    {
        _gather_depth_one();
        // if (_rank == 1)
        // {
        //     printf("R%d: total verts: %d\n", _rank, _mesh->vertices().size());
        //     _mesh->printVertices();
        // }
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