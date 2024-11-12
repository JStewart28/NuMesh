#ifndef NUMESH_MESH_HPP
#define NUMESH_MESH_HPP

// XXX - Add mapping class.

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

// #include <mpi_advance.h>

#ifndef AOSOA_SLICE_INDICES
#define AOSOA_SLICE_INDICES 1
#endif

// Constants for slice indices
#if AOSOA_SLICE_INDICES
    #define S_V_GID 0
    #define S_V_OWNER 1
    #define S_E_VIDS 0
    #define S_E_FIDS 1
    #define S_E_CIDS 2
    #define S_E_PID 3
    #define S_E_GID 4
    #define S_E_OWNER 5
    #define S_F_CID 0
    #define S_F_VIDS 1
    #define S_F_EIDS 2
    #define S_F_GID 3
    #define S_F_PID 4
    #define S_F_OWNER 5
#endif

//#include <NuMesh_Communicator.hpp>
//#include <NuMesh_Grid2DInitializer.hpp>
#include <NuMesh_Types.hpp>

#include <limits>

namespace NuMesh
{

template <std::size_t Size, class Scalar>
auto vectorToArray( std::vector<Scalar> vector )
{
    Kokkos::Array<Scalar, Size> array;
    for ( std::size_t i = 0; i < Size; ++i )
        array[i] = vector[i];
    return array;
}

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

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    // Note: Larger types should be listed first
    using vertex_data = Cabana::MemberTypes<int,       // Vertex global ID                                 
                                            int,       // Owning rank
                                            >;
    using edge_data = Cabana::MemberTypes<  int[2],    // Vertex global ID endpoints of edge    
                                            int[2],    // Face global IDs. The face where it is
                                                       // the lowest edge, starrting at the first
                                                       // vertex and going clockwise, is the first edge.                      
                                            int[2],    // Child edge global IDs, going clockwise from
                                                       // the first vertex of the face
                                            int,       // Parent edge global ID
                                            int,       // Edge global ID
                                            int,       // Owning rank
                                            >;
    using face_data = Cabana::MemberTypes<  int[4],    // Child face global IDs
                                            int[3],    // Vertex global IDs that make up face
                                            int[3],    // Edge global IDs that make up face 
                                            int,       // Face global ID
                                            int,       // Parent face global ID                        
                                            int,       // Owning rank
                                            >;
                                            
    // XXX Change the final parameter of particle_array_type, vector type, to
    // be aligned with the machine we are using
    using v_array_type = Cabana::AoSoA<vertex_data, memory_space, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, memory_space, 4>;
    using f_array_type = Cabana::AoSoA<face_data, memory_space, 4>;
    using size_type = typename memory_space::size_type;

    // Construct a mesh.
    Mesh( MPI_Comm comm ) : _comm ( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
        //MPIX_Info_init(&_xinfo);
        //MPIX_Comm_init(&_xcomm, _comm);
        //MPIX_Comm_topo_init(_xcomm);

        _version = 0;

        _vef_gid_start = Kokkos::View<int*[3], Kokkos::HostSpace>("_vef_gid_start", _comm_size);
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
     * Create faces from vertices and edges
     * Each vertex is associated with at least one face
     */
    void createFaces()
    {
        /* Each vertex contributes 2 faces */
        _faces.resize(_owned_faces);
    
        auto v_gid = Cabana::slice<S_V_GID>(_vertices);
        auto v_owner = Cabana::slice<S_V_OWNER>(_vertices);

        auto e_vid = Cabana::slice<S_E_VIDS>(_edges); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(_edges);

        auto f_vgids = Cabana::slice<S_F_VIDS>(_faces);
        auto f_egids = Cabana::slice<S_F_EIDS>(_faces);
        auto f_gid = Cabana::slice<S_F_GID>(_faces);
        auto f_parent = Cabana::slice<S_F_PID>(_faces);
        auto f_child = Cabana::slice<S_F_CID>(_faces);
        auto f_owner = Cabana::slice<S_F_OWNER>(_faces);

        int rank = _rank;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], MemorySpace> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("Initialize faces", Kokkos::RangePolicy<execution_space>(0, _vertices.size()), KOKKOS_LAMBDA(int i) {
            // Face 1: "left" face; face 2: "right" face   
            int f_lid;

            // Find face 1 values
            // Get the three vertices and edges for face1
            int v_gid0, v_gid1, v_gid2;
            int e_gid0, e_lid0, e_gid1, e_gid2, e_lid2;
            v_gid0 = v_gid(i);
            // Follow first edge to get next vertex
            e_gid0 = v_gid0*3; e_lid0 = e_gid0 - vef_gid_start_d(rank, 1);
            v_gid1 = e_vid(e_lid0, 1);
            // Use second vertex to get next edge
            e_gid1 = v_gid1*3+2;
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - vef_gid_start_d(rank, 1);
            v_gid2 = e_vid(e_lid2, 1);
            
            // Populate face 1 values
            f_lid = i*2;
            f_vgids(f_lid, 0) = v_gid0; f_vgids(f_lid, 1) = v_gid1; f_vgids(f_lid, 2) = v_gid2;
            f_egids(f_lid, 0) = e_gid0; f_egids(f_lid, 1) = e_gid1; f_egids(f_lid, 2) = e_gid2;
            f_gid(f_lid) = f_lid + vef_gid_start_d(rank, 2);
            f_parent(f_lid) = -1;
            f_child(f_lid, 0) = -1; f_child(f_lid, 1) = -1; f_child(f_lid, 2) = -1; f_child(f_lid, 3) = -1;
            f_owner(f_lid) = rank;
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
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - vef_gid_start_d(rank, 1);
            v_gid2 = e_vid(e_lid2, 1);
            // DIFFERENT: Use vertex 2 to get edge 1. Edge 1 the first edge of vertex 2
            e_gid1 = v_gid2*3;

            // Populate face 2 values
            f_lid = i*2+1;
            f_vgids(f_lid, 0) = v_gid0; f_vgids(f_lid, 1) = v_gid1; f_vgids(f_lid, 2) = v_gid2;
            f_egids(f_lid, 0) = e_gid0; f_egids(f_lid, 1) = e_gid1; f_egids(f_lid, 2) = e_gid2;
            f_gid(f_lid) = f_lid + vef_gid_start_d(rank, 2);
            f_parent(f_lid) = -1;
            f_child(f_lid, 0) = -1; f_child(f_lid, 1) = -1; f_child(f_lid, 2) = -1; f_child(f_lid, 3) = -1;
            f_owner(f_lid) = rank;


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
     * Assign edges to locally owned faces. Must be completed before gathering edges to 
     * ghosted edges have face global IDs of faces not owned by the remote process.
     */
    void updateEdges()
    {
        auto e_vid = Cabana::slice<S_E_VIDS>(_edges);
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(_edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(_edges);

        auto f_egids = Cabana::slice<S_F_EIDS>(_faces);

        /* Edges will always be one of the following for faces:
         * - 1st edge and 2nd edge
         * - 1st edge and 3rd edge
         * - 2nd edge and 3rd edge
         * 
         * Iterate over faces and assign 1st and 3rd edges' face 1
         */
        int rank = _rank;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], MemorySpace> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("assign_edges13_to_faces", Kokkos::RangePolicy<execution_space>(0, _faces.size()), KOKKOS_LAMBDA(int f_lid) {
            int f_gid, eX_lid; // eX_gid;
            f_gid = f_lid + _vef_gid_start_d(rank, 2);

            // Where this edge is the first edge, set its face1
            // eX_gid = f_egids(f_lid, 0);
            //if (eX_gid == 14) printf("R%d: e1_gid: %d, f_gid: %d\n", rank, eX_gid, f_gid);
            eX_lid = f_egids(f_lid, 0) - _vef_gid_start_d(rank, 1);
            e_fids(eX_lid, 0) = f_gid;
            
            // Where this edge is the third edge, set its face2
            // eX_gid = f_egids(f_lid, 2);
            //if (eX_gid == 14) printf("R%d: e3_gid: %d, f_gid: %d\n", rank, eX_gid, f_gid);
            eX_lid = f_egids(f_lid, 2) - _vef_gid_start_d(rank, 1);
            e_fids(eX_lid, 1) = f_gid;
        });
    }

    /**
     * Gather face IDs from remotely owned faces and assign them to
     * locally owned edges
     */
    void gatherFaces()
    {
    //     /* Temporary naive solution to store which faces are needed from which processes: 
    //      * Create a (comm_size x num_faces) view.
    //      * If a face is needed from another process, set (owner_rank, f_lid) to the 
    //      * global face ID needed from owner_rank.
    //      */ 
    //     // Set a counter to count number messages that will be sent
    //     using CounterView = Kokkos::View<int, device_type, Kokkos::MemoryTraits<Kokkos::Atomic>>;
    //     CounterView counter("counter");
    //     Kokkos::deep_copy(counter, 0);
    //     Kokkos::View<int**, device_type> sendvals_unpacked("sendvals_unpacked", _comm_size, _faces.size());
    //     Kokkos::deep_copy(sendvals_unpacked, -1);
    //     int rank = _rank, comm_size = _comm_size;
    //     // Copy _vef_gid_start to device
    //     Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
    //     auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
    //     Kokkos::deep_copy(hv_tmp, _vef_gid_start);
    //     Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
    //     auto f_egids = Cabana::slice<S_F_EIDS>(_faces);
    //     Kokkos::parallel_for("find_needed_edge2", Kokkos::RangePolicy<execution_space>(0, _faces.size()), KOKKOS_LAMBDA(int f_lid) {
    //         int e2_gid, from_rank = -1;

    //         // Where this edge is the first edge, set its face1
    //         e2_gid = f_egids(f_lid, 1);

    //         // If e2_gid < (rank GID start) or (> (rank+1) GID start), 
    //         // this edge is owned by another process 
    //         if ((rank != comm_size-1) && ((e2_gid < _vef_gid_start_d(rank, 1)) || (e2_gid >= _vef_gid_start_d(rank+1, 1))))
    //         {
    //             if (e2_gid < _vef_gid_start_d(0, 1)) from_rank = 0;
    //             else if (e2_gid >= _vef_gid_start_d(comm_size-1, 1)) from_rank = comm_size-1;
    //             else
    //             {
    //                 for (int r = 0; r < comm_size-1; r++)
    //                 {
    //                     if (r == rank) continue;
    //                     //printf("checking btw R%d: [%d, %d)\n", r, _vef_gid_start_d(r, 1))
    //                     if ((e2_gid >= _vef_gid_start_d(r, 1)) && (e2_gid < _vef_gid_start_d(r+1, 1))) from_rank = r;
    //                 }
    //             }
    //             sendvals_unpacked(from_rank, f_lid) = e2_gid;
    //             counter()++;
    //             //if (rank == 1) printf("R%d: sendvals_unpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvals_unpacked(from_rank, f_lid));

    //         }
    //         // If rank == comsize-1 we need a seperate condition
    //         else if (rank == comm_size-1)
    //         {
    //             if (e2_gid < _vef_gid_start_d(rank, 1))
    //             {
    //                 for (int r = 0; r < rank; r++)
    //                 {
    //                     //printf("checking btw R%d: [%d, %d)\n", r, _vef_gid_start_d(r, 1))
    //                     if ((e2_gid >= _vef_gid_start_d(r, 1)) && (e2_gid < _vef_gid_start_d(r+1, 1)))
    //                     {
    //                         from_rank = r;
    //                         sendvals_unpacked(from_rank, f_lid) = e2_gid;
    //                         counter()++;
    //                         //if (rank == 1) printf("R%d: sendvals_unpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvals_unpacked(from_rank, f_lid));
    //                     }
    //                 }
    //             }
    //         }

    //         // if (e_fids(e2_lid, 0) != -1) e_fids(e2_lid, 0) = f_gid;
    //         // else if (e_fids(e2_lid, 1) != -1) e_fids(e2_lid, 1) = f_gid;
    //     });
    //     Kokkos::fence();
    //     int num_sends = -1;
    //     Kokkos::deep_copy(num_sends, counter);
    }

    /**
     * After the vertex and edge connectivity has been set and
     * global IDs assigned, create faces from the edges and vertices.
     *  1. Create the faces
     *  2. Update the edges to reflect which faces they are part of
     *  3. Gather face global IDs from remotely owned faces to the edges
     */
    void finializeInit()
    {
        createFaces();
        updateEdges();
    }

    
    /**
     * Turn a 2D array into an unstructured mesh of vertices, edges, and faces
     * by turning each (i, j) index into a vertex and creating edges between 
     * all eight neighbor vertices in the grid
     */
    template <class CabanaArray>
    void initializeFromArray( CabanaArray& array )
    {
        static_assert( Cabana::Grid::is_array<CabanaArray>::value, "NuMesh::Mesh::initializeFromArray: Cabana::Grid::Array required" );
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

        updateGlobalIDs();

        // Copy vef_gid_start to device
        Kokkos::View<int*[3], MemorySpace> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        // We should convert the following loops to a Cabana::simd_parallel_for at some point to get better write behavior

        // Initialize the vertices, edges, and faces
        auto v_gid = Cabana::slice<S_V_GID>(_vertices);
        auto v_owner = Cabana::slice<S_V_OWNER>(_vertices);

        auto e_vid = Cabana::slice<S_E_VIDS>(_edges); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(_edges);
        auto e_cids = Cabana::slice<S_E_CIDS>(_edges);
        auto e_pid = Cabana::slice<S_E_PID>(_edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(_edges);
        int rank = _rank;
        auto topology = Cabana::Grid::getTopology( *local_grid );
        auto device_topology = vectorToArray<9>( topology );
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
        Kokkos::parallel_for("populate_ve", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
        KOKKOS_LAMBDA(int i, int j) {

            // Initialize vertices
            int v_lid = (i - istart) * (jend - jstart) + (j - jstart);
            int v_gid_ = vef_gid_start_d(rank, 0) + v_lid;
            //printf("i/j/vid: %d, %d, %d\n", i, j, v_lid);
            v_gid(v_lid) = v_gid_;
            v_owner(v_lid) = rank;
            // for (int dim = 0; dim < 3; dim++) {
            //     v_xyz(v_lid, dim) = z(i, j, dim);
            // }

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
                v_gid_other = vef_gid_start_d(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                //printf("R%d: e_gid: %d, e_lid: %d, v_lid: %d\n", rank, vef_gid_start_d(rank, 1) + e_lid, e_lid, v_lid);
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
                e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                // Edge 1: northeast
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3 + 1;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
                e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                // Edge 2: east
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
                e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                //printf("ij: %d, %d: e3: vid: %d, vo: %d\n", i, j, v_lid, v_lid_other);
            }
            // Boundary edges on east boundary
            else if ((i == iend-1) && (j < jend-1))
            {   
                // Edge 0: north
                v_gid_other = vef_gid_start_d(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
                e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                // Edges 1 and 2
                neighbor_rank = device_topology[5];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 1
                    offset = v_lid % (jend-jstart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset + 1;
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                    // Edge 2
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                }
            }
            // Boundary edges on north boundary
            else if ((j == jend-1) && (i < iend-1))
            {
                // Edge 2: east
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
                e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                // Edges 0 and 1
                neighbor_rank = device_topology[7];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 0
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;

                    // Edge 1
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + (offset+1) * (iend-istart);
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
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
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                } 
                else 
                {
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                }

                // Edge 1
                neighbor_rank = device_topology[8];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                } 
                else 
                {
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0);
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                }

                // Edge 2
                neighbor_rank = device_topology[5];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                } 
                else 
                {
                    offset = v_lid % (jend-jstart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                    e_pid(e_lid) = -1; e_cids(e_lid, 0) = -1; e_cids(e_lid, 1) = -1;
                }
            }
        });
        Kokkos::fence();
        finializeInit();
        // printVertices();
        // printEdges(3);
        // initialize_edges();
    }
        

    /**
     * Refine a face by splitting it into four smaller triangles 
     * Refining a face removes all ghosted data in the AoSoA. This
     * is done in order to maintain the consistency of ghosted data
     * being placed at the end of the AoSoA.
     * 
     * @param fid: local face ID of face to be refined
     * 
     * XXX - Currently only works on interior faces
     */
    void _refineEdges(int p_lfid)
    {        
        /**
         * Determine if neighboring faces have been refined. If so,
         * use those vertices and edges rather than creating new ones.
         * We must do this to avoid redundant vertices and edges
         */
        // Atomic counters
        using CounterView = Kokkos::View<int, memory_space, Kokkos::MemoryTraits<Kokkos::Atomic>>;
        CounterView new_edges("new_edges");
        Kokkos::deep_copy(new_edges, 0);
        int num_new_verts;
        int num_new_edges;
        int edge_children[3][2];
        int edge_children_verts[3][2];

        // Global IDs
        int v_gid_start = _vef_gid_start(_rank, 0);
        int e_gid_start = _vef_gid_start(_rank, 1);
        int f_gid_start = _vef_gid_start(_rank, 2);

        int rank = _rank;
        auto faces = _faces;
        auto edges = _edges;
        // This needs to be done in a parallel for loop to avoid memory space errors
        Kokkos::parallel_for("new_vertices_and_edges", Kokkos::RangePolicy<execution_space>(0, 3),
            KOKKOS_LAMBDA(int i) {
                
            auto face_tuple = faces.getTuple(p_lfid);
            auto ex_lid = Cabana::get<S_F_EIDS>(face_tuple, i) - e_gid_start;
            if (ex_lid < 0) return; // Edge not owned
            auto edge_tuple = edges.getTuple(ex_lid);

            // Check if the edges has children
            int child_id = Cabana::get<S_E_CIDS>(edge_tuple, 0);
            if (child_id == -1) {new_edges()++;}        
        });
        Kokkos::deep_copy(num_new_edges, new_edges);
        num_new_edges *= 2;
        num_new_verts = num_new_edges / 2;
        printf("Refine FID: %d, new ve: %d, %d\n", p_lfid, num_new_verts, num_new_edges+3);
        
        // Create 4 new faces
        int f_lid_start = _owned_faces;
        _owned_faces += 4;
        _faces.resize(_owned_faces);
        auto f_vgids = Cabana::slice<S_F_VIDS>(_faces);
        auto f_egids = Cabana::slice<S_F_EIDS>(_faces);
        auto f_gids = Cabana::slice<S_F_GID>(_faces);
        auto f_pgids = Cabana::slice<S_F_PID>(_faces);
        auto f_cgids = Cabana::slice<S_F_CID>(_faces);
        auto f_ranks = Cabana::slice<S_F_OWNER>(_faces);

        // Create new edges
        int e_lid_start = _owned_edges;
        _owned_edges += 3 + num_new_edges;
        _edges.resize(_owned_edges);
        auto e_gids = Cabana::slice<S_E_GID>(_edges);
        auto e_vids = Cabana::slice<S_E_VIDS>(_edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(_edges);
        auto e_ranks = Cabana::slice<S_E_OWNER>(_edges);
        auto e_cid = Cabana::slice<S_E_CIDS>(_edges);
        auto e_pid = Cabana::slice<S_E_PID>(_edges);

        // Create new vertices
        int v_lid_start = _owned_vertices;
        _owned_vertices += num_new_verts;
        _vertices.resize(_owned_vertices);
        auto v_gids = Cabana::slice<S_V_GID>(_vertices);
        auto v_ranks = Cabana::slice<S_V_OWNER>(_vertices);

        if (num_new_verts == 3)
        {
            // No neighbor faces have been refined and we generate new all new vertices and edges,
            // then assign them to faces
            Kokkos::parallel_for("new_vertices_and_edges", Kokkos::RangePolicy<execution_space>(0, 3),
            KOKKOS_LAMBDA(int i) {
            
                /* Set the 3 new vertices and connect three new edges between them */
        
                // Vertices
                int v_lid = v_lid_start + i;
                int v_gid = v_gid_start + v_lid;
                v_gids(v_lid) = v_gid;
                v_ranks(v_lid) = rank;

                /**
                 * New edges 0, 1, and 2 (of 9)
                 *  Edge 0: connects vertex 0 to vertex 1
                 *  Edge 1: vertex 1 to vertex 2
                 *  Edge 2: vertex 0 to vertex 2
                 * 
                 * No parent edges
                 */
                int e_lid = e_lid_start + i;
                e_gids(e_lid) = e_gid_start + e_lid;
                e_ranks(e_lid) = rank;
                e_cid(e_lid, 0) = -1; e_cid(e_lid, 1) = -1;
                e_pid(e_lid) = -1;
                if (i < 2) {e_vids(e_lid, 0) = v_gid; e_vids(e_lid, 1) = v_gid+1;}
                if (i == 2) {e_vids(e_lid, 0) = v_gid-2; e_vids(e_lid, 1) = v_gid;}
            
                /**
                 * New edges 3, 4, and 5 (of 9)
                 *  Edge 3: 0th vertex of parent face to new vertex 0
                 *      - Parent: edge 0 of parent face
                 *  Edge 4: 1st vertex of parent face to new vertex 1
                 *      - Parent: edge 1 of parent face
                 *  Edge 5: 2nd vertex of parent face to new vertex 2
                 *      - Parent: edge 2 of parent face
                 * 
                 * Also, for the parent edges, set these new edges as their children
                 */
                e_lid += 3;
                int parent_vertex = f_vgids(p_lfid, i);
                e_gids(e_lid) = e_gid_start + e_lid;
                e_ranks(e_lid) = rank;
                e_vids(e_lid, 0) = parent_vertex; e_vids(e_lid, 1) = v_gid;
                int parent_egid = f_egids(p_lfid, i);
                e_pid(e_lid) = parent_egid;
                e_cid(e_lid, 0) = -1; e_cid(e_lid, 1) = -1;
                // Parent edges
                int parent_elid = parent_egid - e_gid_start;
                e_cid(parent_elid, 0) = e_gid_start + e_lid;


                /**
                 * New edges 6, 7, and 8 (of 9, these are the last three)
                 *  Edge 6: new vertex 0 to 1st vertex of parent face
                 *      - Parent: edge 0 of parent face
                 *  Edge 7: new vertex 1 to 2nd vertex of parent face
                 *      - Parent: edge 1 of parent face
                 *  Edge 8: new vertex 2 to 0th vertex of parent face
                 *      - Parent: edge 2 of parent face
                 * 
                 * Also, for the parent edges, set these new edges as their children
                 */
                e_lid += 3;
                e_gids(e_lid) = e_gid_start + e_lid;
                e_ranks(e_lid) = rank;
                if (i < 2) parent_vertex = f_vgids(p_lfid, i+1);
                if (i == 2) parent_vertex = f_vgids(p_lfid, 0);
                e_vids(e_lid, 0) = v_gid; e_vids(e_lid, 1) = parent_vertex;
                e_pid(e_lid) = parent_egid;
                e_cid(e_lid, 0) = -1; e_cid(e_lid, 1) = -1;
                // Parent edges
                e_cid(parent_elid, 1) = e_gid_start + e_lid;
            });
            Kokkos::fence();

            int owned_edges = _owned_edges;
            Kokkos::parallel_for("new_faces", Kokkos::RangePolicy<execution_space>(0, 4),
            KOKKOS_LAMBDA(int i) {

                // Populate these first; they are easy
                int f_lid = f_lid_start + i;
                int f_gid = f_gid_start + f_lid;
                int f_gid_parent = p_lfid + f_gid_start;
                f_gids(f_lid) = f_gid;          // Global ID
                f_ranks(f_lid) = rank;          // Owning rank
                f_pgids(f_lid) = f_gid_parent;  // Parent face
                for (int j = 0; j < 4; j++) {f_cgids(f_lid, j) = -1;}            // No children

                /**
                 * Set vertices for new faces 0, 1, 2, and 3
                 * pv = parent face vertex
                 * nv = new vertex
                 *  Face 0: (pv0, nv0, nv2)
                 *  Face 1: (nv0, pv1, nv1)
                 *  Face 2: (nv2, nv1, pv2)
                 *  Face 3: (nv0, nv1, nv2) -> this is the center face
                 */
                int v0, v1, v2;
                if (i == 0)
                {
                    v0 = f_vgids(p_lfid, 0);
                    v1 = v_gids(v_lid_start);
                    v2 = v_gids(v_lid_start+2);
                }
                if (i == 1)
                {
                    v0 = v_gids(v_lid_start);
                    v1 = f_vgids(p_lfid, 1);
                    v2 = v_gids(v_lid_start+1);
                }
                if (i == 2)
                {
                    v0 = v_gids(v_lid_start+2);
                    v1 = v_gids(v_lid_start+1);
                    v2 = f_vgids(p_lfid, 2);
                }
                if (i == 3)
                {
                    v0 = v_gids(v_lid_start);
                    v1 = v_gids(v_lid_start+1);
                    v2 = v_gids(v_lid_start+2);
                }
                f_vgids(f_lid, 0) = v0;
                f_vgids(f_lid, 1) = v1;
                f_vgids(f_lid, 2) = v2;

                /**
                 * Set these faces as child faces of the face being refined
                 */
                f_cgids(p_lfid, i) = f_gid;

                /**
                 * Assign edges by walking the end of the edge array and
                 * checking which edges match the correct vertices
                 */ 
                int idx = 0;
                for (int j = e_lid_start; j < owned_edges; j++)
                {
                    int ev0 = e_vids(j, 0), ev1 = e_vids(j, 1);
                    if ((ev0 == v0 || ev0 == v1 || ev0 == v2) && (ev1 == v0 || ev1 == v1 || ev1 == v2))
                    {
                        // Both ev0 and ev1 are present in two of either v0, v1, or v2
                        f_egids(f_lid, idx++) = j;
                    }

                }
                
            });
        }

        else
        {
            /**
             * Some, or all, neighbor faces have been refined so we use their vertex
             * and edge data for refined sides.
             * 
             * The AoSoAs may live on the GPU, so all accesses must be made in a parallel
             * for loop. However, there isn't miuch parallelism to be gained, and
             * coding this in parallel is much more complicated, so this is
             * a serial parallel for loop.
             * 
             * NOTES:
             * Do this in phases: figure out all the edges that need to be refined, refine vertexes and edges
             * amd communicate, 
             */
            Kokkos::parallel_for("new_vertices_and_edges", Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(int i) {
            
            /**
             * For each edge, check if it has been previously split (i.e. has children)
             * If so, use this data when refining the face. 
             * If not, split the edge and create a new vertex.
             */
            // int edge_index = e_gid_start - owned_edges;
            // auto face_tuple = faces.getTuple(p_lfid);
            // int e_lid, v0, v1, ce0_lid, ce1_lid;

            // // Edge 0
            // e_lid = Cabana::get<S_F_EIDS>(face_tuple, 0) - e_gid_start;
            // auto edge0 = edges.getTuple(e_lid);
            // ce0_lid = Cabana::get<S_E_CIDS>(edge0, 0) - e_gid_start;
            // ce1_lid = Cabana::get<S_E_CIDS>(edge0, 1) - e_gid_start;
            // if (ce0_lid == -1)
            // {
            //     // No children; this edge needs to be split
            // }


            });
        }
    }

    /**
     * Refine all faces specified in the fids vector 
     * Calling this function increments the version of the mesh
     * and updates global IDS
     * 
     * @param fids: local IDs of faces to be refined
     */
    template <class View>
    void refine(View& fids)
    {
        _version++;
        const int rank = _rank;
        const int num_face_refinements = fids.extent(0);

        // Global IDs
        int v_gid_start = _vef_gid_start(_rank, 0);
        int e_gid_start = _vef_gid_start(_rank, 1);
        // int f_gid_start = _vef_gid_start(_rank, 2);

        /********************************************************
         * Phase 1.0: Collect all edges that need to be refined,
         * then refine them in parallel
         *******************************************************/

        using atomic_view = Kokkos::View<int*, memory_space, Kokkos::MemoryTraits<Kokkos::Atomic>>;
        using counter_view = Kokkos::View<int, memory_space>;
        atomic_view edge_needs_refine("edge_needs_refine", _owned_edges);
        counter_view counter("counter");
        Kokkos::deep_copy(counter, 0);
        auto f_egid = Cabana::slice<S_F_EIDS>(_faces);
        Kokkos::parallel_for("populate edge_needs_refine", Kokkos::RangePolicy<execution_space>(0, num_face_refinements),
            KOKKOS_LAMBDA(int i) {
            
            for (int j = 0; j < 3; j++)
            {
                int ex_lid = f_egid(fids(i), j);
                edge_needs_refine(ex_lid) = 1;
                Kokkos::atomic_increment(&counter()); // Each edge refinement contributes one new vertex
            }
        });
        int new_vertices;
        Kokkos::deep_copy(new_vertices, counter);

        printf("New verts: %d\n", new_vertices);

        for (int i = 0; i < _owned_edges; i++)
        {
            if (edge_needs_refine(i) == 1)
            {
                printf("Edge %d marked\n", i+e_gid_start);
            }
        }

        // Resize arrays
        // int f_lid_start = _owned_faces;
        _owned_faces += num_face_refinements * 4;
        _faces.resize(_owned_faces);
        int e_lid_start = _owned_edges;
        // Each face contributes 3 new edges; each vertex contributes 2
        _owned_edges += num_face_refinements * 3 + new_vertices * 2;
        _edges.resize(_owned_edges);
        int v_lid_start = _owned_vertices;
        _owned_vertices += new_vertices;
        _vertices.resize(_owned_vertices);


        // Vertex slices
        auto v_gid = Cabana::slice<S_V_GID>(_vertices);
        auto v_rank = Cabana::slice<S_V_OWNER>(_vertices);

        // Populate new vertices
        Kokkos::parallel_for("populate new vertices", Kokkos::RangePolicy<execution_space>(0, new_vertices),
            KOKKOS_LAMBDA(int i) {
            
            // Populate new vertices
            int v_l = v_lid_start + i;
            int v_g = v_l + v_gid_start;
            v_gid(v_l) = v_g;
            v_rank(v_l) = rank;

        });

        // Edge slices
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_vid = Cabana::slice<S_E_VIDS>(_edges);
        auto e_fid = Cabana::slice<S_E_FIDS>(_edges);
        auto e_rank = Cabana::slice<S_E_OWNER>(_edges);
        auto e_cid = Cabana::slice<S_E_CIDS>(_edges);
        auto e_pid = Cabana::slice<S_E_PID>(_edges);

        // Refine existing  edges
        Kokkos::deep_copy(counter, 0);
        Kokkos::parallel_for("refine_edges", Kokkos::RangePolicy<execution_space>(0, e_lid_start),
            KOKKOS_LAMBDA(int i) {
            
            // Check if this edge needs to be split
            if (edge_needs_refine(i) == 0) return;

            printf("Refining edge: %d\n", i);
            
            // Create new edge local IDs
            int offset = Kokkos::atomic_fetch_add(&counter(), 1);
            int ec_lid0 = e_lid_start + offset*2;
            int ec_lid1 = e_lid_start + offset*2 + 1;
            int ec_gid0 = e_gid_start + ec_lid0;
            int ec_gid1 = e_gid_start + ec_lid1;

            printf("Offset: %d, adding edges %d, %d\n", offset, ec_lid0, ec_lid1);

            // Global IDs = global ID start + local ID
            e_gid(ec_lid0) = ec_gid0;
            e_gid(ec_lid1) = ec_gid1;

            // Set parent and child edges for the split edges
            e_pid(ec_lid0) = i; e_cid(ec_lid0, 0) = -1; e_cid(ec_lid0, 1) = -1;
            e_pid(ec_lid1) = i; e_cid(ec_lid1, 0) = -1; e_cid(ec_lid1, 1) = -1;
            
            // Set these edges to be the child edges of their parents edges
            e_cid(i, 0) = ec_gid0; e_cid(i, 1) = ec_gid1;

            // Owning rank
            e_rank(ec_lid0) = rank; e_rank(ec_lid1) = rank;

            // Face IDs
            e_fid(ec_lid0, 0) = -1; e_fid(ec_lid0, 1) = -1;
            e_fid(ec_lid1, 0) = -1; e_fid(ec_lid1, 1) = -1;

            /**
             * Set vertices:
             *  ec_lid0: (First vertex of parent edge, new vertex)
             *  ec_lid1: (new vertex, second vertex of parent edge)
             */
            e_vid(ec_lid0, 0) = e_vid(i, 0); e_vid(ec_lid0, 1) = v_lid_start + offset;
            e_vid(ec_lid1, 0) = v_lid_start + offset; e_vid(ec_lid1, 1) = e_vid(i, 1);
        });

        // Populate the three new, internal edges for each face
        Kokkos::parallel_for("new internal edges", Kokkos::RangePolicy<execution_space>(0, num_face_refinements),
            KOKKOS_LAMBDA(int i) {
            
            // Create new edge local IDs
            int offset = Kokkos::atomic_fetch_add(&counter(), 1) * 2;
            int e_lid0 = e_lid_start + offset;
            int e_lid1 = e_lid_start + offset + 1;
            int e_lid2 = e_lid_start + offset + 2;

            printf("Offset: %d, Adding new internal edges %d, %d, %d\n", offset, e_lid0, e_lid1, e_lid2);

            // Global IDs = global ID start + local ID
            e_gid(e_lid0) = e_gid_start + e_lid0;
            e_gid(e_lid1) = e_gid_start + e_lid1;
            e_gid(e_lid2) = e_gid_start + e_lid2;

            // Set parent and child edges
            e_pid(e_lid0) = -1; e_cid(e_lid0, 0) = -1; e_cid(e_lid0, 1) = -1;
            e_pid(e_lid1) = -1; e_cid(e_lid1, 0) = -1; e_cid(e_lid1, 1) = -1;
            e_pid(e_lid2) = -1; e_cid(e_lid2, 0) = -1; e_cid(e_lid2, 1) = -1;

            // Owning rank
            e_rank(e_lid0) = rank; e_rank(e_lid1) = rank; e_rank(e_lid2) = rank;

            // Face IDs
            e_fid(e_lid0, 0) = -1; e_fid(e_lid0, 1) = -1;
            e_fid(e_lid1, 0) = -1; e_fid(e_lid1, 1) = -1;
            e_fid(e_lid2, 0) = -1; e_fid(e_lid2, 1) = -1;

            /**
             * Set vertices:
             *  e_lid0: (new vertex 0, new vertex 1)
             *  e_lid1: (new vertex 1, new vertex 2)
             *  e_lid2: (new vertex 0, new vertex 2)
             */
            e_vid(e_lid0, 0) = v_lid_start + i; e_vid(e_lid0, 1) = v_lid_start + i + 1;
            e_vid(e_lid1, 0) = v_lid_start + i + 1; e_vid(e_lid1, 1) = v_lid_start + i + 2;
            e_vid(e_lid2, 0) = v_lid_start + i; e_vid(e_lid2, 1) = v_lid_start + i + 2;

        });

        /********************************************************
         * Phase 1.1: Communicate new edge and vertex
         * global IDs to all processes
         *******************************************************/
        updateGlobalIDs();
    }

    /**
     * Update all global ID values in the mesh
     */
    void updateGlobalIDs()
    {
        // Temporarily save old starting positions
        Kokkos::View<int*[3], Kokkos::HostSpace> old_vef_start("old_vef_start", _comm_size);
        Kokkos::deep_copy(old_vef_start, _vef_gid_start);

        int vef[3] = {_owned_vertices, _owned_edges, _owned_faces};
        MPI_Allgather(vef, 3, MPI_INT, _vef_gid_start.data(), 3, MPI_INT, _comm);

        // Find where each process starts its global IDs
        for (int i = 1; i < _comm_size; ++i) {
            _vef_gid_start(i, 0) += _vef_gid_start(i - 1, 0);
            _vef_gid_start(i, 1) += _vef_gid_start(i - 1, 1);
            _vef_gid_start(i, 2) += _vef_gid_start(i - 1, 2);
        }
        for (int i = _comm_size - 1; i > 0; --i) {
            _vef_gid_start(i, 0) = _vef_gid_start(i - 1, 0);
            _vef_gid_start(i, 1) = _vef_gid_start(i - 1, 1);
            _vef_gid_start(i, 2) = _vef_gid_start(i - 1, 2);
        }
        _vef_gid_start(0, 0) = 0;
        _vef_gid_start(0, 1) = 0;
        _vef_gid_start(0, 2) = 0;

        if (old_vef_start(0, 0) == -1)
        {
            // Don't update global IDs when the mesh is initially formed
            return;
        }

        int dv = _vef_gid_start(_rank, 0) - old_vef_start(_rank, 0);
        int de = _vef_gid_start(_rank, 1) - old_vef_start(_rank, 1);
        int df = _vef_gid_start(_rank, 2) - old_vef_start(_rank, 2);

        // Update vertices
        auto v_gid = Cabana::slice<S_V_GID>(_vertices);
        Kokkos::parallel_for("update vertex GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_vertices),
            KOKKOS_LAMBDA(int i) { v_gid(i) += dv; });

        // Update edges
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_vid = Cabana::slice<S_E_VIDS>(_edges);
        auto e_fid = Cabana::slice<S_E_FIDS>(_edges);
        auto e_cid = Cabana::slice<S_E_CIDS>(_edges);
        auto e_pid = Cabana::slice<S_E_PID>(_edges);
        Kokkos::parallel_for("update edge GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_edges),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            e_gid(i) += de;

            // Vertex association GIDs
            e_vid(i, 0) += dv; e_vid(i, 1) += dv;

            // Face association GIDs
            if (e_fid(i, 0) != -1) e_fid(i, 0) += df;
            if (e_fid(i, 1) != -1) e_fid(i, 1) += df;

            // Child edge GIDs
            if (e_cid(i, 0) != -1) {e_cid(i, 0) += de; e_cid(i, 1) += de;}

            // Parent edge GID
            if (e_pid(i) != -1) e_pid(i) += de;
        
        });

        // Update Faces
        auto f_gid = Cabana::slice<S_F_GID>(_faces);
        auto f_cid = Cabana::slice<S_F_CID>(_faces);
        auto f_vid = Cabana::slice<S_F_VIDS>(_faces);
        auto f_eid = Cabana::slice<S_F_EIDS>(_faces);
        auto f_pid = Cabana::slice<S_F_PID>(_faces);
        Kokkos::parallel_for("update edge GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_faces),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            f_gid(i) += df;

            // Child face association GIDs
            if (f_cid(i, 0) != -1) {f_cid(i, 0) += df; f_cid(i, 1) += df; f_cid(i, 2) += df; f_cid(i, 3) += df;}

            // Vertex association GIDs
            f_vid(i, 0) += dv; f_vid(i, 1) += dv; f_vid(i, 2) += dv;

            // Edge association GIDs
            f_eid(i, 0) += de; f_eid(i, 1) += de; f_eid(i, 2) += de;

            // Parent face
            if (f_pid(i) != -1) f_pid(i) += df;
        });
    }

    /**
     * Gather mesh connectivity information for vertices within 'dist'
     * edges away
     */
    void gather()
    {

    }

    v_array_type vertices() {return _vertices;}
    e_array_type edges() {return _edges;}
    f_array_type faces() {return _faces;}

    void printVertices()
    {
        auto v_gid = Cabana::slice<S_V_GID>(_vertices);
        auto v_owner = Cabana::slice<S_V_OWNER>(_vertices);
        for (int i = 0; i < (int) _vertices.size(); i++)
        {
            printf("R%d: %d, %d\n", _rank,
                v_gid(i), 
                v_owner(i));
        }
    }
    /**
     * opt: 1 = owned, 2 = ghost, 3 = all
     */
    void printEdges(int opt)
    {
        auto e_vid = Cabana::slice<S_E_VIDS>(_edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(_edges);
        auto e_children = Cabana::slice<S_E_CIDS>(_edges);
        auto e_parent = Cabana::slice<S_E_PID>(_edges);
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(_edges);
        int start = 0, end = _edges.size();
        if (opt == 1) end = _owned_edges;
        else if (opt == 2) start = _owned_edges;
        for (int i = start; i < end; i++)
        {
            printf("%d, v(%d, %d), f(%d, %d), c(%d, %d), p(%d) %d, %d\n",
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1),
                e_fids(i, 0), e_fids(i, 1),
                e_children(i, 0), e_children(i, 1),
                e_parent(i),
                e_owner(i), _rank);
        }
    }

    void printFaces()
    {
        auto f_vgids = Cabana::slice<S_F_VIDS>(_faces);
        auto f_egids = Cabana::slice<S_F_EIDS>(_faces);
        auto f_gid = Cabana::slice<S_F_GID>(_faces);
        auto f_parent = Cabana::slice<S_F_PID>(_faces);
        auto f_child = Cabana::slice<S_F_CID>(_faces);
        auto f_owner = Cabana::slice<S_F_OWNER>(_faces);
        for (int i = 0; i < (int) _faces.size(); i++)
        {
            printf("%d, v(%d, %d, %d), e(%d, %d, %d), c(%d, %d, %d, %d), p(%d), %d\n",
                f_gid(i),
                f_vgids(i, 0), f_vgids(i, 1), f_vgids(i, 2),
                f_egids(i, 0), f_egids(i, 1), f_egids(i, 2),
                f_child(i, 0), f_child(i, 1), f_child(i, 2), f_child(i, 3),
                f_parent(i),
                f_owner(i));
        }
    }

    // Variables
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

    // How many vertices, edges, and faces each process owns
    // Index = rank
    Kokkos::View<int*[3], Kokkos::HostSpace> _vef_gid_start;

    // Version number to keep mesh in sync with other objects. Updates on mesh refinement
    int _version;

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