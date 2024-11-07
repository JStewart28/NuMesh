#ifndef NUMESH_MESH_HPP
#define NUMESH_MESH_HPP

// XXX - Add mapping class.

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

#include <mpi_advance.h>

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
    #define S_F_VIDS 0
    #define S_F_EIDS 1
    #define S_F_GID 2
    #define S_F_PID 3
    #define S_F_CID 4
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
    using face_data = Cabana::MemberTypes<  int[3],    // Vertex global IDs that make up face
                                            int[3],    // Edge global IDs that make up face 
                                            int,       // Face global ID
                                            int,       // Parent face global ID
                                            int,       // Child face global ID                        
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
        MPIX_Info_init(&_xinfo);
        MPIX_Comm_init(&_xcomm, _comm);
        MPIX_Comm_topo_init(_xcomm);

        _version = 0;

        _owned_vertices = 0, _owned_edges = 0, _owned_faces = 0;
        _ghost_vertices = 0, _ghost_edges = 0, _ghost_faces = 0;
    };

    ~Mesh()
    {
        MPIX_Info_free(&_xinfo);
        MPIX_Comm_free(&_xcomm);
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
            f_child(f_lid) = -1;
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
            f_child(f_lid) = -1;
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

        // Get the number of vertices (i.e., array size) for each process for global IDs
        int vef[3] = {ov, oe, of};
        _vef_gid_start = Kokkos::View<int*[3], Kokkos::HostSpace>("_vef_gid_start", _comm_size);

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
    void refine(int p_lfid)
    {
        /*
        using vertex_data = Cabana::MemberTypes<int,       // Vertex global ID                                 
                                                int,       // Owning rank
                                                >;
        using edge_data = Cabana::MemberTypes<  int[2],    // Vertex global ID endpoints of edge    
                                                int[2],    // Face global IDs. The face where it is
                                                           // the lowest edge, starrting at the first
                                                           // vertex and going clockwise, is the first edge.                      
                                                int,       // Edge global ID
                                                int,       // Owning rank
                                                >;
        using face_data = Cabana::MemberTypes<  int[3],    // Vertex global IDs that make up face
                                                int[3],    // Edge global IDs that make up face 
                                                int,       // Face global ID
                                                int,       // Parent face global ID
                                                int,       // Child face global ID                        
                                                int,       // Owning rank
                                                >;
        */
        
        /**
         * Determine if neighboring faces have been refined. If so,
         * use those vertices and edges rather than creating new ones.
         * We must do this to avoid redundant vertices and edges
         */
        int num_new_verts = 0;
        int num_new_edges = 0;
        auto face_tuple = _faces.getTuple(p_lfid);
        int edge_children[3][2];
        int edge_children_verts[3][2];
        for (int i = 0; i < 3; i++)
        {
            int e_lid = Cabana::get<S_F_EIDS>(face_tuple, i) - _vef_gid_start(_rank, 1);
            if (e_lid < 0) throw std::runtime_error("NuMesh::refine: Cannot refine face on boundary" );
            auto edge_tuple = _edges.getTuple(e_lid);
            for (int j = 0; j < 2; j++)
            {
                int child_id = Cabana::get<S_E_CIDS>(edge_tuple, j);
                int vert_id = Cabana::get<S_E_VIDS>(edge_tuple, j);
                if (child_id == -1) {num_new_edges++;}
                edge_children[i][j] = child_id;
                edge_children_verts[i][j] = vert_id;
            }
        }
        num_new_verts = num_new_edges / 2;
        
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

        // Create 9 new edges
        int e_lid_start = _owned_edges;
        _owned_edges += 3 + num_new_edges;
        _edges.resize(_owned_edges);
        auto e_gids = Cabana::slice<S_E_GID>(_edges);
        auto e_vids = Cabana::slice<S_E_VIDS>(_edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(_edges);
        auto e_ranks = Cabana::slice<S_E_OWNER>(_edges);
        auto e_cids = Cabana::slice<S_E_CIDS>(_edges);
        auto e_pid = Cabana::slice<S_E_PID>(_edges);

        // Create 3 new vertices
        int v_lid_start = _owned_vertices;
        _owned_vertices += num_new_verts;
        _vertices.resize(_owned_vertices);
        auto v_gids = Cabana::slice<S_V_GID>(_vertices);
        auto v_ranks = Cabana::slice<S_V_OWNER>(_vertices);

        // Global IDs
        int v_gid_start = _vef_gid_start(_rank, 0);
        int e_gid_start = _vef_gid_start(_rank, 1);
        int f_gid_start = _vef_gid_start(_rank, 2);

        int rank = _rank;
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
                 */
                int e_lid = e_lid_start + i;
                e_gids(e_lid) = e_gid_start + e_lid;
                e_ranks(e_lid) = rank;
                if (i < 2) {e_vids(e_lid, 0) = v_gid; e_vids(e_lid, 1) = v_gid+1;}
                if (i == 2) {e_vids(e_lid, 0) = v_gid-2; e_vids(e_lid, 1) = v_gid;}
            
                /**
                 * New edges 3, 4, and 5 (of 9)
                 *  Edge 3: 0th vertex of parent face to new vertex 0
                 *  Edge 4: 1st vertex of parent face to new vertex 1
                 *  Edge 5: 2nd vertex of parent face to new vertex 2
                 */
                e_lid += 3;
                int parent_vertex = f_vgids(p_lfid, i);
                e_gids(e_lid) = e_gid_start + e_lid;
                e_ranks(e_lid) = rank;
                e_vids(e_lid, 0) = parent_vertex; e_vids(e_lid, 1) = v_gid;

                /**
                 * New edges 6, 7, and 8 (of 9, these are the last three)
                 *  Edge 6: new vertex 0 to 1st vertex of parent face
                 *  Edge 7: new vertex 1 to 2nd vertex of parent face
                 *  Edge 8: new vertex 2 to 0th vertex of parent face
                 */
                e_lid += 3;
                e_gids(e_lid) = e_gid_start + e_lid;
                e_ranks(e_lid) = rank;
                if (i < 2) parent_vertex = f_vgids(p_lfid, i+1);
                if (i == 2) parent_vertex = f_vgids(p_lfid, 0);
                e_vids(e_lid, 0) = v_gid; e_vids(e_lid, 1) = parent_vertex;
            });
            Kokkos::fence();

            Kokkos::parallel_for("new_faces", Kokkos::RangePolicy<execution_space>(0, 4),
            KOKKOS_LAMBDA(int i) {

                // Populate these first; they are easy
                int f_lid = f_lid_start + i;
                int f_gid = f_gid_start + f_lid;
                int f_gid_parent = p_lfid + f_gid_start;
                f_gids(f_lid) = f_gid;          // Global ID
                f_ranks(f_lid) = rank;          // Owning rank
                f_pgids(f_lid) = f_gid_parent;  // Parent face
                f_cgids(f_lid) = -1;            // No children

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
                    v2 = v1 = v_gids(v_lid_start+1);
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
            });
        }
    }

    /**
     * Refine all faces specified in the fids vector 
     * Calling this function increments the version of the mesh
     * and updates global IDS
     */
    void batchRefine(std::vector<int> fids)
    {
        _version++;
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
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(_edges);
        int start = 0, end = _edges.size();
        if (opt == 1) end = _owned_edges;
        else if (opt == 2) start = _owned_edges;
        for (int i = start; i < end; i++)
        {
            printf("%d, v(%d, %d), f(%d, %d), %d, %d\n",
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1),
                e_fids(i, 0), e_fids(i, 1),
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
            printf("%d, v(%d, %d, %d), e(%d, %d, %d), %d\n",
                f_gid(i),
                f_vgids(i, 0), f_vgids(i, 1), f_vgids(i, 2),
                f_egids(i, 0), f_egids(i, 1), f_egids(i, 2), 
                f_owner(i));
        }
    }

    // Variables
  private:
    MPI_Comm _comm;
    int _rank, _comm_size;

    MPIX_Comm* _xcomm;
    MPIX_Info* _xinfo;

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