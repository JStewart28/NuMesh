#ifndef NUMESH_MESH_HPP
#define NUMESH_MESH_HPP

// XXX - Add mapping class.

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include <mpi.h>

// #include <mpi_advance.h>

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
    #define F_VIDS 1
    #define F_EIDS 2
    #define F_GID 3
    #define F_PID 4
    #define F_LAYER 5
    #define F_OWNER 6
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
                                            int[3],    // Vertex global IDs that make up face
                                            int[3],    // Edge global IDs that make up face 
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
    
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        auto v_owner = Cabana::slice<V_OWNER>(_vertices);

        auto e_vid = Cabana::slice<E_VIDS>(_edges); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_owner = Cabana::slice<E_OWNER>(_edges);

        auto f_vgids = Cabana::slice<F_VIDS>(_faces);
        auto f_egids = Cabana::slice<F_EIDS>(_faces);
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_parent = Cabana::slice<F_PID>(_faces);
        auto f_child = Cabana::slice<F_CID>(_faces);
        auto f_owner = Cabana::slice<F_OWNER>(_faces);
        auto f_layer = Cabana::slice<F_LAYER>(_faces);

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
     * Assign edges to locally owned faces. Must be completed before gathering edges to 
     * ghosted edges have face global IDs of faces not owned by the remote process.
     */
    void updateEdges()
    {
        // auto e_vid = Cabana::slice<E_VIDS>(_edges);
        // auto e_gid = Cabana::slice<E_GID>(_edges);
        // auto e_owner = Cabana::slice<E_OWNER>(_edges);

        // auto f_egids = Cabana::slice<F_EIDS>(_faces);

        // /* Edges will always be one of the following for faces:
        //  * - 1st edge and 2nd edge
        //  * - 1st edge and 3rd edge
        //  * - 2nd edge and 3rd edge
        //  * 
        //  * Iterate over faces and assign 1st and 3rd edges' face 1
        //  */
        // int rank = _rank;
        // // Copy _vef_gid_start to device
        // Kokkos::View<int*[3], MemorySpace> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        // auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        // Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        // Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        // Kokkos::parallel_for("assign_edges13_to_faces", Kokkos::RangePolicy<execution_space>(0, _faces.size()), KOKKOS_LAMBDA(int f_lid) {
        //     int f_gid, eX_lid; // eX_gid;
        //     f_gid = f_lid + _vef_gid_start_d(rank, 2);

        //     // Where this edge is the first edge, set its face1
        //     // eX_gid = f_egids(f_lid, 0);
        //     //if (eX_gid == 14) printf("R%d: e1_gid: %d, f_gid: %d\n", rank, eX_gid, f_gid);
        //     eX_lid = f_egids(f_lid, 0) - _vef_gid_start_d(rank, 1);
        //     e_fids(eX_lid, 0) = f_gid;
            
        //     // Where this edge is the third edge, set its face2
        //     // eX_gid = f_egids(f_lid, 2);
        //     //if (eX_gid == 14) printf("R%d: e3_gid: %d, f_gid: %d\n", rank, eX_gid, f_gid);
        //     eX_lid = f_egids(f_lid, 2) - _vef_gid_start_d(rank, 1);
        //     e_fids(eX_lid, 1) = f_gid;
        // });
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
    //     Kokkos::View<int**, device_type> sendvalunpacked("sendvalunpacked", _comm_size, _faces.size());
    //     Kokkos::deep_copy(sendvalunpacked, -1);
    //     int rank = _rank, comm_size = _comm_size;
    //     // Copy _vef_gid_start to device
    //     Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
    //     auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
    //     Kokkos::deep_copy(hv_tmp, _vef_gid_start);
    //     Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
    //     auto f_egids = Cabana::slice<F_EIDS>(_faces);
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
    //             sendvalunpacked(from_rank, f_lid) = e2_gid;
    //             counter()++;
    //             //if (rank == 1) printf("R%d: sendvalunpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvalunpacked(from_rank, f_lid));

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
    //                         sendvalunpacked(from_rank, f_lid) = e2_gid;
    //                         counter()++;
    //                         //if (rank == 1) printf("R%d: sendvalunpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvalunpacked(from_rank, f_lid));
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
        

        // Initialize values that are shared for all edges
        Kokkos::parallel_for("init edge shared values", Kokkos::RangePolicy<execution_space>(0, _edges.size()),
            KOKKOS_LAMBDA(int i) {

            e_vid(i, 2) = -1; // No edge has been split
            e_pid(i) = -1;
            e_cids(i, 0) = -1; e_cids(i, 1) = -1;
            e_owner(i) = rank;
            e_layer(i) = 0;
        });

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
                //printf("R%d: e_gid: %d, e_lid: %d, v_lid: %d\n", rank, vef_gid_start_d(rank, 1) + e_lid, e_lid, v_lid);
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                

                // Edge 1: northeast
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3 + 1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                // Edge 2: east
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                //printf("ij: %d, %d: e3: vid: %d, vo: %d\n", i, j, v_lid, v_lid_other);
            }
            // Boundary edges on east boundary
            else if ((i == iend-1) && (j < jend-1))
            {   
                // Edge 0: north
                v_gid_other = vef_gid_start_d(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                // Edges 1 and 2
                neighbor_rank = device_topology[5];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 1
                    offset = v_lid % (jend-jstart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset + 1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                    // Edge 2
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                }
            }
            // Boundary edges on north boundary
            else if ((j == jend-1) && (i < iend-1))
            {
                // Edge 2: east
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                // Edges 0 and 1
                neighbor_rank = device_topology[7];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 0
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;

                    // Edge 1
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + (offset+1) * (iend-istart);
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
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
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
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
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0);
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
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
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                } 
                else 
                {
                    offset = v_lid % (jend-jstart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
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
     * Part one for refining a face. Splits all edges on faces
     * marked to be refined and adds new interior edges for
     * faces to be refined.
     * 
     * Split into a seperate function for ease of testing
     * 
     * @param fid: Kokkos view holding the global IDs
     *             of faces to be refined
     * 
     */
    template <class View>
    void _refineAndAddEdges(View fgids)
    {        
        const int rank = _rank, comm_size = _comm_size;
        const int num_face_refinements = fgids.extent(0);

        /********************************************************
         * Phase 1.0: Collect all edges that need to be refined,
         * then refine them in parallel
         *******************************************************/

        using counter_vec = Kokkos::View<int*, memory_space>;
        using counter_int = Kokkos::View<int, memory_space>;

        /**
         * List of locally-owned face IDs this process needs to split
         * Since we don't know the size a priori, make it as large as the
         * total number of face refinements.
         */
        // 
        counter_vec local_face_lids("local_face_lids", num_face_refinements);

        /**
         * edge_needrefine:
         *      values > 0: This edge is owned and can be refined locally
         * 
         * remote_edge_needrefine:
         *      Holds GIDs of remote edges that must be refined.
         *      XXX - currently assumes remote edges encompass no more than 1/5 of owned edges
         * 
         * element_export_ranks:
         *      Maps to remote_edge_needrefine. Holds the destination rank of the remote
         *          edge that must be refined.
         *      XXX - currently assumes remote edges encompass no more than 1/5 of owned edges
         */
        counter_vec edge_needrefine("edge_needrefine", _owned_edges);
        int remote_edge_needrefine_size = _owned_edges/5;
        counter_vec remote_edge_needrefine("remote_edge_needrefine", remote_edge_needrefine_size);
        counter_vec element_export_ranks("element_export_ranks", remote_edge_needrefine_size);


        counter_int vert_counter("vert_counter");
        counter_int edge_counter("edge_counter");
        counter_int face_counter("face_counter");
        counter_int remote_edge_counter("remote_edge_counter");
        Kokkos::deep_copy(edge_needrefine, 0);
        Kokkos::deep_copy(vert_counter, 0);
        Kokkos::deep_copy(edge_counter, 0);
        Kokkos::deep_copy(face_counter, 0);
        Kokkos::deep_copy(remote_edge_counter, 0);
        Kokkos::deep_copy(remote_edge_needrefine, -1);
        auto f_egid = Cabana::slice<F_EIDS>(_faces);
        int owned_edges = _owned_edges;
        int owned_faces = _owned_faces;

        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], MemorySpace> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("populate edge_needrefine", Kokkos::RangePolicy<execution_space>(0, num_face_refinements),
            KOKKOS_LAMBDA(int i) {
            
            int f_lid = fgids(i) - _vef_gid_start_d(rank, 2);
            if ((f_lid >= 0) && (f_lid < owned_faces)) // Make sure we own the face
            {
                int index = Kokkos::atomic_fetch_add(&face_counter(), 1);
                local_face_lids(index) = f_lid;

                // Each face refinement adds 3 new edges
                Kokkos::atomic_add(&edge_counter(), 3);
                for (int j = 0; j < 3; j++)
                {
                    int ex_gid = f_egid(f_lid, j);
                    int ex_lid = ex_gid - _vef_gid_start_d(rank, 1);
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
                        remote_edge_needrefine(idx) = f_egid(f_lid, j);

                        // Find the remote rank this edge belongs to
                        int export_rank = -1;
                        for (int r = 0; r < comm_size-1; r++)
                        {
                            if (r == rank) continue;

                            // Special case for highest rank because can't index into r+1
                            if (r == comm_size - 1)
                            {
                                if (ex_gid >= _vef_gid_start_d(r, 1))
                                {
                                    export_rank = r;
                                }
                            }
                            else if ((ex_gid >= _vef_gid_start_d(r, 1)) && (ex_gid < _vef_gid_start_d(r+1, 1)))
                            {
                                export_rank = r;
                            }
                        }
                        element_export_ranks(idx) = export_rank;
                    }
                }
            }
        });
        int remote_edges;
        Kokkos::deep_copy(remote_edges, remote_edge_counter);


        /********************************************************************************
         * Communicate remote edges that must be refined. We know how we are sending to
         * but not who we are receiving from.
         * 
         * We have:
         *  - remote_edge_needrefine: Kokkos view of length remote_edges that holds GIDs
         *      remote edges that must be refined
         *  - element_export_ranks: Maps to remote_edge_needrefine. Kokkos view of length
         *      remote_edges that holds the destination (owner) rank of each remote edge
         *      that must be refined.
         * 
         * We need:
         *  - 
         *******************************************************************************/
        
        auto distributor = Cabana::Distributor<memory_space>( _comm, element_export_ranks );

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
        _refineAndAddEdges(fgids);
        // _gatherEdges()
    }
    
    void updateGlobalIDs()
    {
        // Update mesh version
        _version++;

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
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        Kokkos::parallel_for("update vertex GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_vertices),
            KOKKOS_LAMBDA(int i) { v_gid(i) += dv; });

        // Update edges
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_vid = Cabana::slice<E_VIDS>(_edges);
        auto e_cid = Cabana::slice<E_CIDS>(_edges);
        auto e_pid = Cabana::slice<E_PID>(_edges);
        Kokkos::parallel_for("update edge GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_edges),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            e_gid(i) += de;

            // Vertex association GIDs
            e_vid(i, 0) += dv; e_vid(i, 1) += dv;
            if (e_vid(i, 2) != -1) e_vid(i, 2) += dv;

            // Child edge GIDs
            if (e_cid(i, 0) != -1) {e_cid(i, 0) += de; e_cid(i, 1) += de;}

            // Parent edge GID
            if (e_pid(i) != -1) e_pid(i) += de;
        
        });

        // Update Faces
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_cid = Cabana::slice<F_CID>(_faces);
        auto f_vid = Cabana::slice<F_VIDS>(_faces);
        auto f_eid = Cabana::slice<F_EIDS>(_faces);
        auto f_pid = Cabana::slice<F_PID>(_faces);
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

    auto get_vef_gid_start() {return _vef_gid_start;}

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
     * opt: 1 = specific edge, 2 = owned, 3 = ghost
     */
    void printEdges(int opt, int egid)
    {
        auto e_vid = Cabana::slice<E_VIDS>(_edges);
        auto e_children = Cabana::slice<E_CIDS>(_edges);
        auto e_parent = Cabana::slice<E_PID>(_edges);
        auto e_gid = Cabana::slice<E_GID>(_edges);
        auto e_owner = Cabana::slice<E_OWNER>(_edges);
        int rank = _rank;
        if (opt == 1)
        {
            int e_gid_start = _vef_gid_start(_rank, 1);
            int owned_edges = _owned_edges;
            Kokkos::parallel_for("print edges", Kokkos::RangePolicy<execution_space>(0, 1),
            KOKKOS_LAMBDA(int x) {
            
            int elid = egid - e_gid_start;
            if ((elid >= 0) && (elid < owned_edges))
            {
            int i = elid;
            printf("R%d: e%d, v(%d, %d, %d), c(%d, %d), p(%d) %d, %d\n", rank,
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1), e_vid(i, 2),
                e_children(i, 0), e_children(i, 1),
                e_parent(i),
                e_owner(i), rank);
            }
            });
            return;
        }
        int start = 0, end = _edges.size();
        if (opt == 2) end = _owned_edges;
        else if (opt == 3) start = _owned_edges;
        Kokkos::parallel_for("print edges", Kokkos::RangePolicy<execution_space>(start, end),
            KOKKOS_LAMBDA(int i) {
            
            printf("%d, v(%d, %d, %d), c(%d, %d), p(%d) %d, %d\n",
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1), e_vid(i, 2),
                e_children(i, 0), e_children(i, 1),
                e_parent(i),
                e_owner(i), rank);

        });
    }

    void printFaces(int opt, int fgid)
    {
        auto f_vgids = Cabana::slice<F_VIDS>(_faces);
        auto f_egids = Cabana::slice<F_EIDS>(_faces);
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_parent = Cabana::slice<F_PID>(_faces);
        auto f_child = Cabana::slice<F_CID>(_faces);
        auto f_owner = Cabana::slice<F_OWNER>(_faces);
        if (opt == 0)
        {
            // Print all faces
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
        else if (opt == 1)
        {
            // Print one face
            int flid = fgid - _vef_gid_start(_rank, 2);
            if ((flid >= 0) && (flid < _owned_faces))
            {
                int i = flid;
                printf("R%d: f%d, v(%d, %d, %d), e(%d, %d, %d), c(%d, %d, %d, %d), p(%d), %d\n", _rank,
                    f_gid(fgid),
                    f_vgids(i, 0), f_vgids(i, 1), f_vgids(i, 2),
                    f_egids(i, 0), f_egids(i, 1), f_egids(i, 2),
                    f_child(i, 0), f_child(i, 1), f_child(i, 2), f_child(i, 3),
                    f_parent(i),
                    f_owner(i));
            }
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
struct inumesh_mesh : std::false_type {};
template <typename ExecutionSpace, typename MemSpace>
struct inumesh_mesh<NuMesh::Mesh<ExecutionSpace, MemSpace>> : std::true_type {};

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