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
    #define F_GID 2
    #define F_PID 3
    #define F_LAYER 4
    #define F_OWNER 5
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
     * After the vertex and edge connectivity has been set and
     * global IDs assigned, create faces from the edges and vertices.
     *  1. Create the faces
     *  2. Update the edges to reflect which faces they are part of
     *  3. Gather face global IDs from remotely owned faces to the edges
     */
    void finializeInit()
    {
        createFaces();
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
        
        if (!isPerfectSquare(_comm_size))
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
    void _refineFaces(View fgids)
    {        
        const int rank = _rank, comm_size = _comm_size;
        const int num_face_refinements = fgids.extent(0);

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
                // If a face has already been refined (i.e. has children), don't refine it again
                if (f_cid_slice0(f_lid, 0) == -1) return;

                int index = Kokkos::atomic_fetch_add(&face_counter(), 1);
                local_face_lids(index) = f_lid;

                // Each face refinement adds 3 new edges
                Kokkos::atomic_add(&edge_counter(), 3);
                for (int j = 0; j < 3; j++)
                {
                    int ex_gid = f_eid_slice0(f_lid, j);
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
                        remote_edge_slice(idx) = f_eid_slice0(f_lid, j);
                        remote_edge_rank_slice(idx) = rank;

                        // Find the remote rank this edge belongs to
                        int export_rank = -1;
                        for (int r = 0; r < comm_size; r++)
                        {
                            if (r == rank) continue;
                            if (ex_gid >= _vef_gid_start_d(r, 1))
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

            int elid = distributor_edges_import_slice(i) - _vef_gid_start_d(rank, 1);
            
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

        // printf("R%d old FGID space: %d to %d\n", rank, _vef_gid_start(_rank, 2), _vef_gid_start(_rank, 2)+_owned_faces);


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
        auto tmp = Kokkos::create_mirror_view(vef_gid_start_old_d);
        Kokkos::deep_copy(tmp, _vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_old_d, tmp);
        updateGlobalIDs();
        
        // printf("R%d new VEF: %d, %d, %d\n", rank, new_vertices, new_edges, face_refinements*4);
        // printf("R%d new VGID space: %d to %d\n", rank, _vef_gid_start(_rank, 0), _vef_gid_start(_rank, 0)+_owned_vertices);
        // printf("R%d new EGID space: %d to %d\n", rank, _vef_gid_start(_rank, 1), _vef_gid_start(_rank, 1)+_owned_edges);
        // printf("R%d new FGID space: %d to %d\n", rank, _vef_gid_start(_rank, 2), _vef_gid_start(_rank, 2)+_owned_faces);
        
        // Copy new _vef_gid_start to device
        auto tmp1 = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(tmp1, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, tmp1);

        /******************************
         * Populate new vertices
         *****************************/
        Kokkos::parallel_for("populate new vertices", Kokkos::RangePolicy<execution_space>(0, new_vertices),
            KOKKOS_LAMBDA(int i) {
            
            int v_l = v_new_lid_start + i;
            int v_g = v_l + _vef_gid_start_d(rank, 0);
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
            int ec_gid0 = _vef_gid_start_d(rank, 1) + ec_lid0;
            int ec_gid1 = _vef_gid_start_d(rank, 1) + ec_lid1;
            int ec_layer = e_layer(i) + 1;

            // if (rank == 1) printf("R%d refining edge %d: new edges %d, %d (offset %d)\n", rank, i+_vef_gid_start_d(rank, 1), ec_gid0, ec_gid1, offset);

            // Global IDs = global ID start + local ID
            e_gid(ec_lid0) = ec_gid0;
            e_gid(ec_lid1) = ec_gid1;

            // Set parent edges for the split edges
            e_pid(ec_lid0) = i + _vef_gid_start_d(rank, 1);
            e_pid(ec_lid1) = i + _vef_gid_start_d(rank, 1);

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
            int new_vgid = _vef_gid_start_d(rank, 0) + v_new_lid_start + offset;
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
            updateGlobalID(Edge(), &distributor_edges_import_slice(i), _vef_gid_start_d, vef_gid_start_old_d);

            // Parent edge LID
            int elid = get_lid(e_gid, distributor_edges_import_slice(i), 0, owned_edges);
            assert(elid != -1);
            halo_export_ids(idx) = elid;

            // First child edge LID
            int clid = e_cid(elid, 0) - _vef_gid_start_d(rank, 1);
            halo_export_ids(idx+1) = clid;

            // Second child edge LID
            clid = e_cid(elid, 1) - _vef_gid_start_d(rank, 1);
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

        // Update lambda capture variables
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
            //     face_id+_vef_gid_start_d(rank, 2),
            //     f_eid(face_id, 0), f_eid(face_id, 1), f_eid(face_id, 2),
            //     e_new_lid_start+offset, e_new_lid_start+offset+1, e_new_lid_start+offset+2, offset);
            for (int j = offset; j < offset+3; j++)
            {
                e_lid = e_new_lid_start + j;

                // Global IDs = global ID start + local ID
                e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;

                // if (rank == 2) printf("R%d face %d (edges %d, %d, %d), adding edge %d (offset %d)\n", rank,
                // face_id+_vef_gid_start_d(rank, 2),
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
                    int lid = egid - _vef_gid_start_d(rank, 1);
                    elid[k] = get_lid(e_gid, egid, 0, num_edges);
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
                // elid0 = f_eid(face_id, 0)-_vef_gid_start_d(rank, 1); elid1 = f_eid(face_id, 1)-_vef_gid_start_d(rank, 1); elid2 = f_eid(face_id, 2)-_vef_gid_start_d(rank, 1);
                // nv0 = e_vid(elid0, 2); nv1 = e_vid(elid1, 2); nv2 = e_vid(elid2, 2); 
                // printf("R%d: nv: %d, %d, %d\n", rank, nv0, nv1, nv2);
                if (j == offset) {e_vid(e_lid, 0) = nv0; e_vid(e_lid, 1) = nv1;}
                else if (j == offset+1) {e_vid(e_lid, 0) = nv1; e_vid(e_lid, 1) = nv2;}
                else if (j == offset+2) {e_vid(e_lid, 0) =nv0; e_vid(e_lid, 1) = nv2;}

                // Middle vertex not set until new edges are further refined
                e_vid(e_lid, 2) = -1;
                // printf("R%d: ne%d: (%d, %d, %d)\n", rank, e_lid+_vef_gid_start_d(rank, 1), e_vid(e_lid, 0), e_vid(e_lid, 1), e_vid(e_lid, 2));
            
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
            int parent_face_gid = parent_face_lid + _vef_gid_start_d(rank, 2);
            int layer = f_layer(parent_face_lid) + 1;
            int offset = Kokkos::atomic_fetch_add(&face_counter(), 4);
            int new_face_lid;
            int new_face_gid;

            // Start with values similar to all new faces; can be looped
            for (int j = offset; j < offset+4; j++)
            {
                new_face_lid = f_new_lid_start + j;
                new_face_gid = new_face_lid + _vef_gid_start_d(rank, 2);

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
            }

            /*****************************
             * Edges, in general:
             *  Edge 0: V0 -> V1
             *  Edge 1: V1 -> V2
             *  Edge 2: V2 -> V0
             ****************************/
            // Vertex global IDs and edge local IDs
            int vg0, vg1, vg2, el0, el1, el2;
            int num_edges = owned_edges + ghost_edges; // Edge AoSoA size

            /******************************
             * Face 0 verts:
             *  V0: Parent face e0, v0
             *  V1: Parent face e0, middle vert
             *  V2: Parent face e2, middle vert
             *****************************/
            new_face_lid = f_new_lid_start + offset;
            el0 = get_lid(e_gid, f_eid(parent_face_lid, 0), 0, num_edges);
            assert(el0 != -1);
            vg0 = e_vid(el0, 0);
            vg1 = e_vid(el0, 2);
            el2 = get_lid(e_gid, f_eid(parent_face_lid, 2), 0, num_edges);
            assert(el2 != -1);
            vg2 = e_vid(el2, 2);
            el0 = find_edge(e_vid, e_new_lid_start, num_edges, vg0, vg1);
            el1 = find_edge(e_vid, e_new_lid_start, num_edges, vg1, vg2);
            el2 = find_edge(e_vid, e_new_lid_start, num_edges, vg2, vg0);
            f_eid(new_face_lid, 0) = e_gid(el0); f_eid(new_face_lid, 1) = e_gid(el1); f_eid(new_face_lid, 2) = e_gid(el2); 
            if (f_gid(new_face_lid) == 296)
            {
                printf("Face 0: FGID: %d, v(%d, %d, %d), e(%d, %d, %d)\n", f_gid(new_face_lid),
                    vg0, vg1, vg2, f_eid(parent_face_lid, 0), f_eid(parent_face_lid, 1), f_eid(parent_face_lid, 2));
            }

            /******************************
             * Face 1 verts:
             *  V0: Parent face e0, middle vert
             *  V1: Parent face e1, middle vert
             *  V2: Parent face e0, v1
             *****************************/
            new_face_lid = f_new_lid_start + offset + 1;
            el0 = get_lid(e_gid, f_eid(parent_face_lid, 0), 0, num_edges);
            assert(el0 != -1);
            vg0 = e_vid(el0, 2);
            el1 = get_lid(e_gid, f_eid(parent_face_lid, 1), 0, num_edges);
            assert(el1 != -1);
            vg1 = e_vid(el1, 2);
            vg2 = e_vid(el0, 1);
            el0 = find_edge(e_vid, e_new_lid_start, num_edges, vg0, vg1);
            el1 = find_edge(e_vid, e_new_lid_start, num_edges, vg1, vg2);
            el2 = find_edge(e_vid, e_new_lid_start, num_edges, vg2, vg0);
            f_eid(new_face_lid, 0) = e_gid(el0); f_eid(new_face_lid, 1) = e_gid(el1); f_eid(new_face_lid, 2) = e_gid(el2); 
            if (f_gid(new_face_lid) == 296)
            {
                printf("Face 1: FGID: %d, v(%d, %d, %d), e(%d, %d, %d)\n", f_gid(new_face_lid),
                    vg0, vg1, vg2, f_eid(parent_face_lid, 0), f_eid(parent_face_lid, 1), f_eid(parent_face_lid, 2));
            }

            /******************************
             * Face 2 verts:
             *  V0: Parent face e2, middle vert
             *  V1: Parent face e1, middle vert
             *  V2: Parent face e2, v1
             *****************************/
            new_face_lid = f_new_lid_start + offset + 2;
            el2 = get_lid(e_gid, f_eid(parent_face_lid, 2), 0, num_edges);
            assert(el2 != -1);
            vg0 = e_vid(el2, 2);
            el1 = get_lid(e_gid, f_eid(parent_face_lid, 1), 0, num_edges);
            assert(el1 != -1);
            vg1 = e_vid(el1, 2);
            vg2 = e_vid(el2, 1);
            // printf("Verts: %d, %d, %d\n", vg0, vg1, vg2);
            el0 = find_edge(e_vid, e_new_lid_start, num_edges, vg0, vg1);
            el1 = find_edge(e_vid, e_new_lid_start, num_edges, vg1, vg2);
            el2 = find_edge(e_vid, e_new_lid_start, num_edges, vg2, vg0);
            f_eid(new_face_lid, 0) = e_gid(el0); f_eid(new_face_lid, 1) = e_gid(el1); f_eid(new_face_lid, 2) = e_gid(el2); 
            if (f_gid(new_face_lid) == 296)
            {
                printf("Face 2: FGID: %d, v(%d, %d, %d), e(%d, %d, %d)\n", f_gid(new_face_lid),
                    vg0, vg1, vg2, f_eid(parent_face_lid, 0), f_eid(parent_face_lid, 1), f_eid(parent_face_lid, 2));
            }


            /******************************
             * Face 3 verts:
             *  V0: Parent face e0, middle vert
             *  V1: Parent face e1, middle vert
             *  V2: Parent face e2, middle vert
             *****************************/
            new_face_lid = f_new_lid_start + offset + 3;
            el0 = get_lid(e_gid, f_eid(parent_face_lid, 0), 0, num_edges);
            assert(el0 != -1);
            vg0 = e_vid(el0, 2);
            el1 = get_lid(e_gid, f_eid(parent_face_lid, 1), 0, num_edges);
            assert(el1 != -1);
            vg1 = e_vid(el1, 2);
            el2 = get_lid(e_gid, f_eid(parent_face_lid, 2), 0, num_edges);
            assert(el2 != -1);
            vg2 = e_vid(el2, 2);
            // printf("Verts: %d, %d, %d\n", vg0, vg1, vg2);
            el0 = find_edge(e_vid, e_new_lid_start, num_edges, vg0, vg1);
            el1 = find_edge(e_vid, e_new_lid_start, num_edges, vg1, vg2);
            el2 = find_edge(e_vid, e_new_lid_start, num_edges, vg2, vg0);
            f_eid(new_face_lid, 0) = e_gid(el0); f_eid(new_face_lid, 1) = e_gid(el1); f_eid(new_face_lid, 2) = e_gid(el2); 
            if (f_gid(new_face_lid) == 296)
            {
                printf("Face 3: FGID: %d, v(%d, %d, %d), e(%d, %d, %d)\n", f_gid(new_face_lid),
                    vg0, vg1, vg2, f_eid(parent_face_lid, 0), f_eid(parent_face_lid, 1), f_eid(parent_face_lid, 2));
            }
        });
        // if (rank == 0) printFaces(0, 0);
        // printFaces(1, 345);

        // if (rank == 0)
        // {
        //     printf("***************AFTER***************\n");
        //     printEdges(3, 0);
        //     // 162, 163
        //     // verts 25, -1; -1, 26
        // }

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
        _refineFaces(fgids);
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

        // Copy vef views to device
        Kokkos::View<int*[3], MemorySpace> vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp1 = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp1, _vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp1);
        Kokkos::View<int*[3], MemorySpace> old_vef_start_d("old_vef_start_d", _comm_size);
        auto hv_tmp2 = Kokkos::create_mirror_view(old_vef_start_d);
        Kokkos::deep_copy(hv_tmp2, old_vef_start);
        Kokkos::deep_copy(old_vef_start_d, hv_tmp2);

        // Update vertices
        auto v_gid = Cabana::slice<V_GID>(_vertices);
        Kokkos::parallel_for("update vertex GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_vertices),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            updateGlobalID(Vertex(), &v_gid(i), vef_gid_start_d, old_vef_start_d);
            
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
            updateGlobalID(Edge(), &e_gid(i), vef_gid_start_d, old_vef_start_d);

            // Vertex association GIDs
            for (int j = 0; j < 3; j++)
            {
                updateGlobalID(Vertex(), &e_vid(i, j), vef_gid_start_d, old_vef_start_d);
            }
            
            // Child edge GIDs
            for (int j = 0; j < 2; j++)
            {
                updateGlobalID(Edge(), &e_cid(i, j), vef_gid_start_d, old_vef_start_d);
            }

            // Parent edge GID
            updateGlobalID(Edge(), &e_pid(i), vef_gid_start_d, old_vef_start_d);
        });

        // Update Faces
        auto f_gid = Cabana::slice<F_GID>(_faces);
        auto f_cid = Cabana::slice<F_CID>(_faces);
        auto f_eid = Cabana::slice<F_EIDS>(_faces);
        auto f_pid = Cabana::slice<F_PID>(_faces);
        Kokkos::parallel_for("update face GIDs", Kokkos::RangePolicy<execution_space>(0, _owned_faces),
            KOKKOS_LAMBDA(int i) {
            
            // Global ID
            updateGlobalID(Face(), &f_gid(i), vef_gid_start_d, old_vef_start_d);

            // Child face association GIDs
            for (int j = 0; j < 4; j++)
            {   
                updateGlobalID(Face(), &f_cid(i, j), vef_gid_start_d, old_vef_start_d);
            }

            // Edge association GIDs
            for (int j = 0; j < 3; j++)
            {
                updateGlobalID(Edge(), &f_eid(i, j), vef_gid_start_d, old_vef_start_d);
            }

            // Parent face
            updateGlobalID(Face(), &f_pid(i), vef_gid_start_d, old_vef_start_d);
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

    int count(Own, Vertex) {return _owned_vertices;}
    int count(Own, Edge) {return _owned_edges;}
    int count(Own, Face) {return _owned_faces;}

    MPI_Comm comm() {return _comm;}
    int version() {return _version;}

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
            printf("i%d, %d, v(%d, %d, %d), c(%d, %d), p(%d), L%d, %d, %d\n", i,
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
            
            printf("i%d, %d, v(%d, %d, %d), c(%d, %d), p(%d), L%d, %d, %d\n", i,
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
                printf("R%d: i%d, f%d, e(%d, %d, %d), c(%d, %d, %d, %d), p(%d), L%d, %d\n", _rank, i,
                    f_gid(i),
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
                printf("R%d: i%d, f%d, e(%d, %d, %d), c(%d, %d, %d, %d), p(%d), L%d, %d\n", _rank, i,
                    f_gid(i),
                    f_egids(i, 0), f_egids(i, 1), f_egids(i, 2),
                    f_child(i, 0), f_child(i, 1), f_child(i, 2), f_child(i, 3),
                    f_parent(i),
                    f_layer(i),
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
struct isnumesh_mesh : std::false_type {};
template <typename ExecutionSpace, typename MemSpace>
struct isnumesh_mesh<NuMesh::Mesh<ExecutionSpace, MemSpace>> : std::true_type {};

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