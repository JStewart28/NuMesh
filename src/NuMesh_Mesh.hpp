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
    #define S_E_GID 2
    #define S_E_OWNER 3
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
    using node_view = Kokkos::View<double***, device_type>;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    // Note: Larger types should be listed first
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
     * Turn a 2D array into an unstructured mesh of vertices, edges, and faces
     * by turning each (i, j) index into a vertex and creating edges between 
     * all eight neighbor vertices in the grid
     */
    template <class ExecutionSpace, class MemorySpace, class CabanaArray>
    auto initializeFromArray( CabanaArray& array )
    {
        static_assert( is_array<CabanaArray>::value, "NuMesh::Mesh::initializeFromArray: Cabana::Grid::Array required" );
        auto local_grid = array.layout()->localGrid()
        auto node_space = local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                Cabana::Grid::Local() );
        
        /* Iterate over the 2D position array to populate AoSoAs in the unstructured mesh*/
        int istart = node_space.min(0), jstart = node_space.min(1);
        int iend = node_space.max(0), jend = node_space.max(1);

        // Create the AoSoA
        int ov = (iend - istart) * (jend - jstart);
        int oe = ov * 3;
        int of =  ov * 2;
        _vertices->resize(ov);
        _edges->resize(oe);
        _faces->resize(of);
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
        auto v_gid = Cabana::slice<S_V_GID>(*vertices);
        auto v_owner = Cabana::slice<S_V_OWNER>(*vertices);

        auto e_vid = Cabana::slice<S_E_VIDS>(*edges); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<S_E_GID>(*edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(*edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(*edges);
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
            for (int dim = 0; dim < 3; dim++) {
                v_xyz(v_lid, dim) = z(i, j, dim);
            }

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

                // Edge 1: northeast
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3 + 1;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;

                // Edge 2: east
                v_gid_other = vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
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

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
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

                    // Edge 2
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
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

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
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

                    // Edge 1
                    offset = v_lid / (iend-istart);
                    v_gid_other = vef_gid_start_d(neighbor_rank, 0) + (offset+1) * (iend-istart);
                    e_lid = v_lid * 3 + 1;
                    e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
                    e_gid(e_lid) = vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
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
                }
            }
        });
        Kokkos::fence();

        printEdges(3);
        // initialize_edges();
        // initialize_faces();
    }

    /**
     * Gather mesh connectivity information for vertices within 'dist'
     * edges away
     */
    void gather()
    {

    }

    v_array_type vertices() {return _vertices};
    e_array_type edges() {return _edges};
    f_array_type faces() {return _faces};

    void printVertices()
    {
        auto v_xyz = Cabana::slice<S_V_XYZ>(_v_array);
        auto v_gid = Cabana::slice<S_V_GID>(_v_array);
        auto v_owner = Cabana::slice<S_V_OWNER>(_v_array);
        for (int i = 0; i < (int) _v_array.size(); i++)
        {
            printf("R%d: [%d, (%0.3lf, %0.3lf, %0.3lf), %d]\n", _rank,
                v_gid(i), 
                v_xyz(i, 1), v_xyz(i, 0), v_xyz(i, 2),
                v_owner(i));
        }
    }
    /**
     * opt: 1 = owned, 2 = ghost, 3 = all
     */
    void printEdges(int opt)
    {
        auto e_vid = Cabana::slice<S_E_VIDS>(_e_array);
        auto e_fids = Cabana::slice<S_E_FIDS>(_e_array);
        auto e_gid = Cabana::slice<S_E_GID>(_e_array);
        auto e_owner = Cabana::slice<S_E_OWNER>(_e_array);
        int start = 0, end = _e_array.size();
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
        auto f_vgids = Cabana::slice<S_F_VIDS>(_f_array);
        auto f_egids = Cabana::slice<S_F_EIDS>(_f_array);
        auto f_gid = Cabana::slice<S_F_GID>(_f_array);
        auto f_parent = Cabana::slice<S_F_PID>(_f_array);
        auto f_child = Cabana::slice<S_F_CID>(_f_array);
        auto f_owner = Cabana::slice<S_F_OWNER>(_f_array);
        for (int i = 0; i < (int) _f_array.size(); i++)
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

  // Functions
  private:
    /**
     * Assign edges to locally owned faces. Must be completed before gathering edges to 
     * ghosted edges have face global IDs of faces not owned by the remote process.
     */
    void initialize_edges()
    {
        auto e_vid = Cabana::slice<S_E_VIDS>(_edges);
        auto e_gid = Cabana::slice<S_E_GID>(_edges);
        auto e_fids = Cabana::slice<S_E_FIDS>(_edges);
        auto e_owner = Cabana::slice<S_E_OWNER>(_edges);

        /* Edges will always be one of the following for faces:
         * - 1st edge and 2nd edge
         * - 1st edge and 3rd edge
         * - 2nd edge and 3rd edge
         * 
         * Iterate over faces and assign 1st and 3rd edges' face 1
         */
        int rank = _rank;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("assign_edges13_to_faces", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
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
     * Create faces from vertices and edges
     * Each vertex is associated with at least one face
     */
    void initialize_faces()
    {
        /* Each vertex contributes 2 faces */
        _faces.resize(_owned_faces);
    
        auto v_gid = Cabana::slice<S_V_GID>(_v_array);
        auto v_owner = Cabana::slice<S_V_OWNER>(_v_array);

        auto e_vid = Cabana::slice<S_E_VIDS>(_e_array); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<S_E_GID>(_e_array);
        auto e_owner = Cabana::slice<S_E_OWNER>(_e_array);

        auto f_vgids = Cabana::slice<S_F_VIDS>(_f_array);
        auto f_egids = Cabana::slice<S_F_EIDS>(_f_array);
        auto f_gid = Cabana::slice<S_F_GID>(_f_array);
        auto f_parent = Cabana::slice<S_F_PID>(_f_array);
        auto f_child = Cabana::slice<S_F_CID>(_f_array);
        auto f_owner = Cabana::slice<S_F_OWNER>(_f_array);

        int rank = _rank;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("Initialize faces", Kokkos::RangePolicy<execution_space>(0, _v_array.size()), KOKKOS_LAMBDA(int i) {
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
        //printFaces();
    }
        

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