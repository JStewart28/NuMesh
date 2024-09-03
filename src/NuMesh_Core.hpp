#ifndef NUMESH_CORE_HPP
#define NUMESH_CORE_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

#include <mpi_advance.h>

#include <NuMesh_Communicator.hpp>

#include <limits>

#ifndef AOSOA_SLICE_INDICES
#define AOSOA_SLICE_INDICES 1
#endif

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
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;
    using mesh_type = Cabana::Grid::UniformMesh<double, 2>;

    using Node = Cabana::Grid::Node;
    using l2g_type = Cabana::Grid::IndexConversion::L2G<mesh_type, Node>;
    using node_view = Kokkos::View<double***, device_type>;

    using halo_type = Cabana::Grid::Halo<MemorySpace>;

    // Note: Larger types should be listed first
    using vertex_data = Cabana::MemberTypes<double[3], // xyz position in space                           
                                            int,       // Vertex global ID                                 
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
                                            // Constants for slice indices
                                            #if AOSOA_SLICE_INDICES
                                            #define S_V_XYZ 0 
                                            #define S_V_GID 1
                                            #define S_V_OWNER 2
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
    // XXX Change the final parameter of particle_array_type, vector type, to
    // be aligned with the machine we are using
    using v_array_type = Cabana::AoSoA<vertex_data, device_type, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, device_type, 4>;
    using f_array_type = Cabana::AoSoA<face_data, device_type, 4>;
    using size_type = typename memory_space::size_type;

    // Construct a mesh.
    Mesh( const std::array<double, 2>& global_low_corner,
            const std::array<double, 2>& global_high_corner,
            const std::array<int, 2>& num_nodes,
            const std::array<bool, 2>& periodic,
            const Cabana::Grid::BlockPartitioner<2>& partitioner,
            MPI_Comm comm )
            : _global_low_corner( global_low_corner)
            , _global_high_corner( global_high_corner )
            , _global_num_cell( num_nodes )
            , _periodic( periodic )
            , _comm ( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        //_communicator = std::make_shared(Communicator<execution_space, memory_space>(_comm));

        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            _global_low_corner, _global_high_corner, _global_num_cell);
        auto global_grid = Cabana::Grid::createGlobalGrid(
            _comm, global_mesh, _periodic, partitioner );
        int halo_width = 0;
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        // Get the topology
        _topology = Kokkos::View<int*[2], Kokkos::HostSpace>("topology", _comm_size);
        int cart_coords[2] = {-1, -1};
        MPI_Cart_coords(global_grid->comm(), _rank, 2, cart_coords);
        MPI_Allgather(cart_coords, 2, MPI_INT, _topology.data(), 2, MPI_INT, global_grid->comm());

        _ghost_edges = 0;
    };

    /**
     * Update the array that stores the global index starting values for each process
     */
    void update_vef_counts()
    {
        // Get the number of vertices (i.e., array size) for each process for global IDs
        int vef[3] = {_owned_vertices, _owned_edges, _owned_faces};
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
    }

    void initialize_from_grid()
    {
        auto own_nodes = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                    Cabana::Grid::Local() );
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *_local_grid );
        l2g_type local_L2G = Cabana::Grid::IndexConversion::createL2G<Cabana::Grid::UniformMesh<double, 2>, Cabana::Grid::Node>(*_local_grid, Cabana::Grid::Node());


        auto node_triple_layout =
                Cabana::Grid::createArrayLayout( _local_grid, 3, Cabana::Grid::Node() );

        // The actual arrays storing mesh quantities
        // 1. The spatial positions of the interface
        auto position = Cabana::Grid::createArray<double, memory_space>(
                "position", node_triple_layout );
        Cabana::Grid::ArrayOp::assign( *position, 0.0, Cabana::Grid::Ghost() );

        double dx = (_global_high_corner[0] - _global_low_corner[0]) / _global_num_cell[0];
        double dy = (_global_high_corner[1] - _global_low_corner[1]) / _global_num_cell[1]; 
        double p = 0.25;
        auto z = position->view();

        /* Step 1: Initialize mesh values in a grid format */
        auto policy = Cabana::Grid::createExecutionPolicy(own_nodes, execution_space());
        Kokkos::parallel_for("Initialize Cells", policy,
            KOKKOS_LAMBDA( const int i, const int j ) {
                int index[2] = { i, j };
                double coords[2];
                local_mesh.coordinates( Cabana::Grid::Node(), index, coords);
                
                double z1 = dx * coords[0];
                double z2 = dy * coords[1];
                double z3 = 0.25 * cos(z1 * (2 * M_PI / p)) * cos(z2 * (2 * M_PI / p));
                double za[3] = {z1, z2, z3};

                for (int d = 0; d < 3; d++)
                {
                    z(i, j, d) = za[d];
                }
            });
        
        /* Step 2: Iterate over the 2D array to populate AoSoA of vertices */
        auto local_space = _local_grid->indexSpace(Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local());

        int istart = local_space.min(0), jstart = local_space.min(1);
        int iend = local_space.max(0), jend = local_space.max(1);

        // Create the AoSoA
        _owned_vertices = (iend - istart) * (jend - jstart);
        _owned_edges = _owned_vertices * 3;
        _owned_faces = _owned_vertices * 2;
        _v_array.resize(_owned_vertices);
        _e_array.resize(_owned_edges);
        _f_array.resize(_owned_faces*2);
        update_vef_counts();

        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);

        // We should convert the following loops to a Cabana::simd_parallel_for at some point to get better write behavior

        // Initialize the vertices, edges, and faces
        auto v_xyz = Cabana::slice<S_V_XYZ>(_v_array);
        auto v_gid = Cabana::slice<S_V_GID>(_v_array);
        auto v_owner = Cabana::slice<S_V_OWNER>(_v_array);

        auto e_vid = Cabana::slice<S_E_VIDS>(_e_array); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<S_E_GID>(_e_array);
        auto e_fids = Cabana::slice<S_E_FIDS>(_e_array);
        auto e_owner = Cabana::slice<S_E_OWNER>(_e_array);
        int rank = _rank;
        auto topology = Cabana::Grid::getTopology( *_local_grid );
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
        Kokkos::parallel_for("populate_ve", Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<2>>({{istart, jstart}}, {{iend, jend}}),
        KOKKOS_LAMBDA(int i, int j) {

            // Initialize vertices
            int v_lid = (i - istart) * (jend - jstart) + (j - jstart);
            int v_gid_ = _vef_gid_start_d(rank, 0) + v_lid;
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
                v_gid_other = _vef_gid_start_d(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                //printf("R%d: e_gid: %d, e_lid: %d, v_lid: %d\n", rank, _vef_gid_start_d(rank, 1) + e_lid, e_lid, v_lid);
                e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;

                // Edge 1: northeast
                v_gid_other = _vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3 + 1;
                e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;

                // Edge 2: east
                v_gid_other = _vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;
                //printf("ij: %d, %d: e3: vid: %d, vo: %d\n", i, j, v_lid, v_lid_other);
            }
            // Boundary edges on east boundary
            else if ((i == iend-1) && (j < jend-1))
            {   
                // Edge 0: north
                v_gid_other = _vef_gid_start_d(rank, 0) + (i - istart) * (jend - jstart) + (j+1 - jstart);
                e_lid = v_lid * 3;
                e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;

                // Edges 1 and 2
                neighbor_rank = device_topology[5];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 1
                    offset = v_lid % (jend-jstart);
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0) + offset + 1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;

                    // Edge 2
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                }
            }
            // Boundary edges on north boundary
            else if ((j == jend-1) && (i < iend-1))
            {
                // Edge 2: east
                v_gid_other = _vef_gid_start_d(rank, 0) + (i+1 - istart) * (jend - jstart) + (j - jstart);
                e_lid = v_lid * 3 + 2;
                e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                e_owner(e_lid) = rank;

                // Edges 0 and 1
                neighbor_rank = device_topology[7];
                if (neighbor_rank == -1) 
                {
                    // Free boundary
                    v_gid_other = -1;
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;

                    v_gid_other = -1;
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                } 
                else 
                {
                    // Periodic or MPI boundary
                    // Edge 0
                    offset = v_lid / (iend-istart);
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    //printf("e_gid: %d, v0: %d, v1: %d, offset: %d\n", e_gid(e_lid), v_gid_, v_gid_other, offset);
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;

                    // Edge 1
                    offset = v_lid / (iend-istart);
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0) + (offset+1) * (iend-istart);
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
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
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                } 
                else 
                {
                    offset = v_lid / (iend-istart);
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0) + offset * (iend-istart);
                    e_lid = v_lid * 3;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
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
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                } 
                else 
                {
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0);
                    e_lid = v_lid * 3 + 1;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
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
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = -1; e_vid(e_lid, 1) = -1;
                    e_owner(e_lid) = rank;
                } 
                else 
                {
                    offset = v_lid % (jend-jstart);
                    v_gid_other = _vef_gid_start_d(neighbor_rank, 0) + offset;
                    e_lid = v_lid * 3 + 2;
                    e_gid(e_lid) = _vef_gid_start_d(rank, 1) + e_lid;
                    e_vid(e_lid, 0) = v_gid_; e_vid(e_lid, 1) = v_gid_other;
                    e_owner(e_lid) = rank;
                }
            }

            e_fids(e_lid, 0) = -1; e_fids(e_lid, 1) = -1;
        });
        Kokkos::fence();
        //printView(local_L2G, _rank, z, 1, 1, 1);
        //printVertices();
        //printEdges();
    }

    /**
     * Create faces from given vertices and edges
     * Each vertex is associated with at least one face
     */
    void initialize_faces()
    {
        // size_type num_vertices = _v_array.size();
        // printf("Num verts: %d\n", num_vertices);

        /* Each vertex contributes 2 faces */
        _f_array.resize(_owned_faces);
        update_vef_counts();
    
        auto v_xyz = Cabana::slice<S_V_XYZ>(_v_array);
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
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("Initialize faces", Kokkos::RangePolicy<execution_space>(0, _v_array.size()), KOKKOS_LAMBDA(int i) {
            // Face 1: "left" face; face 2: "right" face   
            int f_lid;

            // Find face 1 values
            // Get the three vertices and edges for face1
            int v_gid0, v_lid0, v_gid1, v_lid1, v_gid2, v_lid2;
            int e_gid0, e_lid0, e_gid1, e_lid1, e_gid2, e_lid2;
            v_gid0 = v_gid(i); v_lid0 = v_gid0 - _vef_gid_start_d(rank, 0);
            // Follow first edge to get next vertex
            e_gid0 = v_gid0*3; e_lid0 = e_gid0 - _vef_gid_start_d(rank, 1);
            v_gid1 = e_vid(e_lid0, 1); v_lid1 = v_gid1 - _vef_gid_start_d(rank, 0);
            // Use second vertex to get next edge
            e_gid1 = v_gid1*3+2; e_lid1 = e_gid1 - _vef_gid_start_d(rank, 1);
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - _vef_gid_start_d(rank, 1);
            v_gid2 = e_vid(e_lid2, 1); v_lid2 = _vef_gid_start_d(rank, 0);
            
            // Populate face 1 values
            f_lid = i*2;
            f_vgids(f_lid, 0) = v_gid0; f_vgids(f_lid, 1) = v_gid1; f_vgids(f_lid, 2) = v_gid2;
            f_egids(f_lid, 0) = e_gid0; f_egids(f_lid, 1) = e_gid1; f_egids(f_lid, 2) = e_gid2;
            f_gid(f_lid) = f_lid + _vef_gid_start_d(rank, 2);
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
            v_gid1 = e_vid(e_lid0, 1); v_lid1 = v_gid1 - _vef_gid_start_d(rank, 0);
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - _vef_gid_start_d(rank, 1);
            v_gid2 = e_vid(e_lid2, 1); v_lid2 = _vef_gid_start_d(rank, 0);
            // DIFFERENT: Use vertex 2 to get edge 1. Edge 1 the first edge of vertex 2
            e_gid1 = v_gid2*3; e_lid1 = e_gid1 - _vef_gid_start_d(rank, 1);

            // Populate face 2 values
            f_lid = i*2+1;
            f_vgids(f_lid, 0) = v_gid0; f_vgids(f_lid, 1) = v_gid1; f_vgids(f_lid, 2) = v_gid2;
            f_egids(f_lid, 0) = e_gid0; f_egids(f_lid, 1) = e_gid1; f_egids(f_lid, 2) = e_gid2;
            f_gid(f_lid) = f_lid + _vef_gid_start_d(rank, 2);
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

    /**
     * After faces have been created, map each edge to its two faces.
     * The face where the edge is the lowest numbered edge, starting
     * at the first vertex and moving clockwise, is the first edge.
     */
    void assign_edges_to_faces()
    {
        update_vef_counts();

        auto e_vid = Cabana::slice<S_E_VIDS>(_e_array);
        auto e_gid = Cabana::slice<S_E_GID>(_e_array);
        auto e_fids = Cabana::slice<S_E_FIDS>(_e_array);
        auto e_owner = Cabana::slice<S_E_OWNER>(_e_array);

        auto f_vgids = Cabana::slice<S_F_VIDS>(_f_array);
        auto f_egids = Cabana::slice<S_F_EIDS>(_f_array);
        auto f_gid = Cabana::slice<S_F_GID>(_f_array);
        auto f_parent = Cabana::slice<S_F_PID>(_f_array);
        auto f_child = Cabana::slice<S_F_CID>(_f_array);
        auto f_owner = Cabana::slice<S_F_OWNER>(_f_array);

        /* Edges will always be one of the following for faces:
         * - 1st edge and 2nd edge
         * - 1st edge and 3rd edge
         * - 2nd edge and 3rd edge
         * 
         * Iterate over faces and assign 1st and 2nd edges' face 1
         */
        int rank = _rank, comm_size = _comm_size;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("assign_edges13_to_faces", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int f_gid, eX_lid;
            f_gid = f_lid + _vef_gid_start_d(rank, 2);

            // Where this edge is the first edge, set its face1
            eX_lid = f_egids(f_lid, 0) - _vef_gid_start_d(rank, 1);
            e_fids(eX_lid, 0) = f_gid;

            // Where this edge is the third edge, set its face2
            eX_lid = f_egids(f_lid, 2) - _vef_gid_start_d(rank, 1);
            e_fids(eX_lid, 1) = f_gid;
        });

        /* Temporary naive solution to store which edges are needed from which processes: 
         * Create a (comm_size x num_faces) view.
         * If an edge is needed from another process, set (owner_rank, f_lid) to the 
         * global edge ID needed from owner_rank.
         */ 
        // Set a counter to count number of ranks that will have a message sent to it
        using CounterView = Kokkos::View<int, device_type, Kokkos::MemoryTraits<Kokkos::Atomic>>;
        CounterView counter("counter");
        Kokkos::deep_copy(counter, 0);
        Kokkos::View<int**, device_type> sendvals_unpacked("sendvals_unpacked", _comm_size, _f_array.size());
        Kokkos::deep_copy(sendvals_unpacked, -1);
        // Step 2: Iterate over second edges. Populate Face ID that is not filled
        int debug_rank = DEBUG_RANK;
        Kokkos::parallel_for("find_needed_edge2", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int f_gid, e2_gid, from_rank = -1;
            f_gid = f_lid + _vef_gid_start_d(rank, 2);

            // Where this edge is the first edge, set its face1
            e2_gid = f_egids(f_lid, 1);

            // If e2_gid < (rank GID start) or (> (rank+1) GID start), 
            // this edge is owned by another process 
            if ((rank != comm_size-1) && ((e2_gid < _vef_gid_start_d(rank, 1)) || (e2_gid >= _vef_gid_start_d(rank+1, 1))))
            {
                if (e2_gid < _vef_gid_start_d(0, 1)) from_rank = 0;
                else if (e2_gid >= _vef_gid_start_d(comm_size-1, 1)) from_rank = comm_size-1;
                else
                {
                    for (int r = 0; r < comm_size-1; r++)
                    {
                        if (r == rank) continue;
                        //printf("checking btw R%d: [%d, %d)\n", r, _vef_gid_start_d(r, 1))
                        if ((e2_gid >= _vef_gid_start_d(r, 1)) && (e2_gid < _vef_gid_start_d(r+1, 1))) from_rank = r;
                    }
                }
                sendvals_unpacked(from_rank, f_lid) = e2_gid;
                counter()++;
                // if (rank == debug_rank) printf("R%d: sendvals_unpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvals_unpacked(from_rank, f_lid));

            }
            // If rank == comsize-1 we need a seperate condition
            else if (rank == comm_size-1)
            {
                if (e2_gid < _vef_gid_start_d(rank, 1))
                {
                    for (int r = 0; r < rank; r++)
                    {
                        //printf("checking btw R%d: [%d, %d)\n", r, _vef_gid_start_d(r, 1))
                        if ((e2_gid >= _vef_gid_start_d(r, 1)) && (e2_gid < _vef_gid_start_d(r+1, 1)))
                        {
                            from_rank = r;
                            sendvals_unpacked(from_rank, f_lid) = e2_gid;
                            counter()++;
                            // if (rank == debug_rank) printf("R%d: sendvals_unpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvals_unpacked(from_rank, f_lid));
                        }
                    }
                }
            }

            // if (e_fids(e2_lid, 0) != -1) e_fids(e2_lid, 0) = f_gid;
            // else if (e_fids(e2_lid, 1) != -1) e_fids(e2_lid, 1) = f_gid;
        });
        Kokkos::fence();
        int len_sendvals = -1;
        Kokkos::deep_copy(len_sendvals, counter);

        // Reset counter
        Kokkos::deep_copy(counter, 0);

        // Send each process the edges it needs
        // Step 1: Count the number of edges needed from each other process
        Kokkos::View<int*, device_type> sendcounts_unpacked("sendcounts_unpacked", _comm_size);   
        // Parallel reduction to count non -1 values in each row
        int num_cols = (int) _f_array.size();
        Kokkos::parallel_for("count_sends", Kokkos::RangePolicy<execution_space>(0, _comm_size), KOKKOS_LAMBDA(const int i) {
            int count = 0;
            for (int j = 0; j < num_cols; j++) {
                if (sendvals_unpacked(i, j) != -1) {
                    count++;
                }
            }
            if (count > 0) counter()++;
            sendcounts_unpacked(i) = count;
            //if (rank == debug_rank) printf("R%d sendcounts_unpacked(%d): %d\n", rank, i, sendcounts_unpacked(i));
        });
        int send_nnz = 0;
        Kokkos::deep_copy(send_nnz, counter);
        // Pack serially for now, for simplicity. XXX - pack on GPU
        auto sendcounts_unpacked_h = Kokkos::create_mirror_view(sendcounts_unpacked);
        Kokkos::deep_copy(sendcounts_unpacked_h, sendcounts_unpacked);
        auto sendvals_unpacked_h = Kokkos::create_mirror_view(sendvals_unpacked);
        Kokkos::deep_copy(sendvals_unpacked_h, sendvals_unpacked);
        int* sendcounts = new int[send_nnz];
        int* dest = new int[send_nnz];
        int* sdispls = new int[send_nnz];
        int* sendvals = new int[len_sendvals];
        int idx = 0;
        sdispls[0] = 0;
        for (int i = 0; i < sendcounts_unpacked_h.extent(0); i++)
        {
            if (sendcounts_unpacked_h(i)) 
            {
                sendcounts[idx] = sendcounts_unpacked_h(i);
                dest[idx] = i; 
                if ((idx < send_nnz) && (idx > 0)) sdispls[idx] = sdispls[idx-1] + sendcounts[idx];
                // if (rank == DEBUG_RANK) printf("R%d: sendcounts[%d]: %d, dest[%d]: %d, sdispls[%d]: %d\n", rank,
                //     idx, sendcounts[idx], idx, dest[idx], idx, sdispls[idx]);
                idx++;
            }
        }

        // Pack sendvals
        // Pack starting at lowest rank in dest
        idx = 0;
        for (int d = 0; d < send_nnz; d++)
        {
            int dest_rank = dest[d];
            for (int e_lid = 0; e_lid < _f_array.size(); e_lid++)
            {
                if (sendvals_unpacked_h(dest_rank, e_lid) != -1)
                {
                    sendvals[idx] = sendvals_unpacked_h(dest_rank, e_lid);
                    //if (rank == DEBUG_RANK) printf("R%d: sendvals[%d]: %d\n", rank, idx, sendvals[idx]);
                    idx++;
                }
            }
        }

        int recv_nnz, recv_size;
        // XXX - Assuming a there are as many recieved messages as sent messages
        int *rdispls = new int[send_nnz];
        int *src = new int[send_nnz];
        int *recvcounts = new int[send_nnz];
        int *recvvals = new int [len_sendvals];

        MPIX_Comm* xcomm;
        MPIX_Info* xinfo;
        MPIX_Info_init(&xinfo);
        MPIX_Comm_init(&xcomm, MPI_COMM_WORLD);

        MPIX_Alltoallv_crs(send_nnz, len_sendvals, dest, sendcounts, sdispls, MPI_INT, sendvals, 
            &recv_nnz, &recv_size, src, recvcounts, rdispls, MPI_INT, recvvals, xinfo, xcomm);
        // printf("R%d: s_nnz, len: (%d, %d), dest: (%d, %d), sc: (%d, %d), sdispls: (%d, %d), svals: (%d, %d, %d, %d, %d, %d, %d, %d)\n",
        //     rank, send_nnz, len_sendvals, dest[0], dest[1], sendcounts[0], sendcounts[1],
        //     sdispls[0], sdispls[1], sendvals[0], sendvals[1], sendvals[2], sendvals[3], sendvals[4], sendvals[5], sendvals[6], sendvals[7]);
        // printf("R%d: r_nnz, len: (%d, %d), src: (%d, %d), rc: (%d, %d), rdispls: (%d, %d), rvals: (%d, %d, %d, %d, %d, %d, %d, %d)\n",
        //     rank, recv_nnz, recv_size, src[0], src[1], recvcounts[0], recvcounts[1],
        //     rdispls[0], rdispls[1], recvvals[0], recvvals[1], recvvals[2], recvvals[3], recvvals[4], recvvals[5], recvvals[6], recvvals[7]);

        MPIX_Info_free(&xinfo);
        MPIX_Comm_free(&xcomm);

        // Parse the recieved values to send the correct edges to the processes that requested them.
        Kokkos::View<Cabana::Tuple<edge_data>*, Kokkos::HostSpace> send_edges(Kokkos::ViewAllocateWithoutInitializing("send_edges"), recv_size);
        Kokkos::View<Cabana::Tuple<edge_data>*, Kokkos::HostSpace> recv_edges(Kokkos::ViewAllocateWithoutInitializing("recv_edges"), len_sendvals);
        MPI_Request* requests = new MPI_Request[send_nnz+recv_nnz];
        Cabana::Tuple<edge_data> e_tuple;
        std::pair<std::size_t, std::size_t> range = { 0, 0 };
        for (int s = 0; s < recv_nnz; s++)
        {
            int send_to = src[s], send_count = recvcounts[s], displs = rdispls[s];
            int counter = displs;
            while (counter < (displs+send_count))
            {
                // if (rank == DEBUG_RANK) printf("R%d: send_to: %d, accessing recvvals[%d]: %d\n", rank, send_to, counter, recvvals[counter]);
                int e_gid = recvvals[counter];
                int e_lid = e_gid - _vef_gid_start(_rank, 1);
                
                // Get this edge from the edge AoSoA and set it in the buffer
                e_tuple = _e_array.getTuple(e_lid);
                send_edges(counter) = e_tuple;
                counter++;
            }
            // Post this send
            // MPI_Isend(send_buffer.data(), num_elements, MPI_INT, 1, tag, MPI_COMM_WORLD, &requests[0]);
            range.first = displs; range.second = range.first + send_count;
            auto send_subview = Kokkos::subview(send_edges, range);
            if (_rank == DEBUG_RANK)
            {
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(0)));
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(1)));
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(2)));
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(3)));

            }
            MPI_Isend(send_subview.data(), sizeof(e_tuple)*send_subview.size(), MPI_BYTE, send_to, _rank, _comm, &requests[s]);
        }

        // Post receives
        for (int s = 0; s < send_nnz; s++)
        {
            int recv_from = dest[s], recv_count = sendcounts[s], displs = sdispls[s];
            range.first = displs; range.second = range.first + recv_count;
            auto recv_subview = Kokkos::subview(recv_edges, range);
            MPI_Irecv(recv_subview.data(), sizeof(e_tuple)*recv_subview.size(), MPI_BYTE, recv_from, recv_from, _comm, &requests[recv_nnz+s]);
        }

        MPI_Waitall(send_nnz+recv_nnz, requests, MPI_STATUSES_IGNORE);
        delete[] requests;

        if (_rank == 0)
        {
            for (int i = 0; i < recv_edges.extent(0); i++)
            {
                e_tuple = recv_edges(i);
                //if (rank == 0) printf("R%d: got e_gid: %d\n", _rank, Cabana::get<S_E_GID>(e_tuple));
            }
        }

        // Put the ghosted edges in the local edge aosoa
        _ghost_edges = recv_edges.extent(0);
        _e_array.resize(_owned_edges+_ghost_edges);
        int owned_edges = _owned_edges;
        e_array_type e_array = _e_array;
        Kokkos::parallel_for("add_ghosted_edges", Kokkos::RangePolicy<execution_space>(0, _ghost_edges), KOKKOS_LAMBDA(const int i) {
            int e_lid = i+owned_edges;
            e_array.setTuple(e_lid, recv_edges(i));
        });

        // Finally, assign the ghosted edges to their faces
        /* Edges will always be one of the following for faces:
         * - Its 1st edge and 2nd edge
         * - Its 1st edge and 3rd edge
         * - Its 2nd edge and 3rd edge
         * 
         * Iterate over faces and assign 1st and 2nd edges' face 1
         */
        auto e_gid_slice = Cabana::slice<S_E_GID>(_e_array);
        auto e_fid_slice = Cabana::slice<S_E_FIDS>(_e_array);
        int ghosted_edges = _ghost_edges;
        //printEdges();
        Kokkos::parallel_for("find_needed_edge2", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int f_gid, e2_gid, e2_lid;
            f_gid = f_lid + _vef_gid_start_d(rank, 2);
            e2_gid = f_egids(f_lid, 1);
            e2_lid = e2_gid - _vef_gid_start_d(rank, 1);
            //if (rank == 2) printf("R%d f_gid %d, e2_gid %d, e min/max: (%d, %d)\n", rank, f_gid, e2_gid, _vef_gid_start_d(rank, 1),_vef_gid_start_d(rank+1, 1));

            // Check if edge owned locally
            if ((e2_lid >= 0) && (e2_lid < owned_edges))
            {
                //if (rank == 2) printf("R%d: acessing e_fids(%d)\n", rank, e2_lid);
                // Check if the first face is set; then this f_gid is the second face
                if (e_fid_slice(e2_lid, 0) != -1) e_fid_slice(e2_lid, 1) = f_gid;
                // Otherwise check if the second face is set; then this f_gid is the first face
                else if (e_fid_slice(e2_lid, 1) != -1) e_fid_slice(e2_lid, 0) = f_gid;
            }
            else
            {
                //printf("R%d: e_gid not owned locally: %d\n", rank, e2_gid);
                // Find the local_id of edge to set its face id(s)
                for (e2_lid = owned_edges; e2_lid < owned_edges+ghosted_edges; e2_lid++)
                {
                    int e2_gid_owned = e_gid_slice(e2_lid);
                    if (e2_gid_owned == e2_gid)
                    {
                        //printf("R%d: ghosted e2_lid %d = e_gid %d\n", rank, e2_lid, e2_gid);
                        if (e_fid_slice(e2_lid, 0) != -1) e_fid_slice(e2_lid, 1) = f_gid;
                        // Otherwise check if the second face is set; then this f_gid is the first face
                        else if (e_fid_slice(e2_lid, 1) != -1) e_fid_slice(e2_lid, 0) = f_gid;
                    }
                }
            }
        });
        // Any edge-face mappings that are -1 at this point are for faces not owned by the process

        printEdges();
        delete[] dest;
        delete[] sendcounts;
        delete[] sdispls;
        delete[] sendvals;

        delete[] src;
        delete[] recvcounts;
        delete[] rdispls;
        delete[] recvvals;
    }

    void printVertices()
    {
        auto v_xyz = Cabana::slice<S_V_XYZ>(_v_array);
        auto v_gid = Cabana::slice<S_V_GID>(_v_array);
        auto v_owner = Cabana::slice<S_V_OWNER>(_v_array);
        for (int i = 0; i < _v_array.size(); i++)
        {
            printf("R%d: [%d, (%0.3lf, %0.3lf, %0.3lf), %d]\n", _rank,
                v_gid(i), 
                v_xyz(i, 1), v_xyz(i, 0), v_xyz(i, 2),
                v_owner(i));
        }
    }

    void printEdges()
    {
        auto e_vid = Cabana::slice<S_E_VIDS>(_e_array);
        auto e_fids = Cabana::slice<S_E_FIDS>(_e_array);
        auto e_gid = Cabana::slice<S_E_GID>(_e_array);
        auto e_owner = Cabana::slice<S_E_OWNER>(_e_array);
        for (int i = 0; i < _e_array.size(); i++)
        {
            printf("%d, v(%d, %d), f(%d, %d), %d\n",
                e_gid(i),
                e_vid(i, 0), e_vid(i, 1),
                e_fids(i, 0), e_fids(i, 1),
                e_owner(i));
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
        for (int i = 0; i < _f_array.size(); i++)
        {
            printf("%d, v(%d, %d, %d), e(%d, %d, %d), %d\n",
                f_gid(i),
                f_vgids(i, 0), f_vgids(i, 1), f_vgids(i, 2),
                f_egids(i, 0), f_egids(i, 1), f_egids(i, 2), 
                f_owner(i));
        }
    }

    private:
        std::array<double, 2> _global_low_corner, _global_high_corner;
        std::array<int, 2> _global_num_cell;
        const std::array<bool, 2> _periodic;
        std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;
        MPI_Comm _comm;
        std::shared_ptr<Communicator<execution_space, memory_space>> _communicator;

        int _rank, _comm_size;

        // AoSoAs for the mesh
        v_array_type _v_array;
        e_array_type _e_array;
        f_array_type _f_array;
        int _owned_vertices, _owned_edges, _owned_faces, _ghost_edges;

        // Topology
        Kokkos::View<int*[2], Kokkos::HostSpace> _topology;
        int _neighbors[8]; // Starting from bottom left and going clockwise

        // How many vertices, edges, and faces each proces owns
        // Index = rank
        Kokkos::View<int*[3], Kokkos::HostSpace> _vef_gid_start;
        

};
//---------------------------------------------------------------------------//

} // end namespace NuMesh


#endif // NUMESH_CORE_HPP