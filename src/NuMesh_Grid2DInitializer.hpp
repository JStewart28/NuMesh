#ifndef NUMESH_GRID2DINITIALIZER_HPP
#define NUMESH_GRID2DINITIALIZER_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

#include <NuMesh_Mesh.hpp>

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

template <class ExecutionSpace, class MemorySpace>
class Grid2DInitializer
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

    Grid2DInitializer( const std::array<double, 2>& global_low_corner,
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

        auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
            _global_low_corner, _global_high_corner, _global_num_cell);
        auto global_grid = Cabana::Grid::createGlobalGrid(
            _comm, global_mesh, _periodic, partitioner );
        int halo_width = 0;
        _local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

        // Get the topology
        // _topology = Kokkos::View<int*[2], Kokkos::HostSpace>("topology", _comm_size);
        // int cart_coords[2] = {-1, -1};
        // MPI_Cart_coords(global_grid->comm(), _rank, 2, cart_coords);
        // MPI_Allgather(cart_coords, 2, MPI_INT, _topology.data(), 2, MPI_INT, global_grid->comm());
    }

    ~Grid2DInitializer() {}

    // XXX - Add position initialization functor?
    template <class v_array_type, class e_array_type>
    void from_grid(v_array_type *v_array, e_array_type *e_array, int *owned_vertices, int *owned_edges, int *owned_faces)
    {
        auto own_nodes = _local_grid->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Node(),
                                                    Cabana::Grid::Local() );
        auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( *_local_grid );
        // l2g_type local_L2G = Cabana::Grid::IndexConversion::createL2G<Cabana::Grid::UniformMesh<double, 2>, Cabana::Grid::Node>(*_local_grid, Cabana::Grid::Node());


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
        int ov = (iend - istart) * (jend - jstart);
        int oe = ov * 3;
        int of =  ov * 2;
        v_array->resize(ov);
        e_array->resize(oe);
        *owned_vertices = ov; *owned_edges = oe; *owned_faces = of;

        // XXX - Can we avoid copying this code from NuMesh_Core.hpp?
        // Get the number of vertices (i.e., array size) for each process for global IDs
        int vef[3] = {ov, oe, of};
        Kokkos::View<int*[3], Kokkos::HostSpace> vef_gid_start("vef_gid_start", _comm_size);

        MPI_Allgather(vef, 3, MPI_INT, vef_gid_start.data(), 3, MPI_INT, _comm);
    
        // Find where each process starts its global IDs
        for (int i = 1; i < _comm_size; ++i) {
            vef_gid_start(i, 0) += vef_gid_start(i - 1, 0);
            vef_gid_start(i, 1) += vef_gid_start(i - 1, 1);
            vef_gid_start(i, 2) += vef_gid_start(i - 1, 2);
        }
        for (int i = _comm_size - 1; i > 0; --i) {
            vef_gid_start(i, 0) = vef_gid_start(i - 1, 0);
            vef_gid_start(i, 1) = vef_gid_start(i - 1, 1);
            vef_gid_start(i, 2) = vef_gid_start(i - 1, 2);
        }
        vef_gid_start(0, 0) = 0;
        vef_gid_start(0, 1) = 0;
        vef_gid_start(0, 2) = 0;

        // Copy vef_gid_start to device
        Kokkos::View<int*[3], device_type> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        // We should convert the following loops to a Cabana::simd_parallel_for at some point to get better write behavior

        // Initialize the vertices, edges, and faces
        auto v_xyz = Cabana::slice<S_V_XYZ>(*v_array);
        auto v_gid = Cabana::slice<S_V_GID>(*v_array);
        auto v_owner = Cabana::slice<S_V_OWNER>(*v_array);

        auto e_vid = Cabana::slice<S_E_VIDS>(*e_array); // VIDs from south to north, west to east vertices
        auto e_gid = Cabana::slice<S_E_GID>(*e_array);
        auto e_fids = Cabana::slice<S_E_FIDS>(*e_array);
        auto e_owner = Cabana::slice<S_E_OWNER>(*e_array);
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
    }

  private:
    std::array<double, 2> _global_low_corner, _global_high_corner;
    std::array<int, 2> _global_num_cell;
    const std::array<bool, 2> _periodic;
    std::shared_ptr<Cabana::Grid::LocalGrid<mesh_type>> _local_grid;
    MPI_Comm _comm;

    int _rank, _comm_size;

};

/**
 *  Return a shared pointer to a createGrid2DInitializer object
 */
template <class ExecutionSpace, class MemorySpace>
auto createGrid2DInitializer( const std::array<double, 2>& global_low_corner,
            const std::array<double, 2>& global_high_corner,
            const std::array<int, 2>& num_nodes,
            const std::array<bool, 2>& periodic,
            const Cabana::Grid::BlockPartitioner<2>& partitioner,
            MPI_Comm comm )
{
    return std::make_shared<Grid2DInitializer<ExecutionSpace, MemorySpace>>(
        global_low_corner, global_high_corner, num_nodes, periodic, partitioner, comm);
}

} // end namespace NUMesh


#endif // NUMESH_GRID2DINITIALIZER_HPP