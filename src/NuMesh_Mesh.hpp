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

#include <NuMesh_Communicator.hpp>
#include <NuMesh_Grid2DInitializer.hpp>
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
                                            
    // XXX Change the final parameter of particle_array_type, vector type, to
    // be aligned with the machine we are using
    using v_array_type = Cabana::AoSoA<vertex_data, device_type, 4>;
    using e_array_type = Cabana::AoSoA<edge_data, device_type, 4>;
    using f_array_type = Cabana::AoSoA<face_data, device_type, 4>;
    using size_type = typename memory_space::size_type;

    // Construct a mesh.
    Mesh( MPI_Comm comm )
        : _comm ( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
        _version = 0;

        _owned_vertices = -1, _owned_edges = -1, _owned_faces = -1;
        _ghost_vertices = 0, _ghost_edges = 0, _ghost_faces = 0;
        _communicator = createCommunicator<ExecutionSpace, MemorySpace>(_comm);
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

    template <class InitFunctor>
    void initialize_ve(const InitFunctor init_functor,
                       const std::array<double, 2>& global_low_corner,
                       const std::array<double, 2>& global_high_corner,
                       const std::array<int, 2>& num_nodes,
                       const std::array<bool, 2>& periodic,
                       const Cabana::Grid::BlockPartitioner<2>& partitioner,
                       const double period,
                       MPI_Comm comm)
    {
        _grid2DInitializer = createGrid2DInitializer<ExecutionSpace, MemorySpace>(global_low_corner,
            global_high_corner, num_nodes, periodic, partitioner, period, comm);
        _grid2DInitializer->from_grid(init_functor, &_v_array, &_e_array, &_owned_vertices, &_owned_edges, &_owned_faces);
        _ghost_edges = 0;
        update_vef_counts();
        // printf("R%d: owned edges: %d\n", _rank, _owned_edges);
        // printEdges();
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
            int v_gid0, v_gid1, v_gid2;
            int e_gid0, e_lid0, e_gid1, e_gid2, e_lid2;
            v_gid0 = v_gid(i);
            // Follow first edge to get next vertex
            e_gid0 = v_gid0*3; e_lid0 = e_gid0 - _vef_gid_start_d(rank, 1);
            v_gid1 = e_vid(e_lid0, 1);
            // Use second vertex to get next edge
            e_gid1 = v_gid1*3+2;
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - _vef_gid_start_d(rank, 1);
            v_gid2 = e_vid(e_lid2, 1);
            
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
            v_gid1 = e_vid(e_lid0, 1);
            // Edge 2 GID is always the ID after edge 0
            e_gid2 = e_gid0+1; e_lid2 = e_gid2 - _vef_gid_start_d(rank, 1);
            v_gid2 = e_vid(e_lid2, 1);
            // DIFFERENT: Use vertex 2 to get edge 1. Edge 1 the first edge of vertex 2
            e_gid1 = v_gid2*3;

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
     * Assign edges to locally owned faces. Must be completed before gathering edges to 
     * ghosted edges have face global IDs of faces not owned by the remote process.
     */
    void initialize_edges()
    {
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
         * Iterate over faces and assign 1st and 3rd edges' face 1
         */
        int rank = _rank;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        Kokkos::parallel_for("assign_edges13_to_faces", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int f_gid, eX_lid, eX_gid;
            f_gid = f_lid + _vef_gid_start_d(rank, 2);

            // Where this edge is the first edge, set its face1
            eX_gid = f_egids(f_lid, 0);
            //if (eX_gid == 14) printf("R%d: e1_gid: %d, f_gid: %d\n", rank, eX_gid, f_gid);
            eX_lid = f_egids(f_lid, 0) - _vef_gid_start_d(rank, 1);
            e_fids(eX_lid, 0) = f_gid;
            
            // Where this edge is the third edge, set its face2
            eX_gid = f_egids(f_lid, 2);
            //if (eX_gid == 14) printf("R%d: e3_gid: %d, f_gid: %d\n", rank, eX_gid, f_gid);
            eX_lid = f_egids(f_lid, 2) - _vef_gid_start_d(rank, 1);
            e_fids(eX_lid, 1) = f_gid;
        });
    }
    /**
     * Gather edges owned on other ranks that are part of faces owned by this rank
     * For any face a process owns, it will always own its first and third edges.
     * The second edge needs to be gethered.
     */
    void gather_edges()
    {
        /* Temporary naive solution to store which edges are needed from which processes: 
         * Create a (comm_size x num_faces) view.
         * If an edge is needed from another process, set (owner_rank, f_lid) to the 
         * global edge ID needed from owner_rank.
         */ 
        // Set a counter to count number messages that will be sent
        using CounterView = Kokkos::View<int, device_type, Kokkos::MemoryTraits<Kokkos::Atomic>>;
        CounterView counter("counter");
        Kokkos::deep_copy(counter, 0);
        Kokkos::View<int**, device_type> sendvals_unpacked("sendvals_unpacked", _comm_size, _f_array.size());
        Kokkos::deep_copy(sendvals_unpacked, -1);
        // Step 2: Iterate over second edges. Populate Face ID that is not filled
        int rank = _rank, comm_size = _comm_size;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);
        auto f_egids = Cabana::slice<S_F_EIDS>(_f_array);
        Kokkos::parallel_for("find_needed_edge2", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int e2_gid, from_rank = -1;

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
                //if (rank == 1) printf("R%d: sendvals_unpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvals_unpacked(from_rank, f_lid));

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
                            //if (rank == 1) printf("R%d: sendvals_unpacked(%d, %d): %d\n", rank, from_rank, f_lid, sendvals_unpacked(from_rank, f_lid));
                        }
                    }
                }
            }

            // if (e_fids(e2_lid, 0) != -1) e_fids(e2_lid, 0) = f_gid;
            // else if (e_fids(e2_lid, 1) != -1) e_fids(e2_lid, 1) = f_gid;
        });
        Kokkos::fence();
        int num_sends = -1;
        Kokkos::deep_copy(num_sends, counter);
        _communicator->gather(sendvals_unpacked, _e_array, _vef_gid_start_d, 1, num_sends, _owned_edges, _ghost_edges);
    }

    void assign_ghost_edges_to_faces()
    {
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
         * Iterate over faces and assign 1st and 3rd edges' face 1
         */
        int rank = _rank;
        // Copy _vef_gid_start to device
        Kokkos::View<int*[3], device_type> _vef_gid_start_d("_vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(_vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, _vef_gid_start);
        Kokkos::deep_copy(_vef_gid_start_d, hv_tmp);

        // Assign the 2nd edges and ghosted edges to their faces
        int ghosted_edges = _ghost_edges;
        int owned_edges = _owned_edges;
        // printf("R%d: o: %d, `g: %d\n", rank, owned_edges, ghosted_edges);
        // printEdges();
        Kokkos::parallel_for("assign_edge2", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int f_gid, e2_gid, e2_lid;
            f_gid = f_lid + _vef_gid_start_d(rank, 2);
            e2_gid = f_egids(f_lid, 1);
            e2_lid = e2_gid - _vef_gid_start_d(rank, 1);
            //if (e2_gid == 0) printf("R%d f_gid %d, e2_gid %d, e min/max: (%d, %d)\n", rank, f_gid, e2_gid, _vef_gid_start_d(rank, 1),_vef_gid_start_d(rank+1, 1));

            // Check if edge owned locally
            if ((e2_lid >= 0) && (e2_lid < owned_edges))
            {
                //printf("R%d: acessing e_fids(%d)\n", rank, e2_lid);
                // Check if the first face is set; then this f_gid is the second face
                if (e_fids(e2_lid, 0) != -1) e_fids(e2_lid, 1) = f_gid;
                // Otherwise check if the second face is set; then this f_gid is the first face
                else if (e_fids(e2_lid, 1) != -1) e_fids(e2_lid, 0) = f_gid;
            }
            else
            {
                //printf("R%d: e_gid not owned locally: %d\n", rank, e2_gid);
                // Find the local_id of edge to set its face id(s)
                for (e2_lid = owned_edges; e2_lid < owned_edges+ghosted_edges; e2_lid++)
                {
                    int e2_gid_owned = e_gid(e2_lid);
                    if (e2_gid_owned == e2_gid)
                    {
                        //printf("R%d: ghosted e2_lid %d = e_gid %d\n", rank, e2_lid, e2_gid);
                        if (e_fids(e2_lid, 0) != -1) e_fids(e2_lid, 1) = f_gid;
                        // Otherwise check if the second face is set; then this f_gid is the first face
                        else if (e_fids(e2_lid, 1) != -1) e_fids(e2_lid, 0) = f_gid;
                    }
                }
            }
        });
        // Any edge-face mappings that are -1 at this point are for faces not owned by the process
        // printEdges(3);
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned vertices.
    auto indexSpace( Own, Vertex, Local ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _owned_vertices;

        return Cabana::Grid::IndexSpace<1>( size );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned+ghosted vertices.
    auto indexSpace( Ghost, Vertex, Local ) const
    {
        // Compute the lower bound.
        std::array<long, 1> min;
        min[0] = 0;

        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _owned_vertices + _ghost_vertices;

        return Cabana::Grid::IndexSpace<1>( min, max );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned edges.
    auto indexSpace( Own, Edge, Local ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _owned_edges;

        return Cabana::Grid::IndexSpace<1>( size );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned+ghosted edges.
    auto indexSpace( Ghost, Edge, Local ) const
    {
        // Compute the lower bound.
        std::array<long, 1> min;
        min[0] = 0;

        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _owned_edges + _ghost_edges;

        return Cabana::Grid::IndexSpace<1>( min, max );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned faces.
    auto indexSpace( Own, Face, Local ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _owned_faces;

        return Cabana::Grid::IndexSpace<1>( size );
    }

    std::vector<int> get_owned_and_ghost_counts() const
    {
        std::vector<int> out(6);
        out = {_owned_vertices, _owned_edges, _owned_faces,
               _ghost_vertices, _ghost_edges, _ghost_faces};
        return out;
    }

    int version() {return _version;}

    /**
     * After faces have been created, map each edge to its two faces.
     * The face where the edge is the lowest numbered edge, starting
     * at the first vertex and moving clockwise, is the first edge.
     */
    void assign_edges_to_faces_orig()
    {
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
        // Set a counter to count number messages that will be sent
        using CounterView = Kokkos::View<int, device_type, Kokkos::MemoryTraits<Kokkos::Atomic>>;
        CounterView counter("counter");
        Kokkos::deep_copy(counter, 0);
        Kokkos::View<int**, device_type> sendvals_unpacked("sendvals_unpacked", _comm_size, _f_array.size());
        Kokkos::deep_copy(sendvals_unpacked, -1);
        // Step 2: Iterate over second edges. Populate Face ID that is not filled
        Kokkos::parallel_for("find_needed_edge2", Kokkos::RangePolicy<execution_space>(0, _f_array.size()), KOKKOS_LAMBDA(int f_lid) {
            int e2_gid, from_rank = -1;

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
        // printf("R%d: counter: %d\n", rank, len_sendvals);

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
        for (int i = 0; i < (int)sendcounts_unpacked_h.extent(0); i++)
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
            for (int e_lid = 0; e_lid < (int)_f_array.size(); e_lid++)
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

        // Parse the received values to send the correct edges to the processes that requested them.
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
            for (int i = 0; i < (int)recv_edges.extent(0); i++)
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

    private:
        MPI_Comm _comm;
        int _rank, _comm_size;

        std::shared_ptr<Communicator<execution_space, memory_space>> _communicator;
        std::shared_ptr<Grid2DInitializer<execution_space, memory_space>> _grid2DInitializer;

        // AoSoAs for the mesh
        v_array_type _v_array;
        e_array_type _e_array;
        f_array_type _f_array;
        int _owned_vertices, _owned_edges, _owned_faces, _ghost_vertices, _ghost_edges, _ghost_faces;

        // How many vertices, edges, and faces each proces owns
        // Index = rank
        Kokkos::View<int*[3], Kokkos::HostSpace> _vef_gid_start;

        // Version number to keep mesh in sync with other objects. Updates on mesh refinement
        int _version;
        

};
//---------------------------------------------------------------------------//

/**
 *  Return a shared pointer to a Mesh object
 */
template <class ExecutionSpace, class MemorySpace>
auto createMesh( MPI_Comm comm )
{
    return std::make_shared<Mesh<ExecutionSpace, MemorySpace>>(comm);
}

} // end namespace NuMesh


#endif // NUMESH_MESH_HPP