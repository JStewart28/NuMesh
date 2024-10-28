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
    Mesh( MPI_Comm comm )
        : _comm ( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
        _version = 0;

        _owned_vertices = -1, _owned_edges = -1, _owned_faces = -1;
        _ghost_vertices = 0, _ghost_edges = 0, _ghost_faces = 0;
    };


    private:
        MPI_Comm _comm;
        int _rank, _comm_size;

        // AoSoAs for the mesh
        v_array_type _v_array;
        e_array_type _e_array;
        f_array_type _f_array;
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

/**
 *  Returns a mesh created from a Cabana::LocalGrid object
 */
template <class ExecutionSpace, class MemorySpace, class LocalGrid>
auto createMeshFromLocalGrid( LocalGrid& local_grid )
{
    auto comm = local_grid.globalGrid()->comm();
    auto mesh = createEmptyMesh<MemorySpace, ExecutionSpace>(comm);
    return std::make_shared<Mesh<ExecutionSpace, MemorySpace>>(comm);
}

} // end namespace NuMesh


#endif // NUMESH_MESH_HPP