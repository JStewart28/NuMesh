#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Mesh.hpp>

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

    // using memory_space = typename Mesh::memory_space;
    // using execution_space = typename Mesh::execution_space;

    Halo( Mesh& numesh )
        : _comm ( numesh.comm() )
        , _mesh_version ( numesh.version() )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    };

    ~Halo()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
    }

  private:
    MPI_Comm _comm;
    int _mesh_version;

    int _rank, _comm_size;

    
};

auto createHalo(Mesh<ExecutionSpace, MemorySpace> mesh)
{
    return 0; 
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP