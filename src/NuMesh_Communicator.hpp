#ifndef NUMESH_COMMUNICATOR_HPP
#define NUMESH_COMMUNICATOR_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

#include <mpi_advance.h>

#include <limits>

int DEBUG_RANK = 0;

namespace NuMesh
{

template <class ExecutionSpace, class MemorySpace>
class Communicator
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    Communicator( const MPI_Comm comm )
            : _comm ( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        MPIX_Info_init(&_xinfo);
        MPIX_Comm_init(&_xcomm, _comm);
    }

    ~Communicator()
    {
        MPIX_Info_free(&_xinfo);
        MPIX_Comm_free(&_xcomm);
    }

    void gather()
    {
      
    }


  private:
    const MPI_Comm _comm;
    MPIX_Comm* _xcomm;
    MPIX_Info* _xinfo;

    int _rank, _comm_size;

};

/**
 *  Return a shared pointer to a Communcation object
 */
template <class ExecutionSpace, class MemorySpace>
auto createCommunicator( const MPI_Comm comm )
{
    return std::make_shared<Communicator<ExecutionSpace, MemorySpace>>(
        comm );
}

} // end namespace NUMesh


#endif // NUMESH_COMMUNICATOR_HPP