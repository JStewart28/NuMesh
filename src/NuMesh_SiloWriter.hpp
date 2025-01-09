#ifndef NUMESH_SILOWRITER_HPP
#define NUMESH_SILOWRITER_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Mesh.hpp>

#include <pmpio.h>
#include <silo.h>
#include <sys/stat.h>

#include <mpi.h>

namespace NuMesh
{

//---------------------------------------------------------------------------//
/*!
  \class SiloWriter
  \brief Uses Silo to write the mesh to disk
*/
template <class Mesh>
class SiloWriter
{
  public:

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;

    SiloWriter( std::shared_ptr<Mesh> mesh )
        : _mesh ( mesh )
        , _comm ( mesh->comm() )
    {
        static_assert( isnumesh_mesh<Mesh>::value, "NuMesh::V2E: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

    };

    ~SiloWriter() {}

    /**
     * Write File
     * @param dbile File handler to dbfile
     * @param name File name
     * @param time_step Current time step
     * @param time Current tim
     * @param dt Time Step (dt)
     * @brief Writes the locally-owned portion of the mesh/variables to a file
     **/
    void writeFile( DBfile* dbfile, char* meshname, int time_step, double time,
                    double dt )
    {
        DBfile* silo_file = DBCreate("mesh_data.silo", DB_CLOBBER, DB_LOCAL, "Mesh Data", DB_PDB);


    }    

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;
    
    int _rank, _comm_size;
        
};

} // end namespce NuMesh

#endif // NUMESH_SILOWRITER_HPP
