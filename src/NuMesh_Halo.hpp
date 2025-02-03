#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
#include <NuMesh_Mesh.hpp>
#include <NuMesh_Maps.hpp>

#include "_hypre_parcsr_ls.h"

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

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;
    using int_d = Kokkos::View<size_t, memory_space>;

    /**
     * Create a halo at a given level and depth on an unstructured mesh
     */
    Halo( std::shared_ptr<Mesh> mesh, const int level, const int depth )
        : _mesh ( mesh )
        , _level ( level )
        , _depth ( depth )
    {
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required" );

        if (_depth < 1)
        {
            throw std::runtime_error(
                    "NuMesh::Halo must be initialized with a halo depth of at least 1." );
        }

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );
    };

    ~Halo() {}

    std::shared_ptr<Mesh> mesh() {return _mesh;}
    int level() const {return _level;}
    int depth() const {return _depth;}


  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    int _rank, _comm_size;   

    // Level of tree to halo at
    const int _level;

    // Halo depth into neighboring processes
    const int _depth;
};

template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, const int level, const int depth)
{
    return Halo(mesh, level, depth);
}


template <class HaloType, class ArrayType>
void gather(const HaloType& halo, ArrayType& data)
{
    using entity_type = typename ArrayType::entity_type;
    int mesh_level = halo->mesh()->halo_level();
    int halo_level = halo->level();
    if (mesh_level < halo_level)
    {
        halo->mesh()->gather(halo_level, halo->depth());
    }
    auto export_indices = halo->mesh()->halo_export(entity_type);
    
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP