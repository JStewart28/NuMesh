#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
#include <NuMesh_Mesh.hpp>
#include <NuMesh_Maps.hpp>
#include <NuMesh_Array.hpp>

#include <mpi.h>

namespace NuMesh
{

//---------------------------------------------------------------------------//
/*!
  \class Halo
  \brief Unstructured triangle mesh
*/
template <class Mesh, class EntityType>
class Halo
{
  public:

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;
    using int_d = Kokkos::View<size_t, memory_space>;
    using entity_type = EntityType; // Vertex, edge, face
    using halo_aosoa = typename Mesh::halo_aosoa;

    /**
     * Create a halo at a given level and depth on an unstructured mesh
     */
    Halo( std::shared_ptr<Mesh> mesh, const int level, const int depth )
        : _mesh ( mesh )
        , _level ( level )
        , _depth ( depth )
        , _cabana_halo()
    {
        static_assert( is_numesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required" );

        if (_depth < 1)
        {
            throw std::runtime_error(
                    "NuMesh::Halo must be initialized with a halo depth of at least 1." );
        }

        auto comm = _mesh->comm();
        MPI_Comm_size( comm, &_comm_size );
        if (comm_size == 1) return; // No gather needed for comm size 1

        // Ensure the mesh is haloed to the proper level
        int mesh_halo_depth = _mesh->halo_depth();
        if (mesh_halo_depth < _depth)
        {
            _mesh->gather(_level, _depth);
        }

        // Create the halo
        auto export_data_all = _mesh->halo_export(entity_type());
        _halo_data = export_data_all[mesh_halo_depth-1];
        assert(_halo_data->size() > 0);
        auto export_ids = Cabana::slice<0>(*export_data);
        auto export_ranks = Cabana::slice<1>(*export_data);
        Cabana::Halo<memory_space> cabana_halo(mesh->comm(), mesh->count(Own(), entity_type()),
                                    export_ids, export_ranks);
        _num_local = cabana_halo.numLocal();
        _num_ghost = cabana_halo.numGhost();
    };

    ~Halo() {}

    std::shared_ptr<Mesh> mesh() {return _mesh;}
    std::shared_ptr<halo_aosoa> data() {return _halo_data;}
    int level() const {return _level;}
    int depth() const {return _depth;}
    int comm_size() const {return _comm_size;}
    size_t numLocal() const {return _num_local;}
    size_t numGhost() const {return _num_ghost;}


  private:
    std::shared_ptr<Mesh> _mesh;

    int _comm_size;   

    // Level of tree to halo at
    const int _level;

    // Halo depth into neighboring processes
    const int _depth;

    // Halo data
    std::shared_ptr<halo_aosoa> _halo_data;
    Cabana::Halo<memory_space> _cabana_halo
    size_t _num_local;
    size_t _num_ghost;
};

template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, const int level, const int depth)
{
    return Halo(mesh, level, depth);
}


template <class HaloType, class ArrayType>
void gather(HaloType& halo, std::shared_ptr<ArrayType> data)
{
    static_assert( Array::is_array<ArrayType>::value, "NuMesh::Array required" );

    using entity_type = typename ArrayType::entity_type;
    using memory_space = typename ArrayType::memory_space;
    auto mesh = halo.mesh();
    if (halo.comm_size() == 1) return; // No gather needed for comm size 1

    int mesh_halo_depth = mesh->halo_depth();
    int halo_halo_depth = halo.depth();
    if (mesh_halo_depth < halo_halo_depth)
    {
        throw std::runtime_error(
                    "NuMesh::gather: Mesh not haloed to depth of halo");
    }
    
    auto export_data = halo.data();
    assert(export_data->size() > 0);
    auto export_ids = Cabana::slice<0>(*export_data);
    auto export_ranks = Cabana::slice<1>(*export_data);
    
    // Check that the data view is large anough for the gather
    size_t num_local = halo.numLocal();
    size_t num_ghost = halo.numGhost();
    auto aosoa = data->aosoa();
    if (aosoa.size() != (num_local+num_ghost))
    {
        throw std::runtime_error(
                    "NuMesh::gather: Array extents not large enough for gather");
    }
    Cabana::gather(cabana_halo, aosoa);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP