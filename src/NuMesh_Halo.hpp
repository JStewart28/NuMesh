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
template <class EntityType, class Mesh>
class Halo
{
public:
    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;
    using int_d = Kokkos::View<size_t, memory_space>;
    using entity_type = EntityType; // Vertex, Edge, Face
    using halo_aosoa = typename Mesh::halo_aosoa;
    using halo_type = Cabana::Halo<memory_space>;

    /**
     * Create a halo at a given level and depth on an unstructured mesh
     */
    Halo(std::shared_ptr<Mesh> mesh, int level, int depth, EntityType entity)
        : _mesh(mesh), _level(level), _depth(depth), _entity(entity)
    {
        static_assert(is_numesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required");

        if (_depth < 1)
        {
            throw std::runtime_error("NuMesh::Halo must be initialized with a halo depth of at least 1.");
        }

        auto comm = _mesh->comm();
        MPI_Comm_size(comm, &_comm_size);
        if (_comm_size == 1) return; // No gather needed for comm size 1

        // Ensure the mesh is haloed to the proper level
        int mesh_halo_depth = _mesh->halo_depth();
        if (mesh_halo_depth < _depth)
        {
            _mesh->gather(_level, _depth);
            mesh_halo_depth = _mesh->halo_depth();
        }

        // Create the halo
        auto export_data_all = _mesh->halo_export(_entity);
        _halo_data = export_data_all[mesh_halo_depth - 1];
        assert(_halo_data->size() > 0);

        auto export_ids = Cabana::slice<0>(*_halo_data);
        auto export_ranks = Cabana::slice<1>(*_halo_data);

        _cabana_halo = std::make_shared<halo_type>(
            mesh->comm(), mesh->count(Own(), _entity), export_ids, export_ranks);

        _num_local = _cabana_halo->numLocal();
        _num_ghost = _cabana_halo->numGhost();
    }

    ~Halo() {}

    std::shared_ptr<Mesh> mesh() { return _mesh; }
    std::shared_ptr<halo_aosoa> data() { return _halo_data; }
    std::shared_ptr<halo_type> cabana_halo() {return _cabana_halo;}
    int level() const { return _level; }
    int depth() const { return _depth; }
    int comm_size() const { return _comm_size; }
    size_t numLocal() const { return _num_local; }
    size_t numGhost() const { return _num_ghost; }

private:
    std::shared_ptr<Mesh> _mesh;
    int _comm_size;
    const int _level;
    const int _depth;
    EntityType _entity;
    std::shared_ptr<halo_aosoa> _halo_data;
    std::shared_ptr<halo_type> _cabana_halo;
    size_t _num_local;
    size_t _num_ghost;
};


template <class Mesh, class EntityType>
auto createHalo(std::shared_ptr<Mesh> mesh, int level, int depth, EntityType entity)
{
    return std::make_shared<Halo<EntityType, Mesh>>(mesh, level, depth, entity);
}



template <class HaloType, class ArrayType>
void gather(std::shared_ptr<HaloType>& halo, std::shared_ptr<ArrayType> data)
{
    static_assert( Array::is_array<ArrayType>::value, "NuMesh::Array required" );

    using entity_type = typename ArrayType::entity_type;
    using memory_space = typename ArrayType::memory_space;
    auto mesh = halo->mesh();
    if (halo->comm_size() == 1) return; // No gather needed for comm size 1

    int mesh_halo_depth = mesh->halo_depth();
    int halo_halo_depth = halo->depth();
    if (mesh_halo_depth < halo_halo_depth)
    {
        throw std::runtime_error(
                    "NuMesh::gather: Mesh not haloed to depth of halo");
    }
    
    // Check that the data view is large anough for the gather
    size_t num_local = halo->numLocal();
    size_t num_ghost = halo->numGhost();
    auto aosoa = data->aosoa();
    if (aosoa.size() != (num_local+num_ghost))
    {
        throw std::runtime_error(
                    "NuMesh::gather: Array extents not large enough for gather");
    }
    auto cabana_halo = halo->cabana_halo();
    Cabana::gather(*cabana_halo, aosoa);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP