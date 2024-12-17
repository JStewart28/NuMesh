#ifndef NUMESH_HALO_HPP
#define NUMESH_HALO_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <NuMesh_Utils.hpp>
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

    using memory_space = typename Mesh::memory_space;
    using execution_space = typename Mesh::execution_space;
    using integer_view = Kokkos::View<int*, memory_space>;

    Halo( std::shared_ptr<Mesh> mesh, const int entity, const int level, const int depth )
        : _mesh ( mesh )
        , _entity ( entity )
        , _level ( level )
        , _depth ( depth )
        , _comm ( mesh->comm() )
        , _halo_version ( mesh->version() )
    {
        static_assert( isnumesh_mesh<Mesh>::value, "NuMesh::Halo: NuMesh Mesh required" );

        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        _vertex_ids = integer_view("_vertex_ids", 0);
        _edge_ids = integer_view("_edge_ids", 0);
    };

    ~Halo()
    {
        //MPIX_Info_free(&_xinfo);
        //MPIX_Comm_free(&_xcomm);
    }

    /**
     * Given a list of vertices and a level of the tree, update
     * _vertex_ids and _edge_ids with edges connected to vertices
     * in the input list and the vertices of those connected edges.
     * 
     * Returns a list of new vertices added to the halo.
     */
    integer_view gather_local_neighbors(integer_view verts, int level)
    {
        if (_halo_version != _mesh->version())
        {
            throw std::runtime_error(
                "NuMesh::Halo::gather_local_neighbors: mesh and halo version do not match" );
        }

        const int rank = _rank, comm_size = _comm_size;

        /**
         * Allocate views. Each vertex can have at most six edges and six vertices
         * extending from it.
         * 
         * XXX - optimize this size estimate
         */
        int size = verts.extent(0);
        Kokkos::resize(_vertex_ids, _vertex_ids.extent(0) + size*6);
        Kokkos::resize(_edge_ids, _edge_ids.extent(0) + size*6);

        auto vertices = _mesh->vertices();
        auto edges = _mesh->edges();

        // Get vef_gid_start and copy to device
        auto vef_gid_start = _mesh->get_vef_gid_start();
        Kokkos::View<int*[3], memory_space> vef_gid_start_d("vef_gid_start_d", _comm_size);
        auto hv_tmp = Kokkos::create_mirror_view(vef_gid_start_d);
        Kokkos::deep_copy(hv_tmp, vef_gid_start);
        Kokkos::deep_copy(vef_gid_start_d, hv_tmp);

        int owned_vertices = _mesh->count(Own(), Vertex());

        // Vertex slices
        auto v_gid = Cabana::slice<V_GID>(vertices);

        // Edge slices
        auto e_gid = Cabana::slice<E_GID>(edges);
        auto e_vid = Cabana::slice<E_VIDS>(edges);
        auto e_rank = Cabana::slice<E_OWNER>(edges);
        auto e_cid = Cabana::slice<E_CIDS>(edges);
        auto e_pid = Cabana::slice<E_PID>(edges);
        auto e_layer = Cabana::slice<E_LAYER>(edges);


        // Map edges to vertices and edges to faces
        
        Kokkos::parallel_for("gather neighboring vertex and edge IDs", Kokkos::RangePolicy<execution_space>(0, size),
            KOKKOS_LAMBDA(int i) {
            
            int vlid = v_gid(i) - vef_gid_start_d(rank, 0);
            if ((vlid >= 0) && (vlid < owned_vertices)) // Make sure we own the vertex
            {
                
            }
        });
        Kokkos::fence();
    }

  private:
    std::shared_ptr<Mesh> _mesh;
    MPI_Comm _comm;

    // Entity halo type: 1 = edge, 2 = face
    const int _entity;

    // Level of tree to halo at
    const int _level;

    // Halo depth into neighboring processes
    const int _depth;

    // Mesh version this halo is built from
    int _halo_version;

    int _rank, _comm_size;
    
    // Vertex, Edge, and Face global IDs included in this halo
    integer_view _vertex_ids;
    integer_view _edge_ids;
    integer_view _face_ids;
    
};

template <class Mesh>
auto createHalo(std::shared_ptr<Mesh> mesh, Edge, int level, int depth)
{
    return Halo(mesh, 1, level, depth);
}

} // end namespce NuMesh


#endif // NUMESH_HALO_HPP