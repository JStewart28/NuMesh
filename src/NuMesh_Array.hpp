#ifndef NUMESH_ARRAY_HPP
#define NUMESH_ARRAY_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Mesh.hpp>

namespace NuMesh
{
namespace Array
{

// Design ideas for the NuMesh::Array taken from Cabana::Grid:Array

//---------------------------------------------------------------------------//
/*!
  \brief Entity layout for array data on the local unstructured mesh.

  \tparam EntityType Array entity type: Vertex, Edge, or Face
  \tparam MeshType Mesh type: UnstructuredMesh
*/
template <class EntityType, class MeshType>
class ArrayLayout
{
  public:
    //! Entity type.
    using entity_type = EntityType;

    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 1;

    /*!
      \brief Constructor.
      \param mesh The unstructured mesh over which the layout will be
      constructed.
      \param dofs_per_entity The number of degrees-of-freedom per EntityType entity.
    */
    ArrayLayout( const std::shared_ptr<mesh_type>& mesh,
                 const int dofs_per_entity )
        : _mesh( mesh )
        , _dofs_per_entity( dofs_per_entity )
    {
        update();
    }

    //! Get the local grid over which this layout is defined.
    const std::shared_ptr<mesh_type> mesh() const { return _mesh; }

    //! Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const { return _dofs_per_entity; }

    //! Get the index space of the array elements in the given
    //! decomposition.
    // template <class DecompositionTag, class IndexType>
    // Cabana::Grid::IndexSpace<num_space_dim + 1>
    // indexSpace( DecompositionTag decomposition_tag, IndexType index_type ) const
    // {
    //     return Cabana::Grid::appendDimension( _mesh->indexSpace( decomposition_tag,
    //                                                      EntityType(),
    //                                                      index_type ),
    //                             _dofs_per_entity );
    // }

        //---------------------------------------------------------------------------//
    // Get the local index space of the owned vertices.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Own, Vertex, Local ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _owned_vertices;
        auto is = Cabana::Grid::IndexSpace<1>( size );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned+ghosted vertices.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Ghost, Vertex, Local ) const
    {
        // Compute the lower bound.
        std::array<long, 1> min;
        min[0] = 0;

        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _owned_vertices + _ghost_vertices;

        auto is = Cabana::Grid::IndexSpace<1>( min, max );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned edges.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Own, Edge, Local ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _owned_edges;
        auto is = Cabana::Grid::IndexSpace<1>( size );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned+ghosted edges.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Ghost, Edge, Local ) const
    {
        // Compute the lower bound.
        std::array<long, 1> min;
        min[0] = 0;

        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _owned_edges + _ghost_edges;

        auto is = Cabana::Grid::IndexSpace<1>( min, max );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned faces.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Own, Face, Local ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _owned_faces;
        auto is = Cabana::Grid::IndexSpace<1>( size );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    int version() { return _version; }

    /**
     * Update to the latest owned and ghost counts from the mesh
     */
    void update()
    {
        _version = _mesh->version();
        auto counts = _mesh->get_owned_and_ghost_counts();
        _owned_vertices = counts[0]; _owned_edges = counts[1]; _owned_faces = counts[2];
        _ghost_vertices = counts[3]; _ghost_edges= counts[4]; _ghost_faces = counts[5];
    }


  private:
    std::shared_ptr<mesh_type> _mesh;
    int _dofs_per_entity;

    // Used to keep ArrayLayout in sync with mesh refinements
    int _version;
    int _owned_vertices, _owned_edges, _owned_faces, _ghost_vertices, _ghost_edges, _ghost_faces;
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a local grid.
  \param local_grid The local grid over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
  \return Shared pointer to an ArrayLayout.
  \note EntityType The entity: Cell, Node, Face, or Edge
*/
template <class EntityType, class MeshType>
std::shared_ptr<ArrayLayout<EntityType, MeshType>>
createArrayLayout( const std::shared_ptr<MeshType>& mesh,
                   const int dofs_per_entity, EntityType )
{
    return std::make_shared<ArrayLayout<EntityType, MeshType>>(
        mesh, dofs_per_entity );
}


//---------------------------------------------------------------------------//
/*!
  \brief Array of field data on the local mesh.

  \tparam Scalar Scalar type.
  \tparam EntityType Array entity type (vertex, edge, face).
  \tparam MeshType Mesh type (uniform, non-uniform).
  \tparam Params Kokkos View parameters.
*/
template <class Scalar, class EntityType, class MeshType, class... Params>
class Array
{
  public:
    //! Value type.
    using value_type = Scalar;

    //! Entity type.
    using entity_type = EntityType;

    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 1;

    //! Array layout type.
    using array_layout = ArrayLayout<entity_type, mesh_type>;

    //! View type.
    using view_type = Kokkos::View<value_type**, Params...>;

    //! Memory space.
    using memory_space = typename view_type::memory_space;
    //! Default device type.
    using device_type [[deprecated]] = typename memory_space::device_type;
    //! Default execution space.
    using execution_space = typename memory_space::execution_space;

    /*!
      \brief Create an array with the given layout. Arrays are constructed
      over the ghosted index space of the layout.
      \param label A label for the array.
      \param layout The array layout over which to construct the view.
    */
    Array( const std::string& label,
           const std::shared_ptr<array_layout>& layout )
        : _layout( layout )
        , _data( Cabana::Grid::createView<value_type, Params...>(
              label, layout->indexSpace( Ghost(), entity_type(), Local() ) ) )
    {
        _version = _layout->version();
    }

    /*!
      \brief Create an array with the given layout and view. This view should
      match the array index spaces in size.
      \param layout The layout of the array.
      \param view The array data.
    */
    Array( const std::shared_ptr<array_layout>& layout, const view_type& view )
        : _layout( layout )
        , _data( view )
    {
        for ( std::size_t d = 0; d < num_space_dim + 1; ++d )
            if ( (long)view.extent( d ) !=
                 layout->indexSpace( Ghost(), Local() ).extent( d ) )
                throw std::runtime_error(
                    "Layout and view dimensions do not match" );
    }

    //! Get the layout of the array.
    std::shared_ptr<array_layout> layout() const { return _layout; }

    //! Get a view of the array data.
    view_type view() const { return _data; }

    //! Get the array label.
    std::string label() const { return _data.label(); }

    //! Get the version of the array
    int version() { return _version; }

  private:
    std::shared_ptr<array_layout> _layout;
    view_type _data;

    // The ArrayLayout version, which points to the mesh version, this array is sized for
    int _version;

  public:
    //! Subview type.
    using subview_type = decltype( createSubview(
        _data, _layout->indexSpace( Ghost(), entity_type(), Local() ) ) );
    //! Subview array layout type.
    using subview_layout = typename subview_type::array_layout;
    //! Subview memory traits.
    using subview_memory_traits = typename subview_type::memory_traits;
    //! Subarray type.
    using subarray_type = Array<Scalar, EntityType, MeshType, subview_layout,
                                memory_space, subview_memory_traits>;
};

//---------------------------------------------------------------------------//
// Static type checker.
//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_array : public std::false_type
{
};

template <class Scalar, class EntityType, class MeshType, class... Params>
struct is_array<Array<Scalar, EntityType, MeshType, Params...>>
    : public std::true_type
{
};

template <class Scalar, class EntityType, class MeshType, class... Params>
struct is_array<const Array<Scalar, EntityType, MeshType, Params...>>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array with the given array layout. Views are constructed
  over the ghosted index space of the layout.
  \param label A label for the view.
  \param layout The array layout over which to construct the view.
  \return Shared pointer to an Array.
*/
template <class Scalar, class... Params, class EntityType, class MeshType>
std::shared_ptr<Array<Scalar, EntityType, MeshType, Params...>>
createArray( const std::string& label,
             const std::shared_ptr<ArrayLayout<EntityType, MeshType>>& layout )
{
    return std::make_shared<Array<Scalar, EntityType, MeshType, Params...>>(
        label, layout );
}



} // end namespace Array

//---------------------------------------------------------------------------//
// Array operations.
//---------------------------------------------------------------------------//
namespace ArrayOp
{

template <class Scalar, class... Params, class EntityType, class MeshType>
std::shared_ptr<Array<Scalar, EntityType, MeshType, Params...>>
clone( const Array<Scalar, EntityType, MeshType, Params...>& array )
{
    return createArray<Scalar, Params...>( array.label(), array.layout() );
}

} // end neamspace ArrayOp

} // end namespace NuMesh

#endif // NUMESH_ARRAY_HPP