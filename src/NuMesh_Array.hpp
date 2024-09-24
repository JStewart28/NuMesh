#ifndef NUMESH_ARRAY_HPP
#define NUMESH_ARRAY_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <NuMesh_Mesh.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

namespace NuMesh
{
namespace Array
{

// Design ideas for the NuMesh::Array taken from Cabana::Grid:Array

/* ArrayOp functions copied directly from Cabana::Grid::Array - can't
 * use Cabana::Grid::ArrayOp functions because they either type check
 * for Cabana::Grid::Arrays or take different tags (i.e. Node versus Vertex)
 */ 

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

        // Get the local index space of the owned+ghosted faces.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Ghost, Face, Local ) const
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

//! Array static type checker.
template <class>
struct is_array_layout : public std::false_type
{
};

//! Array static type checker.
template <class EntityType, class MeshType>
struct is_array_layout<ArrayLayout<EntityType, MeshType>>
    : public std::true_type
{
};

//! Array static type checker.
template <class EntityType, class MeshType>
struct is_array_layout<const ArrayLayout<EntityType, MeshType>>
    : public std::true_type
{
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

//---------------------------------------------------------------------------//
/*!
  \brief Copy one array into another over the designated decomposition. A <- B
  \param a The array to which the data will be copied.
  \param b The array from which the data will be copied.
  \param tag The tag for the decomposition over which to perform the operation.
*/
template <class Array_t, class DecompositionTag>
void copy( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cabana::Grid::Array required" );
    using entity_type = typename Array_t::entity_type;
    auto a_space = a.layout()->indexSpace( tag, entity_type(), Local() );
    auto b_space = b.layout()->indexSpace( tag, entity_type(), Local() );
    if ( a_space != b_space )
        throw std::logic_error( "Incompatible index spaces" );
    auto subview_a = Cabana::Grid::createSubview( a.view(), a_space );
    auto subview_b = Cabana::Grid::createSubview( b.view(), b_space );
    Kokkos::deep_copy( subview_a, subview_b );
}

//---------------------------------------------------------------------------//
/*!
  \brief Clone an array and copy its contents into the clone.
  \param array The array to clone.
  \param tag The tag for the decomposition over which to perform the copy.
*/
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> cloneCopy( const Array_t& array, DecompositionTag tag )
{
    auto cln = clone( array );
    copy( *cln, array, tag );
    return cln;
}

/*!
  \brief Update two vectors such that a = alpha * a + beta * b.
  \param a The array that will be updated.
  \param alpha The value to scale a by.
  \param b The array to add to a.
  \param beta The value to scale b by.
  \param tag The tag for the decomposition over which to perform the operation.
*/
template <class Array_t, class DecompositionTag>
std::enable_if_t<1 == Array_t::num_space_dim, void>
update( Array_t& a, const typename Array_t::value_type alpha, const Array_t& b,
        const typename Array_t::value_type beta, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );
    using entity_type = typename Array_t::entity_type;
    auto a_view = a.view();
    auto b_view = b.view();
    Kokkos::parallel_for(
        "ArrayOp::update",
        createExecutionPolicy( a.layout()->indexSpace( tag, entity_type(), Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const long i, const long j ) {
            a_view( i, j ) =
                alpha * a_view( i, j ) + beta * b_view( i, j );
        } );
}

/*!
  \brief Update three vectors such that a = alpha * a + beta * b + gamma * c.
  \param a The array that will be updated.
  \param alpha The value to scale a by.
  \param b The first array to add to a.
  \param beta The value to scale b by.
  \param c The second array to add to a.
  \param gamma The value to scale b by.
  \param tag The tag for the decomposition over which to perform the operation.
*/
template <class Array_t, class DecompositionTag>
std::enable_if_t<2 == Array_t::num_space_dim, void>
update( Array_t& a, const typename Array_t::value_type alpha, const Array_t& b,
        const typename Array_t::value_type beta, const Array_t& c,
        const typename Array_t::value_type gamma, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );
    using entity_type = typename Array_t::entity_type;
    auto a_view = a.view();
    auto b_view = b.view();
    auto c_view = c.view();
    Kokkos::parallel_for(
        "ArrayOp::update",
        createExecutionPolicy( a.layout()->indexSpace( tag, entity_type(), Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            a_view( i, j ) = alpha * a_view( i, j ) +
                                beta * b_view( i, j ) +
                                gamma * c_view( i, j );
        } );
}



} // end neamspace ArrayOp

} // end namespace Array

} // end namespace NuMesh

#endif // NUMESH_ARRAY_HPP