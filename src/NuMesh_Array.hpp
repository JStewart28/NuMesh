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
template <class EntityType, class MeshType, class TupleType>
class ArrayLayout
{
  public:
    using memory_space = typename MeshType::memory_space;

    //! Entity type.
    using entity_type = EntityType;

    //! Mesh type.
    using mesh_type = MeshType;

    //! AoSoA tuple type.
    using tuple_type = TupleType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 1;

    //! Ensure tuple_type is a Cabana::MemberTypes
    static_assert(IsCabanaMemberTypes<tuple_type>::value, "NuMesh::ArrayLayout: Tuple must be a Cabana::MemberType");

    //! Ensure tuple_type contains only a single type or array of types
    static_assert(IsSinglePartMemberTypes<tuple_type>::value, "NuMesh::ArrayLayout: Tuple of single type required");

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
        , _vef_gid_start(mesh->vef_gid_start())
    {
        // Assert tuple_size is the same as dofs_per_entity
        assert(ExtractArraySize<tuple_type>::value == dofs_per_entity);
    }

    //! Get the mesh over which this layout is defined.
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
        size[0] = _mesh->count(Own(), Vertex());
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
        max[0] = _mesh->count(Own(), Vertex()) + _mesh->count(Ghost(), Vertex());

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
        size[0] = _mesh->count(Own(), Edge());
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
        max[0] = _mesh->count(Own(), Edge()); + _mesh->count(Ghost(), Edge());;

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
        size[0] = _mesh->count(Own(), Face());
        auto is = Cabana::Grid::IndexSpace<1>( size );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    // Get the local index space of the owned+ghosted faces.
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( Ghost, Face, Local ) const
    {
        // Compute the lower bound.
        std::array<long, 1> min;
        min[0] = 0;

        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _mesh->count(Own(), Face()) + _mesh->count(Ghost(), Face());

        auto is = Cabana::Grid::IndexSpace<1>( min, max );
        return Cabana::Grid::appendDimension( is, _dofs_per_entity );
    }

    // Get the local index space of the owned+ghosted ArrayLayout entity type.
    template <class DecompositionType, class IndexType>
    Cabana::Grid::IndexSpace<num_space_dim + 1>
    indexSpace( DecompositionType dt, IndexType it ) const
    {
        return indexSpace(dt, entity_type(), it);
    }

    //-------------------------------------------------------------------------
    // Get index spaces for the vertices/edges/faces only, 
    // and not the values associated with them
    //-------------------------------------------------------------------------

    // Get the local index space of the owned vertices.
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( Own, Vertex, Local, Element ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _mesh->count(Own(), Vertex());
        return Cabana::Grid::IndexSpace<1>( size );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned+ghosted vertices.
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( Ghost, Vertex, Local, Element ) const
    {
        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _mesh->count(Own(), Vertex()) + _mesh->count(Ghost(), Vertex());

        return Cabana::Grid::IndexSpace<1>( max );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned edges.
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( Own, Edge, Local, Element ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _mesh->count(Own(), Edge());
        return Cabana::Grid::IndexSpace<1>( size );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned+ghosted edges.
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( Ghost, Edge, Local, Element ) const
    {
        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _mesh->count(Own(), Edge()); + _mesh->count(Ghost(), Edge());

        return Cabana::Grid::IndexSpace<1>( max );
    }

    //---------------------------------------------------------------------------//
    // Get the local index space of the owned faces.
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( Own, Face, Local, Element ) const
    {
        // Compute the size.
        std::array<long, 1> size;
        size[0] = _mesh->count(Own(), Face());
        return Cabana::Grid::IndexSpace<1>( size );
    }

    // Get the local index space of the owned+ghosted faces.
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( Ghost, Face, Local, Element ) const
    {
        // Compute the upper bound.
        std::array<long, 1> max;
        max[0] = _mesh->count(Own(), Face()) + _mesh->count(Ghost(), Face());

        return Cabana::Grid::IndexSpace<1>( max );
    }

    // Get the local index space of the owned+ghosted ArrayLayout entity type, element version.
    template <class DecompositionType, class IndexType, class ElementType>
    Cabana::Grid::IndexSpace<num_space_dim>
    indexSpace( DecompositionType dt, IndexType it, Element e) const
    {
        return indexSpace(dt, entity_type(), it, e);
    }

  private:
    std::shared_ptr<mesh_type> _mesh;
    int _dofs_per_entity;

    // The global ID starts of vertices, edges, and faces that this layout is associated with
    Kokkos::View<int*[3], memory_space> _vef_gid_start;
};

//! Array static type checker.
template <class>
struct is_array_layout : public std::false_type
{
};

//! Array static type checker.
template <class EntityType, class MeshType, class TupleType>
struct is_array_layout<ArrayLayout<EntityType, MeshType, TupleType>>
    : public std::true_type
{
};

//! Array static type checker.
template <class EntityType, class MeshType, class TupleType>
struct is_array_layout<const ArrayLayout<EntityType, MeshType, TupleType>>
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
template <class TupleType, class EntityType, class MeshType>
std::shared_ptr<ArrayLayout<EntityType, MeshType, TupleType>>
createArrayLayout( const std::shared_ptr<MeshType>& mesh,
                   const int dofs_per_entity, EntityType )
{
    return std::make_shared<ArrayLayout<EntityType, MeshType, TupleType>>(
        mesh, dofs_per_entity );
}


//---------------------------------------------------------------------------//
/*!
  \brief A wrapper around a slice of some data in the unstructured mesh

  \tparam Scalar Scalar type.
  \tparam MeshType Mesh type (uniform, non-uniform).
  \tparam Params Kokkos View parameters.
*/
template <class MemorySpace, class LayoutType>
class Array
{
  public:
    //! Memory space.
    using memory_space = MemorySpace;
    //! Default device type.
    using device_type [[deprecated]] = typename memory_space::device_type;
    //! Default execution space.
    using execution_space = typename memory_space::execution_space;

    //! Entity type.
    using entity_type = typename LayoutType::entity_type;

    //! Tuple type.
    using tuple_type = typename LayoutType::tuple_type;

    //! Value type in the tuple
    using value_type = typename ExtractSingleType<
        typename ExtractBaseTypes<tuple_type>::type>::type;

    using aosoa_type = Cabana::AoSoA<tuple_type, memory_space, 4>;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 1;

    /*!
      \brief Create an array with the given layout. Arrays are constructed
      over the ghosted index space of the layout.
      \param label A label for the array.
      \param layout The array layout over which to construct the view.
    */
    Array( const std::string& label,
           const std::shared_ptr<LayoutType>& layout )
        : _layout( layout )
        , _data( Cabana::AoSoA<tuple_type, memory_space, 4>(
              label, layout->indexSpace( Ghost(), entity_type(), Local(), Element() ).extent(0) ) )
        , _vef_gid_start( layout->mesh()->vef_gid_start())
    {
        _version = _layout->mesh()->version();
    }

    //! Update the array to match the size of the new layout
    void update()
    {
        if (_version == _layout->mesh()->version()) return;

        // Resize the array
        size_t new_size = _layout->indexSpace(Ghost(), entity_type(), Local(), Element()).extent(0);
        // printf("Updating size: %d -> %d\n", _data.extent(0), new_size);
        _data.resize(new_size);

        // Update global ID starts
        _vef_gid_start = _layout->mesh()->vef_gid_start();

        // Update version
        _version = _layout->mesh()->version();
    }

    //! Get the layout of the array.
    std::shared_ptr<LayoutType> layout() const { return _layout; }

    //! Get the aosoa of the array data.
    aosoa_type aosoa() const { return _data; }

    //! Get the aosoa label.
    std::string label() const { return _data.label(); }

    //! Get the version of the array
    int version() { return _version; }

  private:
    std::shared_ptr<LayoutType> _layout;
    aosoa_type _data;

    // The ArrayLayout version, which points to the mesh version, this array is sized for
    int _version;

    // The global ID starts of vertices, edges, and faces that this layout is associated with
    Kokkos::View<int*[3], memory_space> _vef_gid_start;

//   public:
//     //! Subview type.
//     using subview_type = decltype( createSubview(
//         _data, _layout->indexSpace( Ghost(), entity_type(), Local() ) ) );
//     //! Subview array layout type.
//     using subview_layout = typename subview_type::array_layout;
//     //! Subview memory traits.
//     using subview_memory_traits = typename subview_type::memory_traits;
//     //! Subarray type.
//     using subarray_type = Array<Scalar, EntityType, MeshType, subview_layout,
//                                 memory_space, subview_memory_traits>;
};

//---------------------------------------------------------------------------//
// Static type checker.
//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_array : public std::false_type
{
};

template <class MemorySpace, class LayoutType>
struct is_array<Array<MemorySpace, LayoutType>>
    : public std::true_type
{
};

template <class MemorySpace, class LayoutType>
struct is_array<const Array<MemorySpace, LayoutType>>
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
template <class MemorySpace, class LayoutType>
std::shared_ptr<Array<MemorySpace, LayoutType>>
createArray( const std::string& label,
             const std::shared_ptr<LayoutType>& layout )
{
    return std::make_shared<Array<MemorySpace, LayoutType>>(
        label, layout );
}

//---------------------------------------------------------------------------//
// Array operations.
// NOTE: The functions in this namespace are copied almost identically from
// Cabana::Array::ArrayOp
// https://github.com/ECP-copa/Cabana/blob/master/grid/src/Cabana_Grid_Array.hpp
//---------------------------------------------------------------------------//
namespace ArrayOp
{

template <class MemorySpace, class LayoutType>
std::shared_ptr<Array<MemorySpace, LayoutType>>
clone( const Array<MemorySpace, LayoutType>& array )
{
    return createArray<MemorySpace>( array.label(), array.layout() );
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
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );

    using execution_space = typename Array_t::execution_space;
    using entity_type = typename Array_t::entity_type;
    using tuple_type = typename Array_t::tuple_type;

    auto a_space = a.layout()->indexSpace( tag, entity_type(), Local() );
    auto b_space = b.layout()->indexSpace( tag, entity_type(), Local() );
    if ( a_space != b_space )
        throw std::logic_error( "NuMesh::ArrayOp::copy: Incompatible index spaces" );
    // auto subview_a = Cabana::subview( a, a.view(), a_space );
    // auto subview_b = Cabana::Grid::createSubview( b.view(), b_space );
    // Kokkos::deep_copy( subview_a, subview_b );
    // printf("aspace: (%d, %d), (%d, %d), bspace: (%d, %d), (%d, %d)\n",
    //     a_space.min(0), a_space.max(0), a_space.min(1), a_space.max(1),
    //     b_space.min(0), b_space.max(0), b_space.min(1), b_space.max(1));
    auto a_data = Cabana::slice<0>(a.aosoa());
    auto b_data = Cabana::slice<0>(b.aosoa());

    constexpr int tuple_size = ExtractArraySize<tuple_type>::value;

    // Slices for non-array tuple AoSoAs have the form slice(i)
    if constexpr (tuple_size == 1)
    {
        // Slices for array tuple AoSoAs have the form slice(i, j)
        auto space = a.layout()->indexSpace( tag, entity_type(), Local(), Element() );
        auto policy = Cabana::Grid::createExecutionPolicy(space, execution_space());
        Kokkos::parallel_for( "NuMesh::ArrayOp::copy", policy,
            KOKKOS_LAMBDA( const int i ) {
                a_data( i ) = b_data( i );
            } );
    }
    else if constexpr (tuple_size > 1)
    {
        // Slices for array tuple AoSoAs have the form slice(i, j)
        auto policy = Cabana::Grid::createExecutionPolicy(a_space, execution_space());
        Kokkos::parallel_for( "NuMesh::ArrayOp::copy", policy,
            KOKKOS_LAMBDA( const int i, const int j) {
                a_data( i, j ) = b_data( i, j );
            } );
    }
    else
    {
        static_assert(tuple_size > 1 || tuple_size == 1,
            "NuMesh::ArrayOp::copy: Invalid tuple size!");
    }
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

/**
 * Create a copy of one dimension of an AoSoA slice
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> copyDim( Array_t& a, int dimA, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using original_tuple_type = typename Array_t::tuple_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    // Extract the base type from the original tuple type
    using extracted_base_tuple = typename ExtractBaseTypes<original_tuple_type>::type;
    static_assert(std::tuple_size<extracted_base_tuple>::value == 1, 
                  "copyDim only supports tuple types with a single unique base type.");

    using base_type = typename std::tuple_element<0, extracted_base_tuple>::type;

    // Define new tuple type with a single non-array value
    using new_tuple_type = Cabana::MemberTypes<base_type>;

    // Create new layout with updated tuple type
    auto layout = NuMesh::Array::createArrayLayout<new_tuple_type>( a.layout(), 1, entity_type() );
    auto out = NuMesh::Array::createArray<memory_space>("copyDim_out", layout);
    auto out_aosoa = out->aosoa();
    auto out_slice = Cabana::slice<0>(out_aosoa);

    // Check dimensions
    auto a_aosoa = a.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    constexpr int aw = ExtractArraySize<original_tuple_type>::value;

    if (dimA >= aw) {
        throw std::invalid_argument(
            "NuMesh::ArrayOp::copyDim: Provided dimension is larger than the number of dimensions in the array.");
    }

    auto policy = Cabana::Grid::createExecutionPolicy(
        a.layout()->indexSpace(tag, entity_type(), Local(), Element()),
        execution_space());

    Kokkos::parallel_for(
        "NuMesh::ArrayOp::copyDim", policy,
        KOKKOS_LAMBDA(const int i) {
            out_slice(i) = a_slice(i, dimA);
        });

    return out;
}

/**
 * Copy dimB from b into dimA from a 
 */
template <class A_t, class B_t, class DecompositionTag>
void copyDim( A_t& a, int dimA, B_t& b, int dimB, DecompositionTag tag )
{
    using a_entity_type = typename A_t::entity_type;
    using b_entity_type = typename B_t::entity_type;
    using a_memory_space = typename A_t::memory_space;
    using b_memory_space = typename B_t::memory_space;
    using a_execution_space = typename A_t::execution_space;
    using b_execution_space = typename B_t::execution_space;
    using a_tuple_type = typename A_t::tuple_type;
    using b_tuple_type = typename B_t::tuple_type;

    // Check that the types are equal for both arrays
    static_assert(std::is_same<a_entity_type, b_entity_type>::value,
        "NuMesh::ArrayOp::copyDim: Types are not the same!");
    static_assert(std::is_same<a_memory_space, b_memory_space>::value,
        "NuMesh::ArrayOp::copyDim: Types are not the same!");
    static_assert(std::is_same<a_execution_space, b_execution_space>::value,
        "NuMesh::ArrayOp::copyDim: Types are not the same!");

    // Check dimensions
    auto a_aosoa = a.aosoa();
    auto b_aosoa = b.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    auto b_slice = Cabana::slice<0>(b_aosoa);

    const int an = a_aosoa.size();
    const int bn = b_aosoa.size();
    constexpr int am = ExtractArraySize<a_tuple_type>::value;
    constexpr int bm = ExtractArraySize<b_tuple_type>::value;
    
    static_assert(!((am == 1) && (bm == 1)), "NuMesh::ArrayOp::copyDim: Use copy when tuples are the same size");

    if (an != bn) {
        throw std::invalid_argument("NuMesh::ArrayOp::copyDim: First dimension of a and b arrays do not match.");
    }
    if (dimA >= am) {
        throw std::invalid_argument("NuMesh::ArrayOp::copyDim: Provided dimension for 'a' is larger than the number of dimensions in the b array.");
    }
    if (dimB >= bm) {
        throw std::invalid_argument("NuMesh::ArrayOp::copyDim: Provided dimension for 'b' is larger than the number of dimensions in the b array.");
    }

    if constexpr ((am == bm) && (am != 1))
    {
        // A and B are not tuple size 1
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->indexSpace( tag, a_entity_type(), Local() ),
            a_execution_space() );
        Kokkos::parallel_for(
            "NuMesh::ArrayOp::copyDim", policy,
            KOKKOS_LAMBDA( const int i, const int j) {
                a_slice( i, dimA ) = b_slice( i, dimB );
        } );
    }
    if constexpr ((am == 1) && (am < bm))
    {
        // If a has tuple size 1 but not b
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->indexSpace( tag, a_entity_type(), Local(), Element() ),
            a_execution_space() );
        Kokkos::parallel_for(
            "NuMesh::ArrayOp::copyDim", policy,
            KOKKOS_LAMBDA( const int i ) {
                a_slice( i ) = b_slice( i, dimB );
        } );
    }
    if constexpr ((bm == 1) && (bm < am))
    {
        // If b has tuple size 1 but not a
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->indexSpace( tag, a_entity_type(), Local(), Element() ),
            a_execution_space() );
        Kokkos::parallel_for(
            "NuMesh::ArrayOp::copyDim", policy,
            KOKKOS_LAMBDA( const int i ) {
                a_slice( i, dimA ) = b_slice( i );
        } );
    }
    else
    {
        throw std::invalid_argument("First array argument must have equal or smaller third dimension than second array argument.");
    }
}

//---------------------------------------------------------------------------//
/*!
  \brief Assign a scalar value to every element of an aosoa slice.
  \param array The array to assign the value to.
  \param alpha The value to assign to the array.
  \param tag The tag for the decomposition over which to perform the operation.
*/

template <class Array_t, class DecompositionTag>
void assign( Array_t& array, typename Array_t::value_type alpha, 
             DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );

    using execution_space = typename Array_t::execution_space;
    using value_type = typename Array_t::value_type;
    using entity_t = typename Array_t::entity_type;
    using tuple_type = typename Array_t::tuple_type;
    
    auto aosoa = array.aosoa();
    auto slice = Cabana::slice<0>(aosoa);
    constexpr int tuple_size = ExtractArraySize<tuple_type>::value;
    if constexpr(tuple_size > 1)
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            array.layout()->indexSpace(tag, entity_t(), Local()), execution_space() );
   
        Kokkos::parallel_for("NuMesh::ArrayOp::assign", policy,
           KOKKOS_LAMBDA( const int i, const int j ) {
            slice(i, j) = alpha;
        });
    }
    else if constexpr(tuple_size == 1)
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            array.layout()->indexSpace(tag, entity_t(), Local(), Element()), execution_space() );
   
        Kokkos::parallel_for("NuMesh::ArrayOp::assign", policy,
           KOKKOS_LAMBDA( const int i ) {
            slice(i) = alpha;
        });
    }
    else
    {
        static_assert(tuple_size > 1 || tuple_size == 1,
            "NuMesh::ArrayOp::assign: Invalid tuple size");
    }
}


/*!
  \brief Scale every element of an aosoa slice by a scalar value. 2D specialization.
  \param array The array to scale.
  \param alpha The value to scale the array by.
  \param tag The tag for the decomposition over which to perform the operation.
*/
template <class Array_t, class DecompositionTag>
std::enable_if_t<1 == Array_t::num_space_dim, void>
scale( Array_t& array, const typename Array_t::value_type alpha, 
       DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );

    using entity_t = typename Array_t::entity_type;
    using tuple_type = typename Array_t::tuple_type;
    using execution_space = typename Array_t::execution_space;
    auto aosoa = array.aosoa();
    auto slice = Cabana::slice<0>(aosoa);
    constexpr int tuple_size = ExtractArraySize<tuple_type>::value;
    if constexpr(tuple_size > 1)
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            array.layout()->indexSpace(tag, entity_t(), Local()), execution_space() );
   
        Kokkos::parallel_for("NuMesh::ArrayOp::scale", policy,
           KOKKOS_LAMBDA( const int i, const int j ) {
            slice(i, j) *= alpha;
        });
    }
    else if constexpr(tuple_size == 1)
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            array.layout()->indexSpace(tag, entity_t(), Local(), Element()), execution_space() );
   
        Kokkos::parallel_for("NuMesh::ArrayOp::scale", policy,
           KOKKOS_LAMBDA( const int i ) {
            slice(i) *= alpha;
        });
    }
    else
    {
        static_assert(tuple_size > 1 || tuple_size == 1, 
            "NuMesh::ArrayOp::scale: Invalid tuple size");
    }
}

/*!
  \brief Apply some function to every element of an aosoa slice
  \param array The array to operate on.
  \param function A functor that operates on the array elements.
  \param tag The tag for the decomposition over which to perform the operation.
*/
template <class Array_t, class Function, class DecompositionTag>
std::enable_if_t<1 == Array_t::num_space_dim, void>
apply( Array_t& array, Function& function, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );

    using tuple_type = typename Array_t::tuple_type;
    using entity_t = typename Array_t::entity_type;
    using execution_space = typename Array_t::execution_space;
    auto aosoa = array.aosoa();
    auto slice = Cabana::slice<0>(aosoa);
    constexpr int tuple_size = ExtractArraySize<tuple_type>::value;
    if constexpr(tuple_size > 1)
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            array.layout()->indexSpace(tag, entity_t(), Local()), execution_space() );
   
        Kokkos::parallel_for("NuMesh::ArrayOp::scale", policy,
           KOKKOS_LAMBDA( const int i, const int j ) {
            slice(i, j) *= function(slice( i, j ));
        });
    }
    else if constexpr(tuple_size == 1)
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            array.layout()->indexSpace(tag, entity_t(), Local(), Element()), execution_space());
   
        Kokkos::parallel_for("NuMesh::ArrayOp::scale", policy,
           KOKKOS_LAMBDA( const int i ) {
            slice(i) *= function(slice( i ));
        });
    }
    else
    {
        static_assert(tuple_size > 1 || tuple_size == 1, 
            "NuMesh::ArrayOp::apply: Invalid tuple size");
    }
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
update( Array_t& a, const typename Array_t::tuple_type alpha, 
        const Array_t& b, const typename Array_t::tuple_type beta, 
        DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );
    using entity_type = typename Array_t::entity_type;
    using tuple_type = typename Array_t::tuple_type;
    auto a_aosoa = a.aosoa();
    auto b_aosoa = b.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    auto b_slice = Cabana::slice<0>(b_aosoa);
    constexpr int tuple_size = ExtractArraySize<tuple_type>::value;
    if constexpr(tuple_size > 1)
    {
        Kokkos::parallel_for( "ArrayOp::update",
            createExecutionPolicy( a.layout()->indexSpace( tag, entity_type(), Local() ),
                                   typename Array_t::execution_space() ),
            KOKKOS_LAMBDA( const long i, const long j ) {
                a_slice( i, j ) =
                    alpha * a_slice( i, j ) + beta * b_slice( i, j );
            });
    }
    else if constexpr(tuple_size == 1)
    {
        Kokkos::parallel_for( "ArrayOp::update",
            createExecutionPolicy( a.layout()->indexSpace( tag, entity_type(), Local(), Element() ),
                                   typename Array_t::execution_space() ),
            KOKKOS_LAMBDA( const long i ) {
                a_slice( i ) =
                    alpha * a_slice( i ) + beta * b_slice( i );
            });
    }
    else
    {
        static_assert(tuple_size > 1 || tuple_size == 1, 
            "NuMesh::ArrayOp::update: Invalid tuple size");
    }
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
std::enable_if_t<1 == Array_t::num_space_dim, void>
update( Array_t& a, const typename Array_t::tuple_type alpha, 
        const Array_t& b, const typename Array_t::tuple_type beta,
        const Array_t& c, const typename Array_t::tuple_type gamma,
        DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "NuMesh::Array required" );
    using entity_type = typename Array_t::entity_type;
    using tuple_type = typename Array_t::tuple_type;
    auto a_aosoa = a.aosoa();
    auto b_aosoa = b.aosoa();
    auto c_aosoa = c.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    auto b_slice = Cabana::slice<0>(b_aosoa);
    auto c_slice = Cabana::slice<0>(c_aosoa);
    constexpr int tuple_size = ExtractArraySize<tuple_type>::value;
    if constexpr(tuple_size > 1)
    {
        Kokkos::parallel_for("ArrayOp::update",
            createExecutionPolicy( a.layout()->indexSpace( tag, entity_type(), Local() ),
                                   typename Array_t::execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                a_slice( i, j ) = alpha * a_slice( i, j ) +
                                    beta * b_slice( i, j ) +
                                    gamma * c_slice( i, j );
            });
    }
    else if constexpr(tuple_size == 1)
    {
        Kokkos::parallel_for("ArrayOp::update",
            createExecutionPolicy( a.layout()->indexSpace( tag, entity_type(), Local(), Element() ),
                                   typename Array_t::execution_space() ),
            KOKKOS_LAMBDA( const int i ) {
                a_slice( i ) = alpha * a_slice( i ) +
                                    beta * b_slice( i ) +
                                    gamma * c_slice( i );
            });
    }
    else
    {
        static_assert(tuple_size > 1 || tuple_size == 1, 
            "NuMesh::ArrayOp::update: Invalid tuple size");
    }
}

/**
 * Element-wise dot product
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_dot( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using original_tuple_type = typename Array_t::tuple_type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    // Extract the base type from the original tuple type
    using extracted_base_tuple = typename ExtractBaseTypes<original_tuple_type>::type;
    static_assert(std::tuple_size<extracted_base_tuple>::value == 1, 
                  "element_dot only supports tuple types with a single unique base type.");

    using base_type = typename std::tuple_element<0, extracted_base_tuple>::type;

    // Define new tuple type for scalar output
    using scalar_tuple_type = Cabana::MemberTypes<base_type>;

    // Create new layout for the scalar output
    auto scalar_layout = NuMesh::Array::createArrayLayout<scalar_tuple_type>(a.layout()->mesh(), 1, entity_type());
    auto dot = NuMesh::Array::createArray<memory_space>("dot", scalar_layout);
    auto dot_aosoa = dot->aosoa();
    auto dot_slice = Cabana::slice<0>(dot_aosoa);

    // Check dimensions
    auto a_aosoa = a.aosoa();
    auto b_aosoa = b.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    auto b_slice = Cabana::slice<0>(b_aosoa);

    const int an = a_aosoa.size();
    const int bn = b_aosoa.size();
    constexpr int am = ExtractArraySize<original_tuple_type>::value;

    // Ensure the second dimension is 3 for 3D vectors
    if (am != 3) {
        throw std::invalid_argument("element_dot: Second dimension must be 3 for 3D vectors.");
    }
    if (an != bn) {
        throw std::invalid_argument("element_dot: First dimension of a and b views do not match.");
    }

    auto policy = Cabana::Grid::createExecutionPolicy(
            scalar_layout->indexSpace(tag, entity_type(), NuMesh::Local(), Element()),
            execution_space());
    
    Kokkos::parallel_for("compute_dot_product", policy,
        KOKKOS_LAMBDA(const int i) {
            dot_slice(i, 0) = a_slice(i, 0) * b_slice(i, 0)
                            + a_slice(i, 1) * b_slice(i, 1)
                            + a_slice(i, 2) * b_slice(i, 2);
        });

    return dot;
}

/**
 * Element-wise cross product
 */
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> element_cross( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    using entity_type = typename Array_t::entity_type;
    using tuple_type = typename  Array_t::tuple_type;
    using value_type = typename ExtractBaseTypes<
            typename Array_t::tuple_type>::type;
    using memory_space = typename Array_t::memory_space;
    using execution_space = typename Array_t::execution_space;

    // The resulting 'dot' array has the shape (i, j, 3)
    auto layout = NuMesh::Array::createArrayLayout<tuple_type>(a.layout()->mesh(), 3, entity_type());
    auto cross = NuMesh::Array::createArray<memory_space>("cross", layout);
    auto cross_aosoa = cross->aosoa();
    auto cross_slice = Cabana::slice<0>(cross_aosoa);

    // Check dimensions
    auto a_aosoa = a.aosoa();
    auto b_aosoa = b.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    auto b_slice = Cabana::slice<0>(b_aosoa);

    const int an = a_aosoa.size();
    const int bn = b_aosoa.size();
    constexpr int am = ExtractArraySize<tuple_type>::value;

    // Ensure the third dimension is 3 for 3D vectors
    if (am != 3) {
        throw std::invalid_argument("Second dimension must be 3 for 3D vectors.");
    }
    if (an != bn) {
        throw std::invalid_argument("First dimension of a and b views do not match.");
    }

    auto policy = Cabana::Grid::createExecutionPolicy(
            layout->indexSpace( tag, entity_type(), NuMesh::Local(), Element() ),
            execution_space() );
    // Create output view for cross product results
    Kokkos::parallel_for("CrossProductKernel", policy,
        KOKKOS_LAMBDA(const int i) {
        value_type a_x = a_slice(i, 0);
        value_type a_y = a_slice(i, 1);
        value_type a_z = a_slice(i, 2);
        
        value_type b_x = b_slice(i, 0);
        value_type b_y = b_slice(i, 1);
        value_type b_z = b_slice(i, 2);

        // Cross product: a x b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
        cross_slice(i, 0) = a_y * b_z - a_z * b_y;
        cross_slice(i, 1) = a_z * b_x - a_x * b_z;
        cross_slice(i, 2) = a_x * b_y - a_y * b_x;
    });

    return cross;
}

/**
 * Element-wise multiplication, where (1, 3) * (4, 7) = (4, 21)
 * If a and b do not have matching second dimensions, place the view with the 
 * smaller second dimension first.
 * 
 * If a has a third dimension of 1, out(x, y) = b(x, y) * a(x, 0) for 0 <= y < b extent
 */ 
template <class A_t, class B_t, class DecompositionTag>
std::shared_ptr<A_t> element_multiply( A_t& a, const B_t& b, DecompositionTag tag )
{
    using a_entity_type = typename A_t::entity_type;
    using b_entity_type = typename B_t::entity_type;
    using a_memory_space = typename A_t::memory_space;
    using b_memory_space = typename B_t::memory_space;
    using a_execution_space = typename A_t::execution_space;
    using b_execution_space = typename B_t::execution_space;
    using a_tuple_type = typename A_t::tuple_type;
    using b_tuple_type = typename B_t::tuple_type;

    // Check that the types are equal for both arrays
    static_assert(std::is_same<a_entity_type, b_entity_type>::value,
        "NuMesh::ArrayOp::copyDim: Types are not the same!");
    static_assert(std::is_same<a_memory_space, b_memory_space>::value,
        "NuMesh::ArrayOp::copyDim: Types are not the same!");
    static_assert(std::is_same<a_execution_space, b_execution_space>::value,
        "NuMesh::ArrayOp::copyDim: Types are not the same!");

    auto out = clone(a);
    auto out_aosoa = out->aosoa();
    auto out_slice = Cabana::slice<0>(out_aosoa);

    // Check dimensions
    auto a_aosoa = a.aosoa();
    auto b_aosoa = b.aosoa();
    auto a_slice = Cabana::slice<0>(a_aosoa);
    auto b_slice = Cabana::slice<0>(b_aosoa);

    const int an = a_aosoa.size();
    const int bn = b_aosoa.size();
    constexpr int am = ExtractArraySize<a_tuple_type>::value;
    constexpr int bm = ExtractArraySize<b_tuple_type>::value;

    if (an != bn) {
        throw std::invalid_argument("First dimension of a and b views do not match.");
    }

    if constexpr ((am == bm) && (am != 1))
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
                a.layout()->indexSpace( tag, a_entity_type(), Local() ),
                a_execution_space() );
        Kokkos::parallel_for(
            "ArrayOp::update", policy,
            KOKKOS_LAMBDA( const int i, const int j ) {
                out_slice( i, j ) = a_slice( i, j ) * b_slice( i, j );
            } );

        return out;
    }
    // If a and b have tuple size 1
    if constexpr ((am == bm) && (am == 1))
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->indexSpace( tag, a_entity_type(), Local() ),
            a_execution_space() );
        Kokkos::parallel_for(
            "ArrayOp::update", policy,
            KOKKOS_LAMBDA( const int i ) {
                out_slice( i ) = a_slice( i ) * b_slice( i );
            } );

        return out;
    }
    // If a has tuple size 1 but not b
    if constexpr ((am == 1) && (am < bm))
    {
        auto policy = Cabana::Grid::createExecutionPolicy(
            a.layout()->indexSpace( tag, a_entity_type(), Local(), Element() ),
            a_execution_space() );
        Kokkos::parallel_for(
            "ArrayOp::update", policy,
            KOKKOS_LAMBDA( const int i) {
                out_slice( i ) = a_slice( i ) * b_slice( i, 0 );
            } );

        return out;
    }
    else
    {
        throw std::invalid_argument("First array argument must have equal or smaller third dimension than second array argument.");
    }
}

} // end neamspace ArrayOp

} // end namespace Array

} // end namespace NuMesh

#endif // NUMESH_ARRAY_HPP