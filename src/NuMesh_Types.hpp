#ifndef NUMESH_TYPES_HPP
#define NUMESH_TYPES_HPP

#include <Cabana_Core.hpp>
#include <type_traits>
#include <tuple>

namespace NuMesh
{

// Design ideas for the NuMesh::Array taken from Cabana::Grid:Array

//---------------------------------------------------------------------------//
// Enums
//---------------------------------------------------------------------------//

enum BoundaryType {FREE = 0, PERIODIC = 1};

//---------------------------------------------------------------------------//
// Entity type tags.
//---------------------------------------------------------------------------//

/*!
  \brief Mesh vertex tag.
*/
struct Vertex {};

/*!
  \brief Mesh edge tag.
*/
struct Edge {};

/*!
  \brief Mesh face tag.
*/
struct Face {};

//---------------------------------------------------------------------------//
// Decomposition tags.
//---------------------------------------------------------------------------//

/*!
  \brief Owned decomposition tag.
*/
struct Own {};

/*!
  \brief Ghosted decomposition tag.
*/
struct Ghost {};

//---------------------------------------------------------------------------//
// Index type tags.
//---------------------------------------------------------------------------//

/*!
  \brief Local index tag.
*/
struct Local {};

/*!
  \brief Global index tag.
*/
struct Global {};

/*!
  \brief Element index tag.
*/
struct Element {};

//---------------------------------------------------------------------------//
// Mesh type tags.
//---------------------------------------------------------------------------//

/*!
  \brief Unstructured 2D surface mesh tag.
*/
template <class Scalar, std::size_t NumSpaceDim = 2>
struct Unstructured2DMesh
{
    //! Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    //! Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

/*!
  \brief Unstructured 3D surface mesh tag.
*/
template <class Scalar, std::size_t NumSpaceDim = 3>
struct Unstructured3DMesh
{
    //! Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    //! Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

//! Helpers to determine is T is a Cabana::MemberTypes<...> type
// Primary template: Assume false
template <typename T, typename Enable = void>
struct IsCabanaMemberTypes : std::false_type {};

// Specialization for Cabana::MemberTypes<Ts...>
template <typename... Ts>
struct IsCabanaMemberTypes<Cabana::MemberTypes<Ts...>> : std::true_type {};

// Primary template: Invalid case (multiple or empty types)
template <typename Tuple, typename Enable = void>
struct ExtractSingleType;

// Specialization for a tuple with exactly one type
template <typename T>
struct ExtractSingleType<std::tuple<T>>
{
    using type = T;
};

//! Helpers to extract the type T in a Tuple<T>
// Specialization for invalid cases (empty or multiple types)
template <typename T, typename U, typename... Rest>
struct ExtractSingleType<std::tuple<T, U, Rest...>>
{
    static_assert(sizeof...(Rest) == 0, 
                  "ExtractSingleType can only be used with a single-type tuple.");
};

// Specialization for empty tuple (should not occur)
template <>
struct ExtractSingleType<std::tuple<>>
{
    static_assert(sizeof(std::tuple<>) != 0, 
                  "ExtractSingleType cannot be used with an empty tuple.");
};

//! Helpers to extract base types of Cabana::MemberTypes<...>
// General template (for non-Cabana::MemberTypes)
template <typename T, typename Enable = void>
struct ExtractBaseTypes
{
    using type = std::tuple<T>;  // Default case: Wrap T in a tuple
};

// Specialization for array types in Cabana::MemberTypes<T[N]>
template <typename T, std::size_t N>
struct ExtractBaseTypes<Cabana::MemberTypes<T[N]>>
{
    using type = std::tuple<T>;  // Extract just 'T' from 'T[N]'
};

// Specialization for general Cabana::MemberTypes (handles multiple types)
template <typename... Ts>
struct ExtractBaseTypes<Cabana::MemberTypes<Ts...>>
{
    using type = std::tuple<std::remove_extent_t<Ts>...>;  // Extract base types
};

//! Utility to check if a tuple has exactly one unique base type
template <typename Tuple>
struct HasSingleUniqueType;

// Specialization for empty tuple (should not occur in practice)
template <>
struct HasSingleUniqueType<std::tuple<>>
{
    static constexpr bool value = false;
};

// Specialization for single-type tuple
template <typename T>
struct HasSingleUniqueType<std::tuple<T>>
{
    static constexpr bool value = true;
};

// Specialization for multi-type tuple (not allowed)
template <typename T, typename U, typename... Rest>
struct HasSingleUniqueType<std::tuple<T, U, Rest...>>
{
    static constexpr bool value = false;
};

//! Main check function for Cabana::MemberTypes
template <typename MemberTypes>
struct IsSinglePartMemberTypes
{
    // Ensure we have a fully resolved type before using it
    using extracted_base_types = typename ExtractBaseTypes<MemberTypes>::type;
    
    // Static check for a single unique base type
    static constexpr bool value = HasSingleUniqueType<extracted_base_types>::value;
};

//! Extract array size from Cabana::MemberTypes<T[N]> or return 1 for scalars.
template <typename T>
struct ExtractArraySize
{
    static constexpr std::size_t value = 1; // Default case for scalars
};

// Specialization for array types in Cabana::MemberTypes<T[N]>
template <typename T, std::size_t N>
struct ExtractArraySize<Cabana::MemberTypes<T[N]>>
{
    static constexpr std::size_t value = N;
};

// Specialization for general Cabana::MemberTypes<T>
template <typename T>
struct ExtractArraySize<Cabana::MemberTypes<T>>
{
    static constexpr std::size_t value = 1;
};

// Specialization for multiple types (invalid case, prevents compilation)
template <typename... Ts>
struct ExtractArraySize<Cabana::MemberTypes<Ts...>>
{
    static_assert(sizeof...(Ts) == 1, "ExtractArraySize can only be used with a single Cabana::MemberType.");
};

} // end namespace NuMesh

#endif // NUMESH_TYPES_HPP