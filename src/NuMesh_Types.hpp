#ifndef NUMESH_TYPES_HPP
#define NUMESH_TYPES_HPP


namespace NuMesh
{

// Design ideas for the NuMesh::Array taken from Cabana::Grid:Array

//---------------------------------------------------------------------------//
// Enums
//---------------------------------------------------------------------------//

enum BoundaryType {PERIODIC = 0, FREE = 1};

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

} // end namespace NuMesh

#endif // NUMESH_TYPES_HPP