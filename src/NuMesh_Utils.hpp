#ifndef NUMESH_UTILS_HPP
#define NUMESH_UTILS_HPP

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

namespace NuMesh
{

namespace Utils
{

//---------------------------------------------------------------------------//
/*!
  Utility functions
*/

template <std::size_t Size, class Scalar>
auto vectorToArray( std::vector<Scalar> vector )
{
    Kokkos::Array<Scalar, Size> array;
    for ( std::size_t i = 0; i < Size; ++i )
        array[i] = vector[i];
    return array;
}

bool isPerfectSquare(int number) {
    if (number < 0) {
        return false; // Negative numbers can't be perfect squares
    }

    int sqrtNumber = static_cast<int>(std::sqrt(number)); // Get the integer part of the square root
    return sqrtNumber * sqrtNumber == number;
}

/**
 * Returns the rank that owns the entity
 * given its global ID
 */
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
int owner_rank(const int index, const int gid, const vef_gid_start_array vef_start)
{
    assert(index < vef_start.extent(1));

    int owner = -1;
    for (int r = 0; r < (int) vef_start.extent(0); r++)
    {
        if (gid >= vef_start(r, index))
        {
            owner = r;
        }
    }
    return owner;
}
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
int owner_rank(Vertex, const int gid, const vef_gid_start_array vef_start)
{
    return owner_rank(0, gid, vef_start);
}
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
int owner_rank(Edge, const int gid, const vef_gid_start_array vef_start)
{
    return owner_rank(1, gid, vef_start);
}
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
int owner_rank(Face, const int gid, const vef_gid_start_array vef_start)
{
    return owner_rank(2, gid, vef_start);
}

/**
 * Update a global ID given the old and new starting array for
 * global ID starts for each rank.
 */
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
void updateGlobalID(const int index, int* gid, const vef_gid_start_array n, const vef_gid_start_array o)
{
    int orank = owner_rank(index, *gid, o);
    int diff = n(orank, index) - o(orank, index);
    *gid += diff;
}
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
void updateGlobalID(Vertex, int* gid, const vef_gid_start_array n, const vef_gid_start_array o)
{
    updateGlobalID(0, gid, n, o);
}
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
void updateGlobalID(Edge, int* gid, const vef_gid_start_array n, const vef_gid_start_array o)
{
    updateGlobalID(1, gid, n, o);
}
template <class vef_gid_start_array>
KOKKOS_INLINE_FUNCTION
void updateGlobalID(Face, int* gid, const vef_gid_start_array n, const vef_gid_start_array o)
{
    updateGlobalID(2, gid, n, o);
}

/**
 * Get the local ID of a ghosted vert/edge/face given its global ID
 * Performs linear search on the GIDs between range start and end.
 * Cannot assume the AoSoA is sorted for GID
 * Returns -1 if the global ID is not found
 */
template <class Slice_t>
KOKKOS_INLINE_FUNCTION
int get_lid(Slice_t& slice, int gid, int start, int end)
{
    for (int i = start; i < end; i++)
    {
        int val = slice(i);
        if (val == gid)
        {
            return i;
        }
    }
    return -1;
}

/**
 * Find the local ID of an edge given its endpoints and
 * a local ID range to search in. Performs a linear search
 * over the domain.
 * 
 * XXX - can this function be optimized?
 * 
 * Returns the local ID, or -1 if not found
 */
template <class Slice_t>
KOKKOS_INLINE_FUNCTION
int find_edge(Slice_t& edge_vert_slice, int start, int end, int v0, int v1)
{
    for (int i = start; i < end; i++)
    {
        int iv0 = edge_vert_slice(i, 0);
        int iv1 = edge_vert_slice(i, 1);
        // printf("i%d: (%d, %d), actual (%d, %d)\n", i, iv0, iv1, v0, v1);
        if ( ((iv0 == v0) && (iv1 == v1)) || ((iv0 == v1) && (iv1 == v0)) )
        {
            return i;
        }
    }
    return -1;
}

} // end namespace Utils
} // end namespce NuMesh


#endif // NUMESH_UTILS_HPP