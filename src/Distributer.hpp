#ifndef DISTRIBUTER_HPP
#define DISTRIBUTER_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

#include <mpi_advance.h>

#include <limits>

#ifndef AOSOA_SLICE_INDICES
#define AOSOA_SLICE_INDICES 1
#endif

int DEBUG_RANK = 0;

namespace NuMesh
{

namespace impl
{
/**
 * General design for this function taken from the distributeData function
 * in Cabana_Distributor.hpp in the Cabana library:
 * https://github.com/ECP-copa/Cabana/ 
 */
template <class ExecutionSpace, class MemorySpace, class AoSoA_t>
void distributeData(ExecutionSpace, MemorySpace, const AoSoA_t& send_array,
    AoSoA_t& recv_array, MPI_comm comm)
    // typename std::enable_if<( is_distributor<Distributor_t>::value &&
    //                           is_aosoa<AoSoA_t>::value ),
    //                         int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "NUMesh::impl::distributeData" );

    // static_assert( is_accessible_from<typename Distributor_t::memory_space,
    //                                   ExecutionSpace>{},
    //                "" );

    // Get the MPI rank we are currently on.
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Allocate a send buffer.
    std::size_t num_send = distributor.totalNumExport() - num_stay;
    Kokkos::View<typename AoSoA_t::tuple_type*, MemorySpace>
        send_buffer( Kokkos::ViewAllocateWithoutInitializing(
                         "distributor_send_buffer" ),
                     num_send );

    // Allocate a receive buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Distributor_t::memory_space>
        recv_buffer( Kokkos::ViewAllocateWithoutInitializing(
                         "distributor_recv_buffer" ),
                     distributor.totalNumImport() );

    // Get the steering vector for the sends.
    auto steering = distributor.getExportSteering();

    // Gather the exports from the source AoSoA into the tuple-contiguous send
    // buffer or the receive buffer if the data is staying. We know that the
    // steering vector is ordered such that the data staying on this rank
    // comes first.
    auto build_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto tpl = src.getTuple( steering( i ) );
        if ( i < num_stay )
            recv_buffer( i ) = tpl;
        else
            send_buffer( i - num_stay ) = tpl;
    };
    Kokkos::RangePolicy<ExecutionSpace> build_send_buffer_policy(
        0, distributor.totalNumExport() );
    Kokkos::parallel_for( "Cabana::Impl::distributeData::build_send_buffer",
                          build_send_buffer_policy, build_send_buffer_func );
    Kokkos::fence();

    // The distributor has its own communication space so choose any tag.
    const int mpi_tag = 1234;

    // Post non-blocking receives.
    std::vector<MPI_Request> requests;
    requests.reserve( num_n );
    std::pair<std::size_t, std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second = recv_range.first + distributor.numImport( n );

        if ( ( distributor.numImport( n ) > 0 ) &&
             ( distributor.neighborRank( n ) != my_rank ) )
        {
            auto recv_subview = Kokkos::subview( recv_buffer, recv_range );

            requests.push_back( MPI_Request() );

            MPI_Irecv( recv_subview.data(),
                       recv_subview.size() *
                           sizeof( typename AoSoA_t::tuple_type ),
                       MPI_BYTE, distributor.neighborRank( n ), mpi_tag,
                       distributor.comm(), &( requests.back() ) );
        }

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t, std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        if ( ( distributor.numExport( n ) > 0 ) &&
             ( distributor.neighborRank( n ) != my_rank ) )
        {
            send_range.second = send_range.first + distributor.numExport( n );

            auto send_subview = Kokkos::subview( send_buffer, send_range );

            MPI_Send( send_subview.data(),
                      send_subview.size() *
                          sizeof( typename AoSoA_t::tuple_type ),
                      MPI_BYTE, distributor.neighborRank( n ), mpi_tag,
                      distributor.comm() );

            send_range.first = send_range.second;
        }
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( requests.size() );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    if ( MPI_SUCCESS != ec )
        throw std::logic_error( "Failed MPI Communication" );

    // Extract the receive buffer into the destination AoSoA.
    auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        dst.setTuple( i, recv_buffer( i ) );
    };
    Kokkos::RangePolicy<ExecutionSpace> extract_recv_buffer_policy(
        0, distributor.totalNumImport() );
    Kokkos::parallel_for( "Cabana::Impl::distributeData::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier(comm);
}

} // end namespace impl

} // end namespace NUMesh


#endif // DISTRIBUTER_HPP