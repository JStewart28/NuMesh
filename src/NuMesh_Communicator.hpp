#ifndef NUMESH_COMMUNICATOR_HPP
#define NUMESH_COMMUNICATOR_HPP


#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <memory>

#include <mpi.h>

#include <mpi_advance.h>

#include <limits>

int DEBUG_RANK = 0;

namespace NuMesh
{

template <class ExecutionSpace, class MemorySpace>
class Communicator
{
  public:
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    Communicator( const MPI_Comm comm )
            : _comm ( comm )
    {
        MPI_Comm_rank( _comm, &_rank );
        MPI_Comm_size( _comm, &_comm_size );

        MPIX_Info_init(&_xinfo);
        MPIX_Comm_init(&_xcomm, _comm);
    }

    ~Communicator()
    {
        MPIX_Info_free(&_xinfo);
        MPIX_Comm_free(&_xcomm);
    }

    template <class Tuple_t, class View_t1, class View_t2, class AoSoA_t>
    void gather(View_t1 sendvals_unpacked, AoSoA_t array, View_t2 vef_gid_start_d, int num_sends, int *owned_count, int *ghosted_count)
    {
        // Step 1: Count the number of edges needed from each other process
        using CounterView = Kokkos::View<int, device_type, Kokkos::MemoryTraits<Kokkos::Atomic>>;
        CounterView counter("counter");
        Kokkos::deep_copy(counter, 0);
        Kokkos::View<int*, device_type> sendcounts_unpacked("sendcounts_unpacked", _comm_size);   
        // Parallel reduction to count non -1 values in each row
        int num_cols = (int) array.size();
        Kokkos::parallel_for("count_sends", Kokkos::RangePolicy<execution_space>(0, _comm_size), KOKKOS_LAMBDA(const int i) {
            int count = 0;
            for (int j = 0; j < num_cols; j++) {
                if (sendvals_unpacked(i, j) != -1) {
                    count++;
                }
            }
            if (count > 0) counter()++;
            sendcounts_unpacked(i) = count;
            //if (rank == debug_rank) printf("R%d sendcounts_unpacked(%d): %d\n", rank, i, sendcounts_unpacked(i));
        });
        int send_nnz = 0;
        Kokkos::deep_copy(send_nnz, counter);
        // Pack serially for now, for simplicity. XXX - pack on GPU
        auto sendcounts_unpacked_h = Kokkos::create_mirror_view(sendcounts_unpacked);
        Kokkos::deep_copy(sendcounts_unpacked_h, sendcounts_unpacked);
        auto sendvals_unpacked_h = Kokkos::create_mirror_view(sendvals_unpacked);
        Kokkos::deep_copy(sendvals_unpacked_h, sendvals_unpacked);
        int* sendcounts = new int[send_nnz];
        int* dest = new int[send_nnz];
        int* sdispls = new int[send_nnz];
        int* sendvals = new int[num_sends];
        int idx = 0;
        sdispls[0] = 0;
        for (int i = 0; i < sendcounts_unpacked_h.extent(0); i++)
        {
            if (sendcounts_unpacked_h(i)) 
            {
                sendcounts[idx] = sendcounts_unpacked_h(i);
                dest[idx] = i; 
                if ((idx < send_nnz) && (idx > 0)) sdispls[idx] = sdispls[idx-1] + sendcounts[idx];
                // if (rank == DEBUG_RANK) printf("R%d: sendcounts[%d]: %d, dest[%d]: %d, sdispls[%d]: %d\n", rank,
                //     idx, sendcounts[idx], idx, dest[idx], idx, sdispls[idx]);
                idx++;
            }
        }

        // Pack sendvals
        // Pack starting at lowest rank in dest
        idx = 0;
        for (int d = 0; d < send_nnz; d++)
        {
            int dest_rank = dest[d];
            for (int lid = 0; lid < array.size(); lid++)
            {
                if (sendvals_unpacked_h(dest_rank, lid) != -1)
                {
                    sendvals[idx] = sendvals_unpacked_h(dest_rank, lid);
                    //if (rank == DEBUG_RANK) printf("R%d: sendvals[%d]: %d\n", rank, idx, sendvals[idx]);
                    idx++;
                }
            }
        }

        int recv_nnz, recv_size;
        // XXX - Assuming a there are as many recieved messages as sent messages
        int *rdispls = new int[send_nnz];
        int *src = new int[send_nnz];
        int *recvcounts = new int[send_nnz];
        int *recvvals = new int [num_sends];

        MPIX_Alltoallv_crs(send_nnz, num_sends, dest, sendcounts, sdispls, MPI_INT, sendvals, 
            &recv_nnz, &recv_size, src, recvcounts, rdispls, MPI_INT, recvvals, _xinfo, _xcomm);
        // printf("R%d: s_nnz, len: (%d, %d), dest: (%d, %d), sc: (%d, %d), sdispls: (%d, %d), svals: (%d, %d, %d, %d, %d, %d, %d, %d)\n",
        //     rank, send_nnz, num_sends, dest[0], dest[1], sendcounts[0], sendcounts[1],
        //     sdispls[0], sdispls[1], sendvals[0], sendvals[1], sendvals[2], sendvals[3], sendvals[4], sendvals[5], sendvals[6], sendvals[7]);
        // printf("R%d: r_nnz, len: (%d, %d), src: (%d, %d), rc: (%d, %d), rdispls: (%d, %d), rvals: (%d, %d, %d, %d, %d, %d, %d, %d)\n",
        //     rank, recv_nnz, recv_size, src[0], src[1], recvcounts[0], recvcounts[1],
        //     rdispls[0], rdispls[1], recvvals[0], recvvals[1], recvvals[2], recvvals[3], recvvals[4], recvvals[5], recvvals[6], recvvals[7]);

        // Parse the received values to send the correct edges to the processes that requested them.
        Kokkos::View<typename AoSoA_t::tuple_type*, Kokkos::HostSpace> send_edges(Kokkos::ViewAllocateWithoutInitializing("send_edges"), recv_size);
        Kokkos::View<typename AoSoA_t::tuple_type*, Kokkos::HostSpace> recv_edges(Kokkos::ViewAllocateWithoutInitializing("recv_edges"), num_sends);
        MPI_Request* requests = new MPI_Request[send_nnz+recv_nnz];
        Cabana::Tuple<Tuple_t> t;
        std::pair<std::size_t, std::size_t> range = { 0, 0 };
        for (int s = 0; s < recv_nnz; s++)
        {
            int send_to = src[s], send_count = recvcounts[s], displs = rdispls[s];
            int counter = displs;
            while (counter < (displs+send_count))
            {
                // if (rank == DEBUG_RANK) printf("R%d: send_to: %d, accessing recvvals[%d]: %d\n", rank, send_to, counter, recvvals[counter]);
                int e_gid = recvvals[counter];
                int e_lid = e_gid - vef_gid_start_d(_rank, 1);
                
                // Get this edge from the edge AoSoA and set it in the buffer
                t = array.getTuple(e_lid);
                send_edges(counter) = t;
                counter++;
            }
            // Post this send
            // MPI_Isend(send_buffer.data(), num_elements, MPI_INT, 1, tag, MPI_COMM_WORLD, &requests[0]);
            range.first = displs; range.second = range.first + send_count;
            auto send_subview = Kokkos::subview(send_edges, range);
            if (_rank == DEBUG_RANK)
            {
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(0)));
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(1)));
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(2)));
                // printf("R%d: sending from send_edges(%d), send_to: %d, gid: %d\n", rank, displs, send_to, Cabana::get<S_E_GID>(send_subview(3)));

            }
            MPI_Isend(send_subview.data(), sizeof(t)*send_subview.size(), MPI_BYTE, send_to, _rank, _comm, &requests[s]);
        }

        // Post receives
        for (int s = 0; s < send_nnz; s++)
        {
            int recv_from = dest[s], recv_count = sendcounts[s], displs = sdispls[s];
            range.first = displs; range.second = range.first + recv_count;
            auto recv_subview = Kokkos::subview(recv_edges, range);
            MPI_Irecv(recv_subview.data(), sizeof(t)*recv_subview.size(), MPI_BYTE, recv_from, recv_from, _comm, &requests[recv_nnz+s]);
        }

        MPI_Waitall(send_nnz+recv_nnz, requests, MPI_STATUSES_IGNORE);
        delete[] requests;

        // if (_rank == 0)
        // {
        //     for (int i = 0; i < recv_edges.extent(0); i++)
        //     {
        //         e_tuple = recv_edges(i);
        //         //if (rank == 0) printf("R%d: got e_gid: %d\n", _rank, Cabana::get<S_E_GID>(e_tuple));
        //     }
        // }

        // Put the ghosted edges in the local edge aosoa
        *ghosted_count = recv_edges.extent(0);
        array.resize(*owned_count+*ghosted_count);
        int owned_count1 = *owned_count;
        Kokkos::parallel_for("add_ghosted_edges", Kokkos::RangePolicy<execution_space>(0, *ghosted_count), KOKKOS_LAMBDA(const int i) {
            int lid = i+owned_count1;
            array.setTuple(lid, recv_edges(i));
        });
        
        delete[] dest;
        delete[] sendcounts;
        delete[] sdispls;
        delete[] sendvals;

        delete[] src;
        delete[] recvcounts;
        delete[] rdispls;
        delete[] recvvals;
    }


  private:
    const MPI_Comm _comm;
    MPIX_Comm* _xcomm;
    MPIX_Info* _xinfo;

    int _rank, _comm_size;

};

/**
 *  Return a shared pointer to a Communcation object
 */
template <class ExecutionSpace, class MemorySpace>
auto createCommunicator( const MPI_Comm comm )
{
    return std::make_shared<Communicator<ExecutionSpace, MemorySpace>>(
        comm );
}

} // end namespace NUMesh


#endif // NUMESH_COMMUNICATOR_HPP