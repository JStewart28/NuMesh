#ifndef HELPERS_HPP
#define HELPERS_HPP

template <class l2g_type, class View>
void printView(l2g_type local_L2G, int rank, View z, int option, int DEBUG_X, int DEBUG_Y)
{
    
    int dims = z.extent(2);

    std::array<long, 2> rmin, rmax;
    for (int d = 0; d < 2; d++) {
        rmin[d] = local_L2G.local_own_min[d];
        rmax[d] = local_L2G.local_own_max[d];
    }
    Cabana::Grid::IndexSpace<2> remote_space(rmin, rmax);

    Kokkos::parallel_for("print views",
        Cabana::Grid::createExecutionPolicy(remote_space, Kokkos::DefaultHostExecutionSpace()),
        KOKKOS_LAMBDA(int i, int j) {
        
        int local_li[2] = {i, j};
        int local_gi[2] = {0, 0};   // global i, j
        local_L2G(local_li, local_gi);
        if (option == 1){
            if (dims == 3) {
                printf("R%d %d %d %d %d %.12lf %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
            }
            else if (dims == 2) {
                printf("R%d %d %d %d %d %.12lf %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
            }
        }
        else if (option == 2) {
            if (local_gi[0] == DEBUG_X && local_gi[1] == DEBUG_Y) {
                if (dims == 3) {
                    printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1), z(i, j, 2));
                }   
                else if (dims == 2) {
                    printf("R%d: %d: %d: %d: %d: %.12lf: %.12lf\n", rank, local_gi[0], local_gi[1], i, j, z(i, j, 0), z(i, j, 1));
                }
            }
        }
    });
}


#endif // HELPERS_HPP