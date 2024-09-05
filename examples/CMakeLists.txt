include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

blt_add_executable( NAME build_from_grid
                SOURCES build_from_grid.cpp 
                OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY})

# Tell CMAKE we need to link with these libraries. Our "FindMPI_Advance" script
# does not define an explicit target, but it does set the variable below.
# We also need to link with MPI, which is done through the target they provide
# (It's not C++ MPI, but rather the libraries for a C++ program using MPI)
target_link_libraries( build_from_grid PRIVATE ${MPI_Advance_LIBRARY} MPI::MPI_CXX )

# Finally, we also need to tell CMAKE where the library header is. 
target_include_directories( build_from_grid PRIVATE ${MPI_Advance_INCLUDE_DIR} )

target_link_libraries( build_from_grid PRIVATE hpctests )


install(TARGETS build_from_grid DESTINATION ${CMAKE_INSTALL_BINDIR})
