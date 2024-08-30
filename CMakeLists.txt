# project settings
cmake_minimum_required(VERSION 3.18)

project(NuMesh LANGUAGES C CXX VERSION 0.1.0)

include(GNUInstallDirs)

# Update the path that CMAKE will search for Find_XXX.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Tell CMAKE to find MPIAdvance
find_package(MPI_Advance REQUIRED)

# We also need MPI for this example
find_package(MPI REQUIRED)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Include MPI_Advance
# set( MPI_Advance_PREFIX "~/executables/mpi_advance" CACHE PATH "")
# find_package(MPI_Advance REQUIRED)

##---------------------------------------------------------------------------##
## Get the dependent packages we need: Cabana, Silo, and ClangFormat         ##
##---------------------------------------------------------------------------##
find_package(Cabana REQUIRED COMPONENTS Cabana::Cajita Cabana::cabanacore)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()

# Get Clang Format to use for making sure the resulting code is
# properly formatted
find_package(CLANG_FORMAT 10)

# library
add_subdirectory(src)

# examples
add_subdirectory(examples)

# Add a target for formatting the code using Clang
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp tests/*.hpp tests/*.cpp examples/*.cpp examples/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
