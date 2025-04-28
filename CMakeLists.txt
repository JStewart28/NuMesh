# project settings
cmake_minimum_required(VERSION 3.18)

project(NuMesh LANGUAGES C CXX VERSION 0.1.0)

# Default to C++17 so we can use if constexpr and [[maybe_unused]]
# and so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++20" CACHE STRING "")
endif()

include(GNUInstallDirs)

# Update the path that CMAKE will search for Find_XXX.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Enable debugging for all builds
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)

##---------------------------------------------------------------------------##
## Bring up the BLT tools
## We use LLNL BLT to help with building, linking, and testing on various HPC 
## systems. Before bringing in BLT, we set a variety of flags whch tell it the 
## featurs we're interested in.
##---------------------------------------------------------------------------##
set(ENABLE_TESTS ON)
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT git submodule is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Disable google test pthreads in BLT to support to ensure we compile in 
# some environments, as some compilers (e.g. craycc) and BLT aren't communicating
# on linking against pthreads.
set(gtest_disable_pthreads ON CACHE BOOL "")
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#------------------------------------------------------------------------------#
# Dependencies
#------------------------------------------------------------------------------#

# standard dependency macro
# Copied from the macro used in Cabana's CMakeLists.txt:
# https://github.com/ECP-copa/Cabana/blob/master/CMakeLists.txt
macro(Numesh_add_dependency)
  cmake_parse_arguments(Numesh_DEPENDENCY "" "PACKAGE;VERSION" "COMPONENTS" ${ARGN})
  find_package( ${Numesh_DEPENDENCY_PACKAGE} ${Numesh_DEPENDENCY_VERSION} QUIET COMPONENTS ${Numesh_DEPENDENCY_COMPONENTS} )
  string(TOUPPER "${Numesh_DEPENDENCY_PACKAGE}" Numesh_DEPENDENCY_OPTION )
  option(
    Numesh_REQUIRE_${Numesh_DEPENDENCY_OPTION}
    "Require Numesh to build with ${Numesh_DEPENDENCY_PACKAGE} support" ${Numesh_DEPENDENCY_PACKAGE}_FOUND)
  if(Numesh_REQUIRE_${Numesh_DEPENDENCY_OPTION})
    find_package( ${Numesh_DEPENDENCY_PACKAGE} ${Numesh_DEPENDENCY_VERSION} REQUIRED COMPONENTS ${Numesh_DEPENDENCY_COMPONENTS} )
  endif()
  set(Numesh_ENABLE_${Numesh_DEPENDENCY_OPTION} ${${Numesh_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# We also need MPI for this example
find_package(MPI REQUIRED)

# find mpiadvance
Numesh_add_dependency( PACKAGE mpiadvance )
set_package_properties(MPIADVANCE PROPERTIES TYPE OPTIONAL PURPOSE "Advanced MPI communication support")

# Add VTK include directories
Numesh_add_dependency( PACKAGE VTK )
set_package_properties(VTK PROPERTIES TYPE REQUIRED PURPOSE "Mesh I/O")
# find_package(VTK REQUIRED)
find_package(SEACASIoss REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
find_package(OpenGL REQUIRED) # Needed to correctly link on Tioga

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})


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

# tests
add_subdirectory(tests)

# Add a target for formatting the code using Clang
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp tests/*.hpp tests/*.cpp examples/*.cpp examples/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

include(CMakePackageConfigHelpers)

# Install the actual target(s) and register them for export
install(TARGETS numesh
        EXPORT numeshTargets
        DESTINATION lib)

# Export the targets into a CMake package
install(EXPORT numeshTargets
        NAMESPACE numesh::
        DESTINATION lib/cmake/numesh)

# Generate the Config and Version files
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/numeshConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/numeshConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/numeshConfig.cmake"
  INSTALL_DESTINATION lib/cmake/numesh)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/numeshConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/numeshConfigVersion.cmake"
  DESTINATION lib/cmake/numesh)
