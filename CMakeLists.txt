# Portions of this CMake code copied from the Cabana library:
# https://github.com/ECP-copa/Cabana

# project settings
cmake_minimum_required(VERSION 3.18)

project(Tessera LANGUAGES C CXX VERSION 0.1.0)

# Default to C++17 so we can use if constexpr and [[maybe_unused]]
# and so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++20" CACHE STRING "")
endif()

# If the user doesn't provide a build type default to release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

# Disable verbose makefiles
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)

# use gnu standard install directories
include(GNUInstallDirs)
set(Tessera_INSTALL_PACKAGEDIR "${CMAKE_INSTALL_DATADIR}/cmake/Tessera" CACHE PATH "Install location of CMake target files")

include(FeatureSummary)

# add local cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# find kokkos
find_package(Kokkos 4.1 REQUIRED)

# set supported kokkos devices
set(TESSERA_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)

# check user required kokkos device types
foreach(_device ${TESSERA_SUPPORTED_DEVICES})
  option(Tessera_REQUIRE_${_device} "Build Tessera with required Kokkos ${device} support" OFF)
  if(Tessera_REQUIRE_${_device})
    kokkos_check( DEVICES ${_device} )
  endif()
endforeach()

# ensure that we can use lambdas
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

# find cabana
find_package(Cabana REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()

# Add VTK include directories, required for mesh I/O
find_package(VTK REQUIRED)
find_package(SEACASIoss REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})
find_package(OpenGL REQUIRED) # Needed to correctly link on Tioga

# find Clang Format
find_package( CLANG_FORMAT 14 )


#------------------------------------------------------------------------------#
# Tests and Documentation
# Bring up the BLT tools
# We use LLNL BLT to help with building, linking, and testing on various HPC 
# systems. Before bringing in BLT, we set a variety of flags whch tell it the 
# featurs we're interested in.
#------------------------------------------------------------------------------#
option(Tessera_ENABLE_TESTING "Build tests" OFF)
add_feature_info(Tests Tessera_ENABLE_TESTING "Build unit tests (requires BLT)")
if(Tessera_ENABLE_TESTING)
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
  add_subdirectory(tests)
endif()

# enable doxygen
option(Tessera_ENABLE_DOXYGEN "Build documentation" OFF)
add_feature_info(Documentation Tessera_ENABLE_DOXYGEN "Build documentation (requires Doxygen)")
if(Tessera_ENABLE_DOXYGEN)
  find_package(Doxygen REQUIRED)
  doxygen_add_docs(doxygen src src)
endif()

##---------------------------------------------------------------------------##
## Libraries and Examples
##---------------------------------------------------------------------------##

if(MPI_FOUND)
  add_subdirectory(src)
else()
  message(FATAL_ERROR "Tessera requires MPI")
endif()

option(Tessera_ENABLE_EXAMPLES "Build tutorial examples" OFF)
add_feature_info(Examples Tessera_ENABLE_EXAMPLES "Build tutorial examples")
if(Tessera_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

#------------------------------------------------------------------------------#
# Architecture - copied from Cabana
#------------------------------------------------------------------------------#
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Tessera_BUILD_MARCH "" CACHE STRING "Arch to use with -march= (if empty CMake will try to use 'native') in release build and only release build")

  # Try -march first. On platforms that don't support it, GCC will issue
  # a hard error, so we'll know not to use it.
  if(Tessera_BUILD_MARCH)
    set(INTERNAL_Tessera_BUILD_MARCH ${Tessera_BUILD_MARCH})
  else()
    set(INTERNAL_Tessera_BUILD_MARCH "native")
  endif()

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=${INTERNAL_Tessera_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_Tessera_BUILD_MARCH}")
  elseif(Tessera_BUILD_MARCH)
    message(FATAL_ERROR "The flag -march=${INTERNAL_Tessera_BUILD_MARCH} is not supported by the compiler")
  else()
    unset(INTERNAL_Tessera_BUILD_MARCH)
  endif()
endif()

##---------------------------------------------------------------------------##
## Code coverage testing - copied from Cabana
##---------------------------------------------------------------------------##
option(Tessera_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(Tessera_ENABLE_COVERAGE_BUILD)
  message(STATUS "Enabling coverage build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  option(WITH_ASAN "Build with address sanitizer" OFF)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
  option(WITH_MSAN "Build with memory sanitizer (experimental; requires a memory-sanitized Python interpreter)" OFF)
endif()

if(WITH_ASAN AND WITH_MSAN)
  message( FATAL_ERROR "Address sanitizer and memory sanitizer cannot be enabled simultaneously")
endif()
if(WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

if(WITH_MSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory")
endif()

##---------------------------------------------------------------------------##
## Print the revision number to stdout - copied from Cabana
##---------------------------------------------------------------------------##
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  EXECUTE_PROCESS(
    COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE   Tessera_GIT_COMMIT_HASH
    )
ELSE()
  SET(Tessera_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "Tessera Revision = '${Tessera_GIT_COMMIT_HASH}'")

##---------------------------------------------------------------------------##
## Package Configuration - copied from Cabana
##---------------------------------------------------------------------------##
write_basic_package_version_file("TesseraConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TesseraConfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/TesseraConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TesseraConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/TesseraConfigVersion.cmake"
  DESTINATION ${Tessera_INSTALL_PACKAGEDIR})

##---------------------------------------------------------------------------##
## Clang Format - copied from Cabana
##---------------------------------------------------------------------------##
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp examples/*cpp examples/*hpp cmake/*cpp cmake/*hpp)
  add_custom_target(cabana-format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
