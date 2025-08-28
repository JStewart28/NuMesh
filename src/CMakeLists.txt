configure_file(Tessera_Config.hpp.cmakein Tessera_Config.hpp)

set(HEADERS_PUBLIC
 Tessera_Array.hpp
 Tessera_Core.hpp
 Tessera_Halo.hpp
 Tessera_Maps.hpp
 Tessera_Mesh.hpp
 Tessera_Types.hpp
 Tessera_Utils.hpp
  )

# Currently no impl headers
# set(HEADERS_IMPL
#   impl/Tessera_CartesianGrid.hpp
#   )

add_library(Tessera INTERFACE)

# Require minimum of C++17
set_target_properties(Tessera PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)

target_link_libraries(Tessera INTERFACE Kokkos::kokkos)

target_link_libraries(Tessera INTERFACE MPI::MPI_CXX)

target_link_libraries(Tessera INTERFACE Cabana::Core Cabana::Grid)

target_include_directories(Tessera INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

install(TARGETS Tessera
  EXPORT Tessera_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT Tessera_Targets
  FILE Tessera_Targets.cmake
  NAMESPACE Tessera::
  DESTINATION ${Tessera_INSTALL_PACKAGEDIR})

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${HEADERS_IMPL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/impl/)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Tessera_Config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
