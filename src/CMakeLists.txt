set(HEADERS
 NuMesh_Array.hpp
 NuMesh_Core.hpp
 NuMesh_Halo.hpp
 NuMesh_Maps.hpp
 NuMesh_Mesh.hpp
 NuMesh_Types.hpp
 NuMesh_Utils.hpp
  )

#set(SOURCES
#  Mesh.cpp
#  ProblemManager.cpp
#  )

set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  MPI::MPI_CXX
  )
 
set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
       
add_library( numesh INTERFACE ${SOURCES} )
target_link_libraries( numesh INTERFACE ${DEPENDS_ON} )
target_include_directories( numesh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
	  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS numesh LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
