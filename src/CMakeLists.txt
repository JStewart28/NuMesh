set(HEADERS
 Tessera_Array.hpp
 Tessera_Core.hpp
 Tessera_Halo.hpp
 Tessera_Maps.hpp
 Tessera_Mesh.hpp
 Tessera_Types.hpp
 Tessera_Utils.hpp
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
       
add_library( Tessera INTERFACE ${SOURCES} )
target_link_libraries( Tessera INTERFACE ${DEPENDS_ON} )
target_link_libraries( Tessera INTERFACE ${VTK_LIBRARIES} )
target_include_directories( Tessera INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
	  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS Tessera LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
