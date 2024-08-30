set(HEADERS
 helpers.hpp
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
       
add_library( hpctests INTERFACE ${SOURCES} )
target_link_libraries( hpctests INTERFACE ${DEPENDS_ON} )
target_include_directories( hpctests INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
	  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS hpctests LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
