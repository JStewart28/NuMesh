# search prefix path
set(Cabana_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE STRING "Help cmake to find Cabana")

# check include
find_path(Cabana_INCLUDE_DIR NAMES Cabana_Core.hpp Cabana_Grid.hpp HINTS ${Cabana_PREFIX}/include)

# check lib
find_library(Cabana_LIBRARY NAMES cabana
	HINTS ${MPI_Advance_PREFIX}/lib)

# setup found
if (Cabana_INCLUDE_DIR AND Cabana_LIBRARY)
	set(Cabana_FOUND ON)
endif()

# handle QUIET/REQUIRED
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Cabana_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Cabana DEFAULT_MSG Cabana_INCLUDE_DIR Cabana_LIBRARY)

# Hide internal variables
mark_as_advanced(Cabana_INCLUDE_DIR Cabana_FOUND Cabana_LIBRARY Cabanae_PREFIX)
