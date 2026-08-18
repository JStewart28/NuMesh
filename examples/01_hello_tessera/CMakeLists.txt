############################################################################
# Copyright (c) 2024, JStewart28                                           #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Tessera library. Tessera is distributed under a #
# BSD 3-Clause license. For the licensing terms see the LICENSE file in   #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

add_executable(hello_tessera hello_tessera.cpp)
target_link_libraries(hello_tessera PRIVATE Tessera::Tessera)

if(Tessera_INSTALL_EXAMPLES)
    install(TARGETS hello_tessera DESTINATION ${CMAKE_INSTALL_DATADIR}/Tessera/examples)
endif()
