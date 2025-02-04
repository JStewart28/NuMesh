if (ENABLE_TESTS)

    blt_add_executable(NAME tstMesh2D
                       SOURCES tstMesh2D.cpp 
                       HEADERS tstMesh2D.hpp
                       DEPENDS_ON numesh gtest
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME MeshTests
                 COMMAND tstMesh2D)

    blt_add_executable(NAME tstMaps
                       SOURCES tstMaps.cpp 
                       HEADERS tstMaps.hpp
                       DEPENDS_ON numesh gtest
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME MapTests
                 COMMAND tstMaps)

    blt_add_executable(NAME tstHalo
                       SOURCES tstHalo.cpp 
                       HEADERS tstHalo.hpp
                       DEPENDS_ON numesh gtest
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME HaloTests
                 COMMAND tstHalo)

    blt_add_executable(NAME tstArray
                       SOURCES tstArray.cpp 
                       HEADERS tstArray.hpp
                       DEPENDS_ON numesh gtest
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME ArrayTests
                 COMMAND tstArray)

endif()
