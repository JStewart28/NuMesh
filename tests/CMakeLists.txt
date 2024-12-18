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
    
                 
    # blt_add_executable( NAME     tstBoundaryCondition
    #              SOURCES tstBoundaryCondition.cpp
    #              HEADERS tstBoundaryCondition.hpp TestingBase.hpp
    #              DEPENDS_ON beatnik gtest
    #              OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    # blt_add_test(NAME BoundaryConditionTests
    #              COMMAND tstBoundaryCondition)

endif()
