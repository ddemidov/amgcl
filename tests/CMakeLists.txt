#----------------------------------------------------------------------------
# Test coverage
#----------------------------------------------------------------------------
option(TEST_COVERAGE "Analyze test coverage with gcov/lcov" OFF)

if(TEST_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")

    # Resets coverage statistics
    add_custom_target(coverage_reset
        COMMAND lcov --zerocounters --directory .
        COMMAND lcov --capture --initial --directory . --base-directory "${CMAKE_SOURCE_DIR}/amgcl" --no-external --output-file coverage.info
        VERBATIM
        )

    # Converts accumulated coverage statistics into coverage/index.html
    # Run
    #   make tests
    # for all OpenCL platforms after
    #   make coverage_reset
    # and before
    #   make coverage
    add_custom_target(coverage
        COMMAND lcov --directory . --base-directory "${CMAKE_SOURCE_DIR}/amgcl" --no-external --capture --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr*' -o coverage.info
        COMMAND genhtml coverage.info --output-directory coverage
        VERBATIM
        )
endif()

#----------------------------------------------------------------------------
# The tests
#----------------------------------------------------------------------------
function(add_amgcl_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} ${Boost_LIBRARIES})
    if (VEXCL_ROOT)
        target_link_libraries(${TEST_NAME} ${VEXCL_BACKEND_LIBRARIES})
    endif()
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

add_amgcl_test(test_skyline_lu  test_skyline_lu.cpp)
add_amgcl_test(test_complex_erf test_complex_erf.cpp)
add_amgcl_test(test_qr          test_qr.cpp)

add_amgcl_test(test_solver_builtin   test_solver_builtin.cpp)
add_amgcl_test(test_solver_complex   test_solver_complex.cpp)
add_amgcl_test(test_solver_block_crs test_solver_block_crs.cpp)

if (BLAZE_INCLUDE)
    add_amgcl_test(test_solver_blaze     test_solver_blaze.cpp)
endif()

if (VIENNACL_INCLUDE)
    add_amgcl_test(test_solver_viennacl  test_solver_viennacl.cpp)
endif()

if (EIGEN_INCLUDE)
    add_amgcl_test(test_eigen_solver     test_eigen_solver.cpp)
    add_amgcl_test(test_solver_eigen     test_solver_eigen.cpp)
    add_amgcl_test(test_solver_nonscalar test_solver_nonscalar.cpp)
endif()
