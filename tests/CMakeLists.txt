#----------------------------------------------------------------------------
# Test coverage
#----------------------------------------------------------------------------
option(AMGCL_TEST_COVERAGE "Analyze test coverage with gcov/lcov" OFF)

if(AMGCL_TEST_COVERAGE)
    target_compile_options(amgcl INTERFACE --coverage)
    target_link_libraries(amgcl INTERFACE --coverage)

    # Resets coverage statistics
    add_custom_target(coverage_reset
        COMMAND lcov --zerocounters --directory .
        COMMAND lcov --capture --initial --directory . --base-directory "${CMAKE_SOURCE_DIR}/amgcl" --no-external --output-file coverage.info
        VERBATIM
        )

    # Converts accumulated coverage statistics into coverage/index.html
    # Run `make tests` after `make coverage_reset` and before `make coverage`.
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
add_library(amgcl_test INTERFACE)

if (NOT Boost_USE_STATIC_LIBS)
    target_compile_definitions(amgcl_test INTERFACE BOOST_TEST_DYN_LINK)
endif ()

target_include_directories(amgcl_test INTERFACE ${Boost_INCLUDE_DIRS})
target_link_libraries(amgcl_test INTERFACE
    amgcl
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )

function(add_amgcl_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} amgcl_test)
    add_test(${TEST_NAME} ${TEST_NAME})
endfunction()

add_amgcl_test(test_skyline_lu        test_skyline_lu.cpp)
add_amgcl_test(test_complex_erf       test_complex_erf.cpp)
add_amgcl_test(test_qr                test_qr.cpp)
add_amgcl_test(test_solver_builtin    test_solver_builtin.cpp)
add_amgcl_test(test_solver_complex    test_solver_complex.cpp)
add_amgcl_test(test_solver_block_crs  test_solver_block_crs.cpp)
add_amgcl_test(test_solver_ns_builtin test_solver_ns_builtin.cpp)
add_amgcl_test(test_io                test_io.cpp)

add_amgcl_test(test_static_matrix test_static_matrix.cpp)
target_compile_options(test_static_matrix PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-std=c++0x>
    $<$<CXX_COMPILER_ID:Clang>:-std=c++0x>
    )

if (TARGET blaze_target)
    add_amgcl_test(test_solver_blaze test_solver_blaze.cpp)
    target_link_libraries(test_solver_blaze blaze_target)
endif()

if (TARGET eigen_target)
    add_amgcl_test(test_eigen_solver    test_eigen_solver.cpp)
    add_amgcl_test(test_solver_eigen    test_solver_eigen.cpp)
    add_amgcl_test(test_solver_ns_eigen test_solver_ns_eigen.cpp)

    target_link_libraries(test_eigen_solver eigen_target)
    target_link_libraries(test_solver_eigen eigen_target)
    target_link_libraries(test_solver_ns_eigen eigen_target)
endif()

if (TARGET viennacl_target)
    add_amgcl_test(test_solver_viennacl  test_solver_viennacl.cpp)
    target_link_libraries(test_solver_viennacl viennacl_target)
endif()

if (TARGET VexCL::OpenCL)
    add_amgcl_test(test_solver_vexcl  test_solver_vexcl.cpp)
    target_link_libraries(test_solver_vexcl VexCL::OpenCL)
endif()

if (AMGCL_HAVE_PYTHON AND NOT WIN32)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test_pyamgcl.py
        ${CMAKE_CURRENT_BINARY_DIR}/test_pyamgcl.py
        COPYONLY
        )
    add_test(test_pyamgcl test_pyamgcl.py)
    set_tests_properties(test_pyamgcl
        PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/..")
endif()
