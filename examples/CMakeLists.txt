include_directories(${CMAKE_SOURCE_DIR}/tests)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/make_poisson.py
    ${CMAKE_BINARY_DIR}/make_poisson.py
    COPYONLY
    )

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyamgcl_example.py
    ${CMAKE_BINARY_DIR}/pyamgcl_example.py
    COPYONLY
    )

add_executable(solver solver.cpp)
target_link_libraries(solver ${Boost_LIBRARIES})

add_executable(ublas ublas.cpp)
target_link_libraries(ublas ${Boost_LIBRARIES})

add_executable(runtime runtime.cpp)
target_link_libraries(runtime ${Boost_LIBRARIES})

add_executable(block_crs block_crs.cpp)
target_link_libraries(block_crs ${Boost_LIBRARIES})

add_executable(call_lib call_lib.cpp)
target_link_libraries(call_lib ${Boost_LIBRARIES} amgcl)

if (EIGEN_INCLUDE)
    add_executable(solve_mm solve_mm.cpp)
    target_link_libraries(solve_mm ${Boost_LIBRARIES})
endif()

if (HPX_FOUND)
    add_hpx_executable(hpx
        ESSENTIAL
        SOURCES hpx.cpp
        )
endif()

if (VEXCL_ROOT)
    add_executable(vexcl vexcl.cpp)
    target_link_libraries(vexcl ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})
    set_target_properties(vexcl PROPERTIES COMPILE_FLAGS ${CXX11_FLAG})

    add_executable(runtime_vexcl runtime_vexcl.cpp)
    target_link_libraries(runtime_vexcl ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})
    set_target_properties(runtime_vexcl PROPERTIES COMPILE_FLAGS ${CXX11_FLAG})
endif()

if (CUDA_FOUND)
    cuda_add_executable(cuda cuda.cu OPTIONS "-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    target_link_libraries(cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
endif()

if (VIENNACL_INCLUDE)
    add_executable(runtime_viennacl runtime_viennacl.cpp)
    target_link_libraries(runtime_viennacl ${Boost_LIBRARIES})
endif()

add_subdirectory(mpi)
