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

add_executable(crs_builder crs_builder.cpp)
target_link_libraries(crs_builder ${Boost_LIBRARIES})

add_executable(ublas ublas.cpp)
target_link_libraries(ublas ${Boost_LIBRARIES})

add_executable(block_crs block_crs.cpp)
target_link_libraries(block_crs ${Boost_LIBRARIES})

add_executable(call_lib call_lib.cpp)
target_link_libraries(call_lib ${Boost_LIBRARIES} amgcl)

add_executable(two_step_precond two_step_precond.cpp)
target_link_libraries(two_step_precond ${Boost_LIBRARIES})

add_executable(custom_adapter custom_adapter.cpp)
target_link_libraries(custom_adapter ${Boost_LIBRARIES})

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

if (CUDA_FOUND AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CUDA_OPTIONS "-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    if (NOT WIN32)
        set(CUDA_OPTIONS "${CUDA_OPTIONS} -Xcompiler -std=c++03")
    endif()
    cuda_add_executable(cuda cuda.cu OPTIONS ${CUDA_OPTIONS})
    if (NOT WIN32)
        set_target_properties(cuda PROPERTIES
            COMPILE_FLAGS -std=c++03
            LINK_FLAGS -std=c++03
            )
    endif()
    target_link_libraries(cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
endif()

if (VIENNACL_INCLUDE)
    add_definitions(-DVIENNACL_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories( ${OPENCL_INCLUDE_DIRS} )
    add_executable(runtime_viennacl runtime_viennacl.cpp)
    target_link_libraries(runtime_viennacl ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
endif()

add_subdirectory(mpi)
