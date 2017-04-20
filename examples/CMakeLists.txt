include_directories(${CMAKE_SOURCE_DIR}/tests)

add_executable(mm2bin mm2bin.cpp)
target_link_libraries(mm2bin ${Boost_LIBRARIES})

add_executable(solver solver.cpp)
target_link_libraries(solver ${Boost_LIBRARIES})

add_executable(solver_complex solver_complex.cpp)
target_link_libraries(solver_complex ${Boost_LIBRARIES})

add_executable(crs_builder crs_builder.cpp)
target_link_libraries(crs_builder ${Boost_LIBRARIES})

add_executable(ublas ublas.cpp)
target_link_libraries(ublas ${Boost_LIBRARIES})

add_executable(block_crs block_crs.cpp)
target_link_libraries(block_crs ${Boost_LIBRARIES})

add_executable(call_lib call_lib.cpp)
target_link_libraries(call_lib ${Boost_LIBRARIES} libamgcl)

add_executable(schur_pressure_correction schur_pressure_correction.cpp)
target_link_libraries(schur_pressure_correction ${Boost_LIBRARIES})

add_executable(cpr cpr.cpp)
target_link_libraries(cpr ${Boost_LIBRARIES})

add_executable(cpr_drs cpr_drs.cpp)
target_link_libraries(cpr_drs ${Boost_LIBRARIES})

add_executable(custom_adapter custom_adapter.cpp)
target_link_libraries(custom_adapter ${Boost_LIBRARIES})

if (HPX_FOUND)
    add_hpx_executable(hpx
        ESSENTIAL
        SOURCES hpx.cpp
        )
endif()

if (EIGEN_INCLUDE)
    add_executable(solver_eigen solver.cpp)
    target_link_libraries(solver_eigen ${Boost_LIBRARIES})
    set_target_properties(solver_eigen PROPERTIES COMPILE_FLAGS "-DSOLVER_BACKEND_EIGEN")
endif()

if (BLAZE_INCLUDE)
    add_executable(solver_blaze solver.cpp)
    target_link_libraries(solver_blaze ${Boost_LIBRARIES})
    set_target_properties(solver_blaze PROPERTIES COMPILE_FLAGS "-DSOLVER_BACKEND_BLAZE")
endif()

if (VEXCL_ROOT)
    add_executable(vexcl vexcl.cpp)
    target_link_libraries(vexcl ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})
    set_target_properties(vexcl PROPERTIES COMPILE_FLAGS ${CXX11_FLAG})

    add_executable(solver_vexcl solver.cpp)
    target_link_libraries(solver_vexcl ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})
    set_target_properties(solver_vexcl PROPERTIES
        COMPILE_FLAGS "${CXX11_FLAG} -DSOLVER_BACKEND_VEXCL"
        )

    add_executable(schur_pressure_correction_vexcl schur_pressure_correction.cpp)
    target_link_libraries(schur_pressure_correction_vexcl ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})
    set_target_properties(schur_pressure_correction_vexcl PROPERTIES
        COMPILE_FLAGS "${CXX11_FLAG} -DSOLVER_BACKEND_VEXCL"
        )

    add_executable(cpr_drs_vexcl cpr_drs.cpp)
    target_link_libraries(cpr_drs_vexcl ${Boost_LIBRARIES} ${VEXCL_BACKEND_LIBRARIES})
    set_target_properties(cpr_drs_vexcl PROPERTIES
        COMPILE_FLAGS "${CXX11_FLAG} -DSOLVER_BACKEND_VEXCL"
        )
endif()

if (CUDA_FOUND AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CUDA_OPTIONS "-DSOLVER_BACKEND_CUDA -arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    if (NOT WIN32)
        set(CUDA_OPTIONS "${CUDA_OPTIONS} -Xcompiler -std=c++03")
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/solver.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/solver.cu
        COPYONLY
        )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/schur_pressure_correction.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/schur_pressure_correction.cu
        COPYONLY
        )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cpr_drs.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/cpr_drs.cu
        COPYONLY
        )

    cuda_add_executable(solver_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/solver.cu
        OPTIONS ${CUDA_OPTIONS})

    cuda_add_executable(schur_pressure_correction_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/schur_pressure_correction.cu
        OPTIONS ${CUDA_OPTIONS})

    cuda_add_executable(cpr_drs_cuda
        ${CMAKE_CURRENT_BINARY_DIR}/cpr_drs.cu
        OPTIONS ${CUDA_OPTIONS})

    if (NOT WIN32)
        set_target_properties(solver_cuda PROPERTIES
            COMPILE_FLAGS -std=c++03
            LINK_FLAGS -std=c++03
            )

        set_target_properties(schur_pressure_correction_cuda PROPERTIES
            COMPILE_FLAGS -std=c++03
            LINK_FLAGS -std=c++03
            )

        set_target_properties(cpr_drs_cuda PROPERTIES
            COMPILE_FLAGS -std=c++03
            LINK_FLAGS -std=c++03
            )
    endif()

    target_link_libraries(solver_cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
    target_link_libraries(schur_pressure_correction_cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
    target_link_libraries(cpr_drs_cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
endif()

if (VIENNACL_INCLUDE)
    add_definitions(-DVIENNACL_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories( ${OPENCL_INCLUDE_DIRS} )

    add_executable(solver_viennacl solver.cpp)
    target_link_libraries(solver_viennacl ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
    set_target_properties(solver_viennacl PROPERTIES COMPILE_FLAGS "-DSOLVER_BACKEND_VIENNACL")

    add_executable(schur_pressure_correction_viennacl schur_pressure_correction.cpp)
    target_link_libraries(schur_pressure_correction_viennacl ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
    set_target_properties(schur_pressure_correction_viennacl PROPERTIES COMPILE_FLAGS "-DSOLVER_BACKEND_VIENNACL")

    add_executable(cpr_drs_viennacl cpr_drs.cpp)
    target_link_libraries(cpr_drs_viennacl ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})
    set_target_properties(cpr_drs_viennacl PROPERTIES COMPILE_FLAGS "-DSOLVER_BACKEND_VIENNACL")
endif()

add_subdirectory(mpi)
