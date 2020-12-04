if (TARGET mpi_target)
    add_library(amgcl_mpi_example INTERFACE)
    target_link_libraries(amgcl_mpi_example INTERFACE
        amgcl mpi_target ${Boost_PROGRAM_OPTIONS_LIBRARY})
    target_compile_definitions(amgcl_mpi_example INTERFACE "AMGCL_BLOCK_SIZES=${AMGCL_BLOCK_SIZES_SEQ}")

    if (TARGET eigen_target)
        target_link_libraries(amgcl_mpi_example INTERFACE eigen_target)
    endif()

    if (TARGET pastix_target)
        target_link_libraries(amgcl_mpi_example INTERFACE pastix_target)
    endif()

    if (TARGET scotch_target)
        target_link_libraries(amgcl_mpi_example INTERFACE scotch_target)
    endif()

    if (TARGET Metis::metis)
        target_link_libraries(amgcl_mpi_example INTERFACE Metis::metis)
    endif()

    function(add_mpi_example example sources)
        add_executable(${example} ${sources})
        target_link_libraries(${example} amgcl_mpi_example)
    endfunction()

    add_mpi_example(check_direct        check_direct.cpp)
    add_mpi_example(runtime_sdd         runtime_sdd.cpp)
    add_mpi_example(runtime_sdd_3d      runtime_sdd_3d.cpp)
    add_mpi_example(runtime_bp          runtime_bp.cpp)
    add_mpi_example(schur_pc_mpi        schur_pc_mpi.cpp)
    add_mpi_example(solve_mm_mpi        solve_mm_mpi.cpp)
    add_mpi_example(call_mpi_lib        call_mpi_lib.cpp)
    add_mpi_example(test_transpose      test_transpose.cpp)
    add_mpi_example(test_spmm           test_spmm.cpp)
    add_mpi_example(spmm_scaling        spmm_scaling.cpp)
    add_mpi_example(mpi_solver          mpi_solver.cpp)
    add_mpi_example(cpr_mpi             cpr_mpi.cpp)

    if (TARGET Metis::metis)
        add_executable(partition partition.cpp)
        target_link_libraries(partition
            amgcl
            Metis::metis
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            )
    endif()

    target_link_libraries(call_mpi_lib libamgcl_mpi)

    if (TARGET cuda_target)
        foreach(example runtime_sdd runtime_sdd_3d schur_pc_mpi mpi_solver)
            configure_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/${example}.cu
                COPYONLY
                )
            cuda_add_executable(${example}_cuda
                ${CMAKE_CURRENT_BINARY_DIR}/${example}.cu
                OPTIONS -DSOLVER_BACKEND_CUDA)
            target_include_directories(${example}_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
            target_link_libraries(${example}_cuda amgcl_example cuda_target mpi_target)
        endforeach()
    endif()

    if (VexCL_FOUND)
        foreach(example runtime_sdd runtime_sdd_3d schur_pc_mpi mpi_solver)
            vexcl_add_executables(${example}_vexcl ${example}.cpp)
            target_compile_definitions(${example}_vexcl INTERFACE SOLVER_BACKEND_VEXCL)
            target_link_libraries(${example}_vexcl INTERFACE amgcl_mpi_example)
        endforeach()
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/plot_result
        ${CMAKE_CURRENT_BINARY_DIR}/plot_result
        COPYONLY
        )
endif()
