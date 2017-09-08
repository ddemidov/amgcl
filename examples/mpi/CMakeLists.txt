if (TARGET mpi_target)
    function(add_mpi_example example sources)
        add_executable(${example} ${sources})
        target_link_libraries(${example}
            amgcl mpi_target
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            )

        if (TARGET pastix_target)
            target_link_libraries(${example} pastix_target)
        endif()
    endfunction()

    if (TARGET pastix_target)
        add_mpi_example(check_pastix check_pastix.cpp)
    endif()

    if (TARGET eigen_target AND TARGET metis_target)
        add_executable(partition partition.cpp)
        target_link_Libraries(partition amgcl eigen_target metis_target
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            )
    endif()

    add_mpi_example(runtime_sdd         runtime_sdd.cpp)
    add_mpi_example(runtime_sdd_3d      runtime_sdd_3d.cpp)
    add_mpi_example(runtime_bp          runtime_bp.cpp)
    add_mpi_example(schur_pc_mpi        schur_pc_mpi.cpp)
    add_mpi_example(solve_mm_mpi        solve_mm_mpi.cpp)
    add_mpi_example(call_mpi_lib        call_mpi_lib.cpp)

    target_link_libraries(call_mpi_lib libamgcl_mpi)

    if (TARGET cuda_target)
        foreach(example runtime_sdd runtime_sdd_3d)
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

    if (TARGET vexcl_target)
        foreach(example runtime_sdd runtime_sdd_3d)
            add_mpi_example(${example}_vexcl ${example}.cpp)
            target_compile_definitions(${example}_vexcl PRIVATE SOLVER_BACKEND_VEXCL)
            target_link_libraries(${example}_vexcl amgcl vexcl_target)
        endforeach()
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/plot_result
        ${CMAKE_CURRENT_BINARY_DIR}/plot_result
        COPYONLY
        )
endif()
