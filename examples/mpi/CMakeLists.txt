if (MPI_CXX_FOUND)
    function(add_mpi_example example sources)
        add_executable(${example} ${sources})
        set_target_properties(${example} PROPERTIES
            COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
            LINK_FLAGS    "${MPI_CXX_LINK_FLAGS}"
            )
        target_link_libraries(${example}
            ${Boost_LIBRARIES}
            ${MPI_CXX_LIBRARIES}
            )

        if (HAVE_PASTIX)
            target_link_libraries(${example}
                ${PASTIX_LIBRARIES}
                ${SCOTCH_LIBRARIES}
                ${BLAS_LIBRARIES}
                pthread
                )
        endif()
    endfunction()

    if (HAVE_PASTIX)
        add_mpi_example(check_pastix check_pastix.cpp)
    endif()

    if (EIGEN_INCLUDE AND METIS_INCLUDES)
        add_executable(partition partition.cpp)
        target_link_Libraries(partition
            ${Boost_LIBRARIES}
            ${METIS_LIBRARY}
            )
    endif()

    add_mpi_example(runtime_sdd         runtime_sdd.cpp)
    add_mpi_example(runtime_sdd_3d      runtime_sdd_3d.cpp)
    add_mpi_example(solve_mm_mpi        solve_mm_mpi.cpp)
    add_mpi_example(call_mpi_lib        call_mpi_lib.cpp)
    target_link_libraries(call_mpi_lib amgcl amgcl_mpi)

    if (VEXCL_ROOT)
        add_mpi_example(runtime_sdd_vexcl runtime_sdd_vexcl.cpp)
        target_link_libraries(runtime_sdd_vexcl ${VEXCL_BACKEND_LIBRARIES})
        set_target_properties(runtime_sdd_vexcl PROPERTIES COMPILE_FLAGS ${CXX11_FLAG})
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/plot_result
        ${CMAKE_CURRENT_BINARY_DIR}/plot_result
        COPYONLY
        )
endif()
