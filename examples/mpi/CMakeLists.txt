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
                )
        endif()
    endfunction()

    if (HAVE_PASTIX)
        add_mpi_example(check_pastix check_pastix.cpp)
    endif()

    add_mpi_example(subdomain_deflation subdomain_deflation.cpp)
    add_mpi_example(call_mpi_lib        call_mpi_lib.cpp)
    target_link_libraries(call_mpi_lib amgcl amgcl_mpi)

    if (VEXCL_ROOT)
        add_mpi_example(subdomain_deflation_vexcl subdomain_deflation_vexcl.cpp)
        target_link_libraries(subdomain_deflation_vexcl ${VEXCL_BACKEND_LIBRARIES})
        set_target_properties(subdomain_deflation_vexcl PROPERTIES
            COMPILE_FLAGS -std=c++11)
    endif()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/plot_result
        ${CMAKE_CURRENT_BINARY_DIR}/plot_result
        COPYONLY
        )
endif()
