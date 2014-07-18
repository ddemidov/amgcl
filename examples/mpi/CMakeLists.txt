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

find_package(MPI QUIET)

if (MPI_CXX_FOUND)
    include_directories(${MPI_CXX_INCLUDE_PATH})

    #------------------------------------------------------------------------
    # Find Pastix
    #------------------------------------------------------------------------
    find_package(Pastix QUIET)
    find_package(Scotch QUIET)
    find_package(BLAS   QUIET)

    if (PASTIX_INCLUDES AND SCOTCH_INCLUDES AND BLAS_FOUND)
        set(HAVE_PASTIX ON)
        include_directories(${PASTIX_INCLUDES} ${SCOTCH_INCLUDES})
        add_definitions(-DHAVE_PASTIX)
        add_mpi_example(check_pastix check_pastix.cpp)
    endif()

    add_mpi_example(subdomain_deflation subdomain_deflation.cpp)

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
