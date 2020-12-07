add_executable(nullspace nullspace.cpp)
target_link_libraries(nullspace amgcl)

if (TARGET mpi_target)
    add_executable(nullspace_mpi nullspace_mpi.cpp)
    target_link_libraries(nullspace_mpi amgcl mpi_target)

    if (TARGET scotch_target)
        target_link_libraries(nullspace_mpi scotch_target)
    endif()

    if (TARGET Metis::metis)
        target_link_libraries(nullspace_mpi Metis::metis)
    endif()

endif()
