add_executable(serena serena.cpp)
target_link_libraries(serena amgcl)

if (VexCL_FOUND)
    vexcl_add_executables(serena_vexcl serena_vexcl.cpp)
    target_link_libraries(serena_vexcl INTERFACE amgcl)
endif()

if (TARGET mpi_target)
    add_executable(serena_mpi serena_mpi.cpp)
    target_link_libraries(serena_mpi amgcl mpi_target)

    if (TARGET scotch_target)
        target_link_libraries(serena_mpi scotch_target)
    endif()

    if (TARGET Metis::metis)
        target_link_libraries(serena_mpi Metis::metis)
    endif()

    if (VexCL_FOUND)
        vexcl_add_executables(serena_mpi_vexcl serena_mpi_vexcl.cpp)
        target_link_libraries(serena_mpi_vexcl INTERFACE amgcl mpi_target)

        if (TARGET scotch_target)
            target_link_libraries(serena_mpi_vexcl INTERFACE scotch_target)
        endif()

        if (TARGET Metis::metis)
            target_link_libraries(serena_mpi_vexcl INTERFACE Metis::metis)
        endif()
    endif()
endif()
