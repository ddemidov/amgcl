add_executable(serena serena.cpp)
target_link_libraries(serena amgcl)

if (VexCL_FOUND)
    vexcl_add_executables(serena_vexcl serena_vexcl.cpp)
    target_link_libraries(serena_vexcl INTERFACE amgcl)
endif()

if (TARGET mpi_target)
    add_executable(serena_mpi serena_mpi.cpp)
    target_link_libraries(serena_mpi amgcl mpi_target)

    add_executable(serena_mpi_mixed serena_mpi_mixed.cpp)
    target_link_libraries(serena_mpi_mixed amgcl mpi_target)
endif()
