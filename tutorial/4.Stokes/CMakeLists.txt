add_executable(stokes_ucube stokes_ucube.cpp)
target_link_libraries(stokes_ucube amgcl)

if (VexCL_FOUND)
    vexcl_add_executables(stokes_ucube_vexcl stokes_ucube_vexcl.cpp)
    target_link_libraries(stokes_ucube_vexcl INTERFACE amgcl)
endif()

