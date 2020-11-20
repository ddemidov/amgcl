add_executable(serena serena.cpp)
target_link_libraries(serena amgcl)

if (VexCL_FOUND)
    vexcl_add_executables(serena_vexcl serena_vexcl.cpp)
    target_link_libraries(serena_vexcl INTERFACE amgcl)
endif()
