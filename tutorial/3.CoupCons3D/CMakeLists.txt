add_executable(coupcons3d coupcons3d.cpp)
target_link_libraries(coupcons3d amgcl)

add_executable(coupcons3d_spc coupcons3d_spc.cpp)
target_link_libraries(coupcons3d_spc amgcl)

if (VexCL_FOUND)
    vexcl_add_executables(coupcons3d_vexcl coupcons3d_vexcl.cpp)
    target_link_libraries(coupcons3d_vexcl INTERFACE amgcl)
endif()
