add_executable(coupcons3d coupcons3d.cpp)
target_link_libraries(coupcons3d amgcl)

#if (VexCL_FOUND)
#    vexcl_add_executables(coupcons3d_vexcl coupcons3d_vexcl.cpp)
#    target_link_libraries(coupcons3d_vexcl INTERFACE amgcl)
#endif()
