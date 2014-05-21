include_directories(${CMAKE_SOURCE_DIR}/tests)

add_executable(solver solver.cpp)
target_link_libraries(solver ${Boost_LIBRARIES})

add_executable(solver_vexcl solver_vexcl.cpp)
target_link_libraries(solver_vexcl ${Boost_LIBRARIES} OpenCL)
