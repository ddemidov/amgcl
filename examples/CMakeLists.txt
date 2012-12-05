find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++0x")

find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)

include_directories(${EIGEN_INCLUDE} ${CMAKE_SOURCE_DIR})

add_executable(eigen eigen.cpp)
add_executable(viennacl_eigen viennacl_eigen.cpp)

add_executable(vexcl vexcl.cpp)
target_link_libraries(vexcl OpenCL)

add_executable(aggr aggr.cpp)
target_link_libraries(aggr OpenCL)

add_executable(viennacl_vexcl viennacl_vexcl.cpp)
target_link_libraries(viennacl_vexcl OpenCL)
