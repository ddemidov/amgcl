if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -stdlib=libc++")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

find_path(EIGEN_INCLUDE Eigen/SparseCore
    PATH_SUFFIXES eigen3
    )

include_directories(
    ${EIGEN_INCLUDE}
    ${CMAKE_SOURCE_DIR}
    )

add_executable(eigen eigen.cpp)

add_executable(vexcl vexcl.cpp)
target_link_libraries(vexcl OpenCL)
