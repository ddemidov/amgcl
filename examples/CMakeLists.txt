set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

find_package(OpenMP)
if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (OpenMP_CXX_FLAGS)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

include_directories(${CMAKE_SOURCE_DIR})

add_executable(genproblem genproblem.cpp)

find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
find_path(VIENNACL_INCLUDE viennacl/vector.hpp)
find_path(VEXCL_INCLUDE vexcl/vexcl.hpp)

if (EIGEN_INCLUDE)
    include_directories(${EIGEN_INCLUDE})
endif(EIGEN_INCLUDE)

if (VEXCL_INCLUDE)
    include_directories(${VEXCL_INCLUDE})
endif (VEXCL_INCLUDE)

if (VIENNACL_INCLUDE)
    include_directories(${VIENNACL_INCLUDE})
endif (VIENNACL_INCLUDE)

if (VEXCL_INCLUDE OR VIENNACL_INCLUDE)
    find_library(OPENCL_LIBRARY OpenCL)
endif (VEXCL_INCLUDE OR VIENNACL_INCLUDE)

if (EIGEN_INCLUDE)
    add_executable(eigen eigen.cpp)
endif (EIGEN_INCLUDE)

if (VIENNACL_INCLUDE AND EIGEN_INCLUDE)
    add_executable(viennacl_eigen viennacl_eigen.cpp)
endif (VIENNACL_INCLUDE AND EIGEN_INCLUDE)

if (VEXCL_INCLUDE)
add_executable(vexcl vexcl.cpp)
    if (NOT WIN32)
        set_target_properties(vexcl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    endif (NOT WIN32)
    target_link_libraries(vexcl ${OPENCL_LIBRARY})
endif (VEXCL_INCLUDE)

if (VIENNACL_INCLUDE AND VEXCL_INCLUDE)
    add_executable(viennacl viennacl.cpp)
    if (NOT WIN32)
        set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    endif (NOT WIN32)
    target_link_libraries(viennacl ${OPENCL_LIBRARY})

    add_executable(viennacl_vexcl viennacl_vexcl.cpp)
    if (NOT WIN32)
        set_target_properties(viennacl_vexcl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    endif (NOT WIN32)
    target_link_libraries(viennacl_vexcl ${OPENCL_LIBRARY})
endif (VIENNACL_INCLUDE AND VEXCL_INCLUDE)

if (VEXCL_INCLUDE AND EIGEN_INCLUDE AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    find_package(Boost COMPONENTS program_options)
    add_executable(utest utest.cpp)
    if (NOT WIN32)
        set_target_properties(viennacl_vexcl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    endif (NOT WIN32)
    set_target_properties(viennacl_vexcl PROPERTIES INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    target_link_libraries(utest ${OPENCL_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif (VEXCL_INCLUDE AND EIGEN_INCLUDE AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

