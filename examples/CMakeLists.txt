find_package(Boost COMPONENTS program_options chrono)
include_directories(${Boost_INCLUDE_DIRS})

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

find_package(CUDA QUIET)

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
    target_link_libraries(eigen ${Boost_CHRONO_LIBRARY})
endif (EIGEN_INCLUDE)

if (VIENNACL_INCLUDE AND EIGEN_INCLUDE)
    add_executable(viennacl_eigen viennacl_eigen.cpp)
    target_link_libraries(viennacl_eigen ${Boost_CHRONO_LIBRARY})
endif (VIENNACL_INCLUDE AND EIGEN_INCLUDE)

if (VEXCL_INCLUDE)
    add_executable(vexcl vexcl.cpp)
    set_target_properties(vexcl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    target_link_libraries(vexcl ${OPENCL_LIBRARY} ${Boost_CHRONO_LIBRARY})
endif (VEXCL_INCLUDE)

if (VIENNACL_INCLUDE AND VEXCL_INCLUDE)
    add_executable(viennacl viennacl.cpp)
    set_target_properties(viennacl PROPERTIES
        COMPILE_FLAGS "-std=c++0x -DVIENNACL_WITH_OPENCL")
    target_link_libraries(viennacl ${OPENCL_LIBRARY} ${Boost_CHRONO_LIBRARY})

    add_executable(viennacl_vexcl viennacl_vexcl.cpp)
    set_target_properties(viennacl_vexcl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    target_link_libraries(viennacl_vexcl ${OPENCL_LIBRARY} ${Boost_CHRONO_LIBRARY})
endif (VIENNACL_INCLUDE AND VEXCL_INCLUDE)

if (VIENNACL_INCLUDE AND CUDA_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/viennacl.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/viennacl.cu
        COPYONLY)
    include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
    cuda_add_executable(viennacl_cuda ${CMAKE_CURRENT_BINARY_DIR}/viennacl.cu
        OPTIONS "-DVIENNACL_WITH_CUDA;-arch=sm_20")
    target_link_libraries(viennacl_cuda ${Boost_CHRONO_LIBRARY})
endif (VIENNACL_INCLUDE AND CUDA_FOUND)

if (VEXCL_INCLUDE AND EIGEN_INCLUDE AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_executable(utest utest.cpp)
    set_target_properties(utest PROPERTIES COMPILE_FLAGS "-std=c++0x")
    target_link_libraries(utest
        ${OPENCL_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_CHRONO_LIBRARY}
        )
endif (VEXCL_INCLUDE AND EIGEN_INCLUDE AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

if (CUDA_FOUND)
    cuda_add_executable(cuda cuda.cu OPTIONS "-arch=sm_20")
    target_link_libraries(cuda ${CUDA_cusparse_LIBRARY} ${Boost_CHRONO_LIBRARY})
endif (CUDA_FOUND)
