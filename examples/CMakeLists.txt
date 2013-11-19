if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif ()
find_package(Boost COMPONENTS
    program_options thread chrono date_time serialization system filesystem)

include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenMP)
if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif ()
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # Warning hunting:
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wunused-but-set-parameter -Wno-comment")
endif ()

include_directories(${CMAKE_SOURCE_DIR})

add_executable(genproblem2d genproblem2d.cpp)
add_executable(genproblem3d genproblem3d.cpp)

find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
find_path(VIENNACL_INCLUDE viennacl/vector.hpp)
find_path(VEXCL_INCLUDE vexcl/vexcl.hpp)

find_package(CUDA QUIET)
set(CUDA_COMPUTE_CAPABILITY "20" CACHE STRING "CUDA Compute Capability")

if (EIGEN_INCLUDE)
    include_directories(${EIGEN_INCLUDE})
endif()

if (VEXCL_INCLUDE)
    include_directories(${VEXCL_INCLUDE})
endif ()

if (VIENNACL_INCLUDE)
    include_directories(${VIENNACL_INCLUDE})
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
if (WIN32)
    set(OpenCL_INCPATH "$ENV{CUDA_INC_PATH}")
    set(OpenCL_LIBPATH "$ENV{CUDA_LIB_PATH}")
endif ()
find_package(OpenCL)
if (OPENCL_FOUND)
    include_directories(${OPENCL_INCLUDE_DIRS})
endif ()

if (EIGEN_INCLUDE)
    add_executable(eigen eigen.cpp)
    target_link_libraries(eigen ${Boost_LIBRARIES})
endif ()

add_executable(ublas ublas.cpp)
target_link_libraries(ublas ${Boost_LIBRARIES})

if (VIENNACL_INCLUDE AND EIGEN_INCLUDE)
    add_executable(viennacl_eigen viennacl_eigen.cpp)
    target_link_libraries(viennacl_eigen ${Boost_LIBRARIES})
endif ()

if (VEXCL_INCLUDE AND OPENCL_FOUND)
    add_executable(vexcl_cl vexcl.cpp)
    if (WIN32)
        set_target_properties(vexcl_cl PROPERTIES COMPILE_FLAGS "-DVEXCL_BACKEND_OPENCL")
    else ()
	set_target_properties(vexcl_cl PROPERTIES COMPILE_FLAGS "-DVEXCL_BACKEND_OPENCL -std=c++0x")
    endif ()
    target_link_libraries(vexcl_cl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endif ()

if (VEXCL_INCLUDE AND CUDA_FOUND)
    add_executable(vexcl_cu vexcl.cpp)
    if (WIN32)
        set_target_properties(vexcl_cu PROPERTIES COMPILE_FLAGS "-DVEXCL_BACKEND_CUDA -DVEXCL_USE_CUSPARSE")
    else ()
	set_target_properties(vexcl_cu PROPERTIES COMPILE_FLAGS "-DVEXCL_BACKEND_CUDA -DVEXCL_USE_CUSPARSE -std=c++0x")
    endif ()
    target_link_libraries(vexcl_cu ${CUDA_CUDA_LIBRARY} ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
endif ()

if (VIENNACL_INCLUDE AND VEXCL_INCLUDE AND OPENCL_FOUND)
    add_executable(viennacl viennacl.cpp)
    set_target_properties(viennacl PROPERTIES
        COMPILE_FLAGS "-std=c++0x -DVIENNACL_WITH_OPENCL")
    target_link_libraries(viennacl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})

    add_executable(viennacl_vexcl viennacl_vexcl.cpp)
    if (NOT WIN32)
	set_target_properties(viennacl_vexcl PROPERTIES COMPILE_FLAGS "-std=c++0x")
    endif ()
    target_link_libraries(viennacl_vexcl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endif ()

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
        OPTIONS "-DVIENNACL_WITH_CUDA;-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    target_link_libraries(viennacl_cuda ${Boost_LIBRARIES})
endif ()

if (VEXCL_INCLUDE AND EIGEN_INCLUDE AND VIENNACL_INCLUDE AND OPENCL_FOUND AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_executable(utest utest.cpp)
    if (NOT WIN32)
	set_target_properties(utest PROPERTIES COMPILE_FLAGS "-std=c++0x")
    endif ()
    target_link_libraries(utest
        ${OPENCL_LIBRARIES}
        ${Boost_LIBRARIES}
        )
endif ()

if (CUDA_FOUND)
    cuda_add_executable(cuda cuda.cu OPTIONS "-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    target_link_libraries(cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
endif ()
