cmake_minimum_required(VERSION 2.8)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()

project(AMGCL)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

#----------------------------------------------------------------------------
# Find OpenMP
#----------------------------------------------------------------------------
find_package(OpenMP)
if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

#----------------------------------------------------------------------------
# Find Eigen
#----------------------------------------------------------------------------
find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
if (EIGEN_INCLUDE)
    include_directories(${EIGEN_INCLUDE})
    add_definitions(-DAMGCL_HAVE_EIGEN)
endif()

#----------------------------------------------------------------------------
# Find Blaze
#----------------------------------------------------------------------------
find_path(BLAZE_INCLUDE blaze/Math.h)
if (BLAZE_INCLUDE)
    include_directories(${BLAZE_INCLUDE})
    add_definitions(-DAMGCL_HAVE_BLAZE)
endif()

#----------------------------------------------------------------------------
# Find HPX
#----------------------------------------------------------------------------
option(HAVE_HPX "Build HPX examples" OFF)
if (HAVE_HPX)
    find_package(HPX)
    set(BOOST_COMPONENTS serialization)
endif()

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
else ()
    option(
        USE_STATIC_BOOST_LIBS
        "Use static versions of Boost libraries"
        OFF
        )

    if (USE_STATIC_BOOST_LIBS)
        set(Boost_USE_STATIC_LIBS ON)
    else()
        set(Boost_USE_STATIC_LIBS OFF)
        add_definitions(-DBOOST_TEST_DYN_LINK)
    endif()
endif ()

set(BOOST_COMPONENTS ${BOOST_COMPONENTS}
    program_options
    system
    filesystem
    serialization
    unit_test_framework
    )

find_package(Boost COMPONENTS ${BOOST_COMPONENTS})
include_directories( ${Boost_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Find VexCL
#----------------------------------------------------------------------------
set(VEXCL_ROOT $ENV{VEXCL_ROOT} CACHE STRING "VexCL root")
if (VEXCL_ROOT)
    include_directories( ${VEXCL_ROOT} )
    add_definitions(-DAMGCL_HAVE_VEXCL)

    set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA/Compute)")
    set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "OpenCL" "CUDA" "Compute")

    #----------------------------------------------------------------------------
    # Find VexCL backend
    #----------------------------------------------------------------------------
    if ("${VEXCL_BACKEND}" STREQUAL "OpenCL")
        find_package(OpenCL REQUIRED)
        include_directories( ${OPENCL_INCLUDE_DIRS} )
        set(VEXCL_BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
        add_definitions(-DVEXCL_BACKEND_OPENCL)
    elseif ("${VEXCL_BACKEND}" STREQUAL "Compute")
        find_package(OpenCL REQUIRED)
        include_directories( ${OPENCL_INCLUDE_DIRS} )
        set(VEXCL_BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
        add_definitions(-DVEXCL_BACKEND_COMPUTE)
    elseif ("${VEXCL_BACKEND}" STREQUAL "CUDA")
        find_package(CUDA REQUIRED)
        include_directories( ${CUDA_INCLUDE_DIRS} )
        set(VEXCL_BACKEND_LIBRARIES ${CUDA_CUDA_LIBRARY})
        add_definitions(-DVEXCL_BACKEND_CUDA)
    endif()
endif()

#----------------------------------------------------------------------------
# Find ViennaCL
#----------------------------------------------------------------------------
find_path(VIENNACL_INCLUDE viennacl/forwards.h)
if (VIENNACL_INCLUDE)
    add_definitions(-DAMGCL_HAVE_VIENNACL)
    add_definitions(-DVIENNACL_WITH_OPENMP)
    include_directories(${VIENNACL_INCLUDE})
endif()

#----------------------------------------------------------------------------
# Find CUDA
#----------------------------------------------------------------------------
find_package(CUDA QUIET)
if (CUDA_FOUND)
    set(CUDA_COMPUTE_CAPABILITY "35" CACHE STRING "CUDA Compute Capability")
    add_definitions(-DAMGCL_HAVE_CUDA)
    add_definitions(-DBOOST_DISABLE_ASSERTS)
endif ()

#----------------------------------------------------------------------------
# Find MPI
#----------------------------------------------------------------------------
find_package(MPI QUIET)
if (MPI_CXX_FOUND)
    include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

#----------------------------------------------------------------------------
# Find Pastix
#----------------------------------------------------------------------------
find_package(Metis  QUIET)
find_package(Scotch QUIET)
find_package(Pastix QUIET)
find_package(BLAS   QUIET)

if (METIS_INCLUDES)
    include_directories(${METIS_INCLUDES})
endif()

if (PASTIX_INCLUDES AND SCOTCH_INCLUDES AND BLAS_FOUND)
    set(HAVE_PASTIX ON)
    include_directories(${PASTIX_INCLUDES})
    add_definitions(-DAMGCL_HAVE_PASTIX)
endif()

if (SCOTCH_INCLUDES)
    include_directories(${SCOTCH_INCLUDES})
endif()

#----------------------------------------------------------------------------
# Visual Studio compilation flags
#----------------------------------------------------------------------------
if (WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_VARIADIC_MAX=10)
    add_definitions(/bigobj)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(/fp:fast)
endif ()

#----------------------------------------------------------------------------
# Enable C++11 support, set compilation flags
#----------------------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wno-unused-parameter -Wunused-but-set-parameter -Wno-comment -Wno-type-limits -Wno-strict-aliasing -Wno-unused-local-typedefs -Wno-deprecated-declarations -Wno-ignored-attributes -Wno-misleading-indentation")
    set(CXX11_FLAG -std=c++0x)
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CXX11_FLAG -std=c++11)
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wno-unused-parameter -Wno-comment -Wno-tautological-compare -Wno-deprecated-declarations -Wno-unused-local-typedef")
    set(CXX11_FLAG -std=c++11)

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif ()
endif ()

#----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

#----------------------------------------------------------------------------
set(BUILD_SHARED_LIBS ON)
add_subdirectory(lib)
add_subdirectory(examples)
add_subdirectory(docs)

#----------------------------------------------------------------------------
# Build Fortran example
#----------------------------------------------------------------------------
option(HAVE_FORTRAN "Build Fortran example" OFF)
if (HAVE_FORTRAN)
    add_subdirectory(fortran)
endif()

#----------------------------------------------------------------------------
install(DIRECTORY amgcl DESTINATION include)
