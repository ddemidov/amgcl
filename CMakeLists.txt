cmake_minimum_required(VERSION 2.8)
project(amgcl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
option(
    BOOST_TEST_DYN_LINK
    "Link tests against dynamic version of boost unittest library"
    ON
    )

if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
else ()
  if (BOOST_TEST_DYN_LINK)
    add_definitions(-DBOOST_TEST_DYN_LINK)
  endif ()
endif ()

set(BOOST_COMPONENTS
    chrono
    program_options
    system
    unit_test_framework
    )

find_package(Boost COMPONENTS ${BOOST_COMPONENTS})
include_directories( ${Boost_INCLUDE_DIRS} )

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
# Find VexCL
#----------------------------------------------------------------------------
set(VEXCL_ROOT $ENV{VEXCL_ROOT} CACHE STRING "VexCL root")
if (VEXCL_ROOT)
    include_directories( ${VEXCL_ROOT} )
    add_definitions(-DAMGCL_HAVE_VEXCL)
endif()

#----------------------------------------------------------------------------
# Tests
#----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
