cmake_minimum_required(VERSION 2.9)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()

project(AMGCL)

if (NOT (CMAKE_VERSION LESS 3.3))
    cmake_policy(SET CMP0058 OLD)
endif()

set(AMGCL_MASTER_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(AMGCL_MASTER_PROJECT ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if (CMAKE_VERSION VERSION_LESS "3.1.0")
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/opencl)
endif()

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
option(Boost_USE_STATIC_LIBS "Use static versions of Boost libraries" OFF)
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif ()

find_package(Boost REQUIRED COMPONENTS
    chrono
    date_time
    program_options
    serialization
    system
    thread
    unit_test_framework
    )

#----------------------------------------------------------------------------
# Builtin backend
#----------------------------------------------------------------------------
add_library(amgcl INTERFACE)
add_library(amgcl::amgcl ALIAS amgcl)

target_include_directories(amgcl SYSTEM INTERFACE
    ${Boost_INCLUDE_DIRS}
    )

target_include_directories(amgcl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )

find_package(OpenMP)
if (OPENMP_FOUND)
    target_compile_options(amgcl INTERFACE ${OpenMP_CXX_FLAGS})
    target_link_libraries(amgcl INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
        $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
        $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
        )
endif ()

#----------------------------------------------------------------------------
# Common compile options and definitions
#----------------------------------------------------------------------------
target_compile_options(amgcl INTERFACE
    # Compiler is GNU (g++):
    $<$<CXX_COMPILER_ID:GNU>:-ffast-math>
    $<$<CXX_COMPILER_ID:GNU>:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic>>
    # Compiler is Clang:
    $<$<CXX_COMPILER_ID:Clang>:-ffast-math>
    $<$<CXX_COMPILER_ID:Clang>:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic;-Wno-c99-extensions>>
    # Compiler is MSVC:
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    $<$<CXX_COMPILER_ID:MSVC>:/fp:fast>
    $<$<CXX_COMPILER_ID:MSVC>:/wd4715>
    )

target_compile_definitions(amgcl INTERFACE
    # Compiler is MSVC:
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    $<$<CXX_COMPILER_ID:MSVC>:_VARIADIC_MAX=10>
    $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>
    )

#----------------------------------------------------------------------------
# Enable asynchronous setup?
#----------------------------------------------------------------------------
option(AMGCL_ASYNC_SETUP "Enable asynchronous AMG setup" ON)
if (AMGCL_ASYNC_SETUP)
    target_compile_definitions(amgcl INTERFACE AMGCL_ASYNC_SETUP)
    target_include_directories(amgcl INTERFACE ${Boost_INCLUDE_DIRS})
    target_link_libraries(amgcl INTERFACE
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_THREAD_LIBRARY}
        ${Boost_CHRONO_LIBRARY}
        ${Boost_DATE_TIME_LIBRARY}
        )
endif()

#----------------------------------------------------------------------------
# Eigen backend
#----------------------------------------------------------------------------
find_path(EIGEN_INCLUDE Eigen/SparseCore PATH_SUFFIXES eigen3)
if (EIGEN_INCLUDE)
    add_library(eigen_target INTERFACE)
    target_include_directories(eigen_target INTERFACE ${EIGEN_INCLUDE})
    target_compile_options(eigen_target INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-int-in-bool-context>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-long-long>
        )
endif()

#----------------------------------------------------------------------------
# Find Blaze
#----------------------------------------------------------------------------
find_package(blaze QUIET)
if (blaze_FOUND)
    add_library(blaze_target INTERFACE)
    target_link_libraries(blaze_target INTERFACE blaze::blaze)
endif()

#----------------------------------------------------------------------------
# Find VexCL
#----------------------------------------------------------------------------
find_package(VexCL QUIET)
if (VexCL_FOUND)
    add_library(vexcl_target INTERFACE)

    set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA/Compute/JIT)")
    set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "OpenCL" "CUDA" "Compute" "JIT")

    if("${VEXCL_BACKEND}" STREQUAL "OpenCL")
        target_link_libraries(vexcl_target INTERFACE VexCL::OpenCL)
    elseif("${VEXCL_BACKEND}" STREQUAL "Compute")
        target_link_libraries(vexcl_target INTERFACE VexCL::Compute)
    elseif("${VEXCL_BACKEND}" STREQUAL "CUDA")
        target_link_libraries(vexcl_target INTERFACE VexCL::CUDA)
    elseif("${VEXCL_BACKEND}" STREQUAL "JIT")
        target_link_libraries(vexcl_target INTERFACE VexCL::JIT)
    endif()
endif()

#----------------------------------------------------------------------------
# Find ViennaCL
#----------------------------------------------------------------------------
find_path(VIENNACL_INCLUDE viennacl/forwards.h)
if (VIENNACL_INCLUDE)
    add_library(viennacl_target INTERFACE)
    target_link_libraries(viennacl_target INTERFACE amgcl)
    target_include_directories(viennacl_target INTERFACE ${VIENNACL_INCLUDE})
    target_compile_definitions(viennacl_target INTERFACE SOLVER_BACKEND_VIENNACL)

    find_package(OpenCL)
    if (OpenCL_FOUND)
        target_include_directories(viennacl_target INTERFACE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(viennacl_target INTERFACE ${OpenCL_LIBRARY})
        target_compile_definitions(viennacl_target INTERFACE VIENNACL_WITH_OPENCL)
        target_compile_options(viennacl_target INTERFACE
            $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
            $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-ignored-attributes>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>
            )
    else()
        target_compile_definitions(viennacl_target INTERFACE VIENNACL_WITH_OPENMP)
    endif()
endif()

#----------------------------------------------------------------------------
# Find CUDA
#----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    find_package(CUDA)
    if (CUDA_FOUND)
        set(CUDA_TARGET_ARCH "Kepler Maxwell Pascal" CACHE STRING "Target architecture(s) for CUDA")
        cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_TARGET_ARCH})

        list(APPEND CUDA_NVCC_FLAGS
            ${CUDA_ARCH_FLAGS} -Wno-deprecated-gpu-targets)

        if (OPENMP_FOUND)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
        endif()

        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler -std=c++03 -Xcompiler -Wno-vla)
        endif()

        add_library(cuda_target INTERFACE)
        target_link_libraries(cuda_target INTERFACE ${CUDA_cusparse_LIBRARY})
    endif()
endif()

#----------------------------------------------------------------------------
# Find MPI
#----------------------------------------------------------------------------
find_package(MPI)
if (MPI_CXX_FOUND)
    add_library(mpi_target INTERFACE)
    target_link_libraries(mpi_target INTERFACE amgcl ${MPI_CXX_LIBRARIES})
    target_include_directories(mpi_target INTERFACE ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(mpi_target INTERFACE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-long-long>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-long-long>
        )
endif()

#----------------------------------------------------------------------------
# Find Pastix
#----------------------------------------------------------------------------
find_package(Metis  QUIET)
find_package(Scotch QUIET)
find_package(Pastix QUIET)
find_package(BLAS   QUIET)
find_library(PTHREAD_LIBRARY pthread)
find_library(BZ2_LIBRARY bz2)
find_library(HWLOC_LIBRARY hwloc)

if (METIS_INCLUDES)
    add_library(metis_target INTERFACE)
    target_include_directories(metis_target INTERFACE ${METIS_INCLUDES})
    target_link_libraries(metis_target INTERFACE ${METIS_LIBRARY})
endif()

if (PASTIX_INCLUDES AND SCOTCH_INCLUDES AND BLAS_FOUND AND PTHREAD_LIBRARY AND BZ2_LIBRARY)
    add_library(pastix_target INTERFACE)
    target_include_directories(pastix_target INTERFACE
        ${PASTIX_INCLUDES}
        ${SCOTCH_INCLUDES}
        )
    target_link_libraries(pastix_target INTERFACE
        ${PASTIX_LIBRARIES}
        ${SCOTCH_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${PTHREAD_LIBRARY}
        ${BZ2_LIBRARY}
        ${HWLOC_LIBRARY}
        )
    target_compile_definitions(pastix_target INTERFACE AMGCL_HAVE_PASTIX)
endif()

if (AMGCL_MASTER_PROJECT)
    option(AMGCL_BUILD_TESTS    OFF)
    option(AMGCL_BUILD_EXAMPLES OFF)
    option(AMGCL_DISABLE_RARE_COMPONENTS OFF)

    if(AMGCL_DISABLE_RARE_COMPONENTS)
        add_definitions(
            -DAMGCL_RUNTIME_DISABLE_MULTICOLOR_GS
            -DAMGCL_RUNTIME_DISABLE_SPAI1
            -DAMGCL_RUNTIME_DISABLE_CHEBYSHEV
            )
    endif()

    add_subdirectory(docs)

    if (AMGCL_BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()

    if (AMGCL_BUILD_EXAMPLES)
        add_subdirectory(lib)
        add_subdirectory(examples)
        add_subdirectory(pyamgcl)


        #----------------------------------------------------------------------------
        # Build Fortran example
        #----------------------------------------------------------------------------
        option(AMGCL_HAVE_FORTRAN "Build Fortran example" OFF)
        if (AMGCL_HAVE_FORTRAN)
            add_subdirectory(fortran)
        endif()
    endif()

    install(DIRECTORY amgcl DESTINATION include)

    install(TARGETS amgcl EXPORT amgclTargets)

    if(TARGET eigen)
        install(TARGETS eigen EXPORT amgclTargets)
    endif()

    if(TARGET blaze)
        install(TARGETS blaze EXPORT amgclTargets)
    endif()

    if(TARGET viennacl)
        install(TARGETS viennacl EXPORT amgclTargets)
    endif()

    if(TARGET mpi)
        install(TARGETS mpi EXPORT amgclTargets)
    endif()

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/amgcl-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/amgcl-config.cmake"
        COPYONLY
        )

    export(EXPORT amgclTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/amgcl-targets.cmake"
        NAMESPACE amgcl::
        )

    export(PACKAGE amgcl)

    install(EXPORT amgclTargets
        FILE amgcl-targets.cmake
        NAMESPACE amgcl::
        DESTINATION share/amgcl/cmake
        )

    install(
        FILES cmake/amgcl-config.cmake
        DESTINATION share/amgcl/cmake
        )
endif()
