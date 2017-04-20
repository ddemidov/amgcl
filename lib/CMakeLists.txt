set(AMGCL_LIB_SRC amgcl.cpp)
if (WIN32)
    set(AMGCL_LIB_SRC ${AMGCL_LIB_SRC} dll.def)
endif()

add_library(libamgcl ${AMGCL_LIB_SRC})
target_link_libraries(libamgcl ${Boost_LIBRARIES})
set_target_properties(libamgcl PROPERTIES PREFIX "")

if (MPI_CXX_FOUND)
    set(AMGCL_MPI_LIB_SRC amgcl_mpi.cpp)
    if (WIN32)
        set(AMGCL_MPI_LIB_SRC ${AMGCL_MPI_LIB_SRC} mpi_dll.def)
    endif()

    add_library(libamgcl_mpi ${AMGCL_MPI_LIB_SRC})
    set_target_properties(libamgcl_mpi PROPERTIES
        PREFIX        ""
        COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
        LINK_FLAGS    "${MPI_CXX_LINK_FLAGS}"
        )
    target_link_libraries(libamgcl_mpi
        libamgcl
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        )
    if (HAVE_PASTIX)
        target_link_libraries(libamgcl_mpi
            ${PASTIX_LIBRARIES}
            ${SCOTCH_LIBRARIES}
            ${BLAS_LIBRARIES}
            )
    endif()
endif()
