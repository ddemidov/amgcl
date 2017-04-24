add_library(libamgcl
    amgcl.cpp
    $<$<PLATFORM_ID:Windows>:dll.def>
    )
target_link_libraries(libamgcl PUBLIC amgcl)
set_target_properties(libamgcl PROPERTIES PREFIX "")
target_include_directories(libamgcl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (TARGET mpi_target)
    add_library(libamgcl_mpi
        amgcl_mpi.cpp
        $<$<PLATFORM_ID:Windows>:mpi_dll.def>
        )
    set_target_properties(libamgcl_mpi PROPERTIES PREFIX "")
    target_link_libraries(libamgcl_mpi PUBLIC libamgcl mpi_target)

    if (TARGET pastix_target)
        target_link_libraries(libamgcl_mpi PUBLIC pastix_target)
    endif()
endif()
