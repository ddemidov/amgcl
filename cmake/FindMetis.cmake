# Find Metis library

if (NOT (METIS_INCLUDE_DIRS AND METIS_LIBRARIES))
    find_path(METIS_INCLUDE_DIRS
        NAMES metis.h parmetis.h
        PATHS $ENV{METIS_ROOT}
        PATH_SUFFIXES metis include
        )

    find_library(METIS_LIBRARY metis
        PATHS $ENV{METIS_ROOT}
        PATH_SUFFIXES lib
        )

    find_library(PARMETIS_LIBRARY parmetis
        PATHS $ENV{METIS_ROOT}
        PATH_SUFFIXES lib
        )

    set(METIS_LIBRARIES "${PARMETIS_LIBRARY};${METIS_LIBRARY}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    METIS DEFAULT_MSG METIS_INCLUDE_DIRS METIS_LIBRARIES)

mark_as_advanced(METIS_INCLUDE_DIRS METIS_LIBRARIES)

if (METIS_FOUND)
    add_library(Metis_metis INTERFACE)
    add_library(Metis::metis ALIAS Metis_metis)
    target_include_directories(Metis_metis INTERFACE ${METIS_INCLUDE_DIRS})
    target_link_libraries(Metis_metis INTERFACE ${METIS_LIBRARIES})

    if (METIS_LIBRARY)
        target_compile_definitions(Metis_metis INTERFACE AMGCL_HAVE_METIS)
    endif()

    if (PARMETIS_LIBRARY)
        target_compile_definitions(Metis_metis INTERFACE AMGCL_HAVE_PARMETIS)
    endif()
endif()

