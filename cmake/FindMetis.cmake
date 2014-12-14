# Find Metis library

if (NOT (METIS_INCLUDES AND METIS_LIBRARIES))
    find_path(METIS_INCLUDES
        NAMES metis.h
        PATHS $ENV{METISDIR}
        PATH_SUFFIXES metis
        )


    find_library(METIS_LIBRARY metis PATHS $ENV{METISDIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    METIS DEFAULT_MSG METIS_INCLUDES METIS_LIBRARY)

mark_as_advanced(METIS_INCLUDES METIS_LIBRARY)
