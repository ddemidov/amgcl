# Pastix requires Scotch or Metis (partitioning and reordering tools)

if (NOT (Scotch_INCLUDES AND Scotch_LIBRARIES))
    find_path(Scotch_INCLUDES
        NAMES scotch.h
        PATHS $ENV{SCOTCHDIR}
        PATH_SUFFIXES scotch
        )

 
    find_library(SCOTCH_LIBRARY       scotch      PATHS $ENV{SCOTCHDIR})
    find_library(PTSCOTCH_LIBRARY     ptscotch    PATHS $ENV{SCOTCHDIR})
    find_library(SCOTCHERR_LIBRARY    scotcherr   PATHS $ENV{SCOTCHDIR})
    find_library(PTSCOTCHERR_LIBRARY  ptscotcherr PATHS $ENV{SCOTCHDIR})

    set(Scotch_LIBRARIES
        "${SCOTCH_LIBRARY};${PTSCOTCH_LIBRARY};${SCOTCHERR_LIBRARY};${PTSCOTCHERR_LIBRARY}"
        )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Scotch DEFAULT_MSG Scotch_INCLUDES Scotch_LIBRARIES)

mark_as_advanced(Scotch_INCLUDES Scotch_LIBRARIES)
