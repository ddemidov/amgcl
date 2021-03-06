# Pastix requires Scotch or Metis (partitioning and reordering tools)

if (NOT (Scotch_INCLUDES AND Scotch_LIBRARIES))
    find_path(Scotch_INCLUDES
        NAMES scotch.h
        PATHS $ENV{SCOTCHDIR}
        PATH_SUFFIXES scotch
        )

    find_library(SCOTCH_LIBRARY       scotch      PATHS $ENV{SCOTCHDIR})
    find_library(SCOTCHERR_LIBRARY    scotcherr   PATHS $ENV{SCOTCHDIR})
endif()

if (NOT (PTScotch_INCLUDES AND PTScotch_LIBRARIES))
    find_path(PTScotch_INCLUDES
        NAMES ptscotch.h
        PATHS $ENV{PTSCOTCHDIR}
        PATH_SUFFIXES scotch ptscotch
        )

    find_library(PTSCOTCH_LIBRARY     ptscotch    PATHS $ENV{PTSCOTCHDIR})
    find_library(PTSCOTCHERR_LIBRARY  ptscotcherr PATHS $ENV{PTSCOTCHDIR})
endif()

if (Scotch_INCLUDES AND SCOTCH_LIBRARY AND SCOTCHERR_LIBRARY AND
        PTScotch_INCLUDES AND PTSCOTCH_LIBRARY AND PTSCOTCHERR_LIBRARY)
    set(Scotch_LIBRARIES
        "${SCOTCH_LIBRARY};${SCOTCHERR_LIBRARY};${PTSCOTCH_LIBRARY};${PTSCOTCHERR_LIBRARY}"
        )
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
    Scotch DEFAULT_MSG
    Scotch_INCLUDES PTScotch_INCLUDES
    Scotch_LIBRARIES PTScotch_LIBRARIES
    )

mark_as_advanced(
    Scotch_INCLUDES
    Scotch_LIBRARIES
    PTScotch_INCLUDES
    PTScotch_LIBRARIES
    )
