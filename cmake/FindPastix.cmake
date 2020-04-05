# Pastix lib requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.
# Pastix requires Scotch or Metis (partitioning and reordering tools) as well

if (NOT (Pastix_INCLUDES AND Pastix_LIBRARIES))
    find_path(Pastix_INCLUDES NAMES pastix.h PATHS $ENV{PASTIXDIR})
    find_library(Pastix_LIBRARIES pastix PATHS $ENV{PASTIXDIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Pastix DEFAULT_MSG Pastix_INCLUDES Pastix_LIBRARIES)

mark_as_advanced(Pastix_INCLUDES Pastix_LIBRARIES)
