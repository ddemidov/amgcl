# - Try to find VexCL
#
# To set manually the paths, define these environment variables:
# VexCL_INCPATH    - Include path (e.g. VexCL_INCPATH=/home/username/src/vexcl)
#
# Once done this will define
#  VEXCL_FOUND        - system has VexCL
#  VEXCL_INCLUDE_DIRS - the VexCL include directory
#

find_package(PackageHandleStandardArgs)

find_path(VEXCL_INCLUDE_DIRS vexcl/vexcl.hpp
	PATHS
	/usr/local/include
	/usr/include
	ENV VexCL_INCPATH)

find_package_handle_standard_args(VexCL DEFAULT_MSG VEXCL_INCLUDE_DIRS)
