find_package(PythonInterp)
find_package(PythonLibs ${PYTHON_VERSION_STRING})
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(NumPy QUIET)
if (NUMPY_FOUND)
    include_directories(${NUMPY_INCLUDE_DIRS})

    find_package(Boost COMPONENTS
        chrono
        system
        filesystem
        python
        )

    add_library(pyamgcl_ext pyamgcl.cpp)
    set_target_properties(pyamgcl_ext PROPERTIES PREFIX "")
    target_link_Libraries(pyamgcl_ext ${Boost_LIBRARIES})

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        COPYONLY
        )

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test_pyamgcl
        ${CMAKE_CURRENT_BINARY_DIR}/../test_pyamgcl
        COPYONLY
        )

    add_test(test_pyamgcl ${CMAKE_CURRENT_BINARY_DIR}/../test_pyamgcl)
endif()
