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

    foreach(py __init__.py test_pyamgcl)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/${py}
            ${CMAKE_CURRENT_BINARY_DIR}/${py}
            COPYONLY
            )
    endforeach()

    add_test(test_pyamgcl test_pyamgcl)
endif()
