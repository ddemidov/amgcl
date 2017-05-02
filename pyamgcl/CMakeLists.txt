if (NOT WIN32)
    add_subdirectory(pybind11)

    pybind11_add_module(pyamgcl_ext pyamgcl.cpp)
    target_link_libraries(pyamgcl_ext PRIVATE amgcl::amgcl)

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        COPYONLY
        )
endif()
