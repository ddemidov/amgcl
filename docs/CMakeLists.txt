cmake_minimum_required(VERSION 2.6)
project(doc)

find_package(Doxygen QUIET)

if (DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND sphinx-build -b html -c ${CMAKE_CURRENT_SOURCE_DIR} . ${CMAKE_CURRENT_BINARY_DIR}/html
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generate HTML documentation"
        )
endif()
