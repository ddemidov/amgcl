cmake_minimum_required(VERSION 3.7)
project(doc)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/conf.py
    ${CMAKE_CURRENT_BINARY_DIR}/conf.py
    COPYONLY
    )

add_custom_target(doc
    COMMAND sphinx-build -b html -c ${CMAKE_CURRENT_SOURCE_DIR} . ${CMAKE_CURRENT_BINARY_DIR}/html
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generate HTML documentation"
    )
