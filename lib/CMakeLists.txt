set(AMGCL_LIB_SRC amgcl.cpp)
if (WIN32)
    set(AMGCL_LIB_SRC ${AMGCL_LIB_SRC} dll.def)
endif()

add_library(amgcl ${AMGCL_LIB_SRC})
