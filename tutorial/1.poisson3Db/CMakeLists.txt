add_executable(poisson3Db poisson3Db.cpp)
target_link_libraries(poisson3Db amgcl)

add_executable(poisson3Db_mixed poisson3Db.cpp)
target_link_libraries(poisson3Db_mixed amgcl)
target_compile_definitions(poisson3Db_mixed PRIVATE MIXED_PRECISION)

if (TARGET cuda_target)
    cuda_add_executable(poisson3Db_cuda poisson3Db_cuda.cu)
    target_link_libraries(poisson3Db_cuda amgcl cuda_target)
endif()
