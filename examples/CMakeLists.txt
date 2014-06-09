include_directories(${CMAKE_SOURCE_DIR}/tests)

add_executable(solver solver.cpp)
target_link_libraries(solver ${Boost_LIBRARIES})

add_executable(block_crs block_crs.cpp)
target_link_libraries(block_crs ${Boost_LIBRARIES})

add_executable(call_lib call_lib.cpp)
target_link_libraries(call_lib ${Boost_LIBRARIES} amgcl)

if (CUDA_FOUND)
    cuda_add_executable(cuda cuda.cu OPTIONS "-arch=sm_${CUDA_COMPUTE_CAPABILITY}")
    target_link_libraries(cuda ${CUDA_cusparse_LIBRARY} ${Boost_LIBRARIES})
endif()
