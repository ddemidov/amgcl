enable_language(Fortran)

add_executable(poisson poisson.f90 amgcl.f90)
target_link_libraries(poisson amgcl)
