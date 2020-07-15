program poisson
    use, intrinsic :: iso_c_binding
    use amgcl
    implicit none

    integer :: n, n2, idx, nnz, i, j
    integer(c_int), allocatable :: ptr(:), col(:)
    real(c_double), allocatable :: val(:), rhs(:), x(:)
    integer(c_size_t) :: solver, params
    type(conv_info) :: cnv

    ! Assemble matrix in CRS format for a Poisson problem in n x n square.
    n  = 256
    n2 = n * n

    allocate(ptr(n2 + 1))
    allocate(col(n2 * 5))
    allocate(val(n2 * 5))
    ptr(1) = 1

    idx = 1
    nnz = 0
    do i = 1,n
        do j = 1,n
            if (i > 1) then
                nnz = nnz + 1
                col(nnz) = idx - n
                val(nnz) = -1
            end if

            if (j > 1) then
                nnz = nnz + 1
                col(nnz) = idx - 1
                val(nnz) = -1
            end if

            nnz = nnz + 1
            col(nnz) = idx
            val(nnz) = 4

            if (j < n) then
                nnz = nnz + 1
                col(nnz) = idx + 1
                val(nnz) = -1
            end if

            if (i < n) then
                nnz = nnz + 1
                col(nnz) = idx + n
                val(nnz) = -1
            end if

            idx = idx + 1
            ptr(idx) = nnz + 1
        end do
    end do

    allocate(rhs(n2))
    allocate(x(n2))
    rhs = 1
    x = 0

    ! Create solver parameters.
    params = amgcl_params_create()
    call amgcl_params_sets(params, "solver.type", "cg")
    call amgcl_params_setf(params, "solver.tol", 1e-6)

    ! Create solver, printout its structure.
    solver = amgcl_solver_create(n2, ptr, col, val, params)
    call amgcl_solver_report(solver)

    ! Solve the problem for the given right-hand-side.
    cnv = amgcl_solver_solve(solver, rhs, x)
    write(*,"('Iterations:', I3, ', residual: ', E13.6)") cnv%iterations, cnv%residual

    ! Solve the same problem with explicitly provided matrix.
    cnv = amgcl_solver_solve_mtx(solver, ptr, col, val, rhs, x)
    write(*,"('Iterations:', I3, ', residual: ', E13.6)") cnv%iterations, cnv%residual

    ! Destroy solver and parameter pack.
    call amgcl_solver_destroy(solver)
    call amgcl_params_destroy(params)
end program poisson
