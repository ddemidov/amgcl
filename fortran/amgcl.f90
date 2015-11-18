module amgcl
    use iso_c_binding

    type, bind(C) :: conv_info
        integer (c_int)    :: iterations
        real    (c_double) :: residual
    end type

    interface
        integer(c_size_t) &
        function amgcl_params_create() bind (C, name="amgcl_params_create")
            use iso_c_binding
        end function

        subroutine amgcl_params_seti(prm, name, val) bind (C, name="amgcl_params_seti")
            use iso_c_binding
            integer   (c_size_t), value :: prm
            character (c_char)          :: name(*)
            integer   (c_int),    value :: val
        end subroutine

        subroutine amgcl_params_setf(prm, name, val) bind (C, name="amgcl_params_setf")
            use iso_c_binding
            integer   (c_size_t), value :: prm
            character (c_char)          :: name(*)
            real      (c_float),  value :: val
        end subroutine

        subroutine amgcl_params_sets(prm, name, val) bind (C, name="amgcl_params_sets")
            use iso_c_binding
            integer   (c_size_t), value :: prm
            character (c_char)          :: name(*)
            character (c_char)          :: val(*)
        end subroutine

        subroutine amgcl_params_destroy(prm) bind(C, name="amgcl_params_destroy")
            use iso_c_binding
            integer (c_size_t), value :: prm
        end subroutine

        integer(c_size_t) &
        function amgcl_solver_create (n, ptr, col, val, prm) bind (C, name="amgcl_solver_create_f")
            use iso_c_binding
            integer (c_int),    value :: n
            integer (c_int)           :: ptr(*)
            integer (c_int)           :: col(*)
            real    (c_double)        :: val(*)
            integer (c_size_t), value :: prm
        end function

        type(conv_info) &
        function amgcl_solver_solve(solver, rhs, x) bind (C, name="amgcl_solver_solve")
            use iso_c_binding
            integer (c_size_t), value :: solver
            real    (c_double)        :: rhs(*)
            real    (c_double)        :: x(*)

            type, bind(C) :: conv_info
                integer (c_int)    :: iterations;
                real    (c_double) :: residual
            end type
        end function

        subroutine amgcl_solver_report(solver) bind(C, name="amgcl_solver_report")
            use iso_c_binding
            integer (c_size_t), value :: solver
        end subroutine

        subroutine amgcl_solver_destroy(solver) bind(C, name="amgcl_solver_destroy")
            use iso_c_binding
            integer (c_size_t), value :: solver
        end subroutine
    end interface
end module
