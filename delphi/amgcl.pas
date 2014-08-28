{
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

\file   delphi/amgc.pas
\author Denis Demidov <dennis.demidov@gmail.com>
\brief  Delphi bindings for AMGCL
}
unit amgcl;

interface

type
    {$Z4}
    TBackend  = (
        backendBuiltin  = 1,
        backendBlockCRS = 2
    );

    {$Z4}
    TCoarsening = (
        coarseningRugeStuben          = 1,
        coarseningAggregation         = 2,
        coarseningSmoothedAggregation = 3,
        coarseningSmoothedAggrEMin    = 4
        );

    {$Z4}
    TRelaxation = (
        relaxationDampedJacobi = 1,
        relaxationGaussSeidel  = 2,
        relaxationChebyshev    = 3,
        relaxationSPAI0        = 4,
        relaxationILU0         = 5
        );

    {$Z4}
    TSolverType = (
        solverCG        = 1,
        solverBiCGStab  = 2,
        solverBiCGStabL = 3,
        solverGMRES     = 4
        );

    TParams = class
        private
            h: Pointer;

        public
            constructor Create;
            destructor  Destroy; override;

            procedure setprm(name: string; value: Integer); overload;
            procedure setprm(name: string; value: Single);  overload;
    end;

    TSolver = class
        private
            hp: Pointer;
            hs: Pointer;

        public
            constructor Create(
                backend:     TBackend;
                coarsening:  TCoarsening;
                relaxation:  TRelaxation;
                solver_type: TSolverType;
                prm:         TParams;
                n:           Integer;
                var ptr:     Array of Integer;
                var col:     Array of Integer;
                var val:     Array of Double
                );

            destructor Destroy; override;

            procedure solve(
                var rhs: Array of Double;
                var x:   Array of Double
                );

	    function iters: Integer;
	    function resid: Double;
    end;

    procedure load;
    procedure unload;

implementation

uses Windows, SysUtils;

const
    DLLName = 'amgcl.dll';

Type
    PInteger = ^Integer;
    PDouble  = ^Double;
    EFuncNotFound = class(Exception);

Var
    hlib: Integer;

    amgcl_params_create: function: Pointer; stdcall;

    amgcl_params_seti: procedure(
        p:     Pointer;
        name:  PChar;
        value: Integer
        ); stdcall;

    amgcl_params_setf: procedure(
        p:     Pointer;
        name:  PChar;
        value: Single
        ); stdcall;

    amgcl_params_destroy: procedure(p: Pointer); stdcall;

    amgcl_precond_create: function(
        backend:     TBackend;
        coarsening:  TCoarsening;
        relaxation:  TRelaxation;
        prm:         Pointer;
        n:           Integer;
        ptr:         PInteger;
        col:         PInteger;
        val:         PDouble
        ): Pointer; stdcall;

    amgcl_precond_destroy: procedure(p: Pointer); stdcall;

    amgcl_solver_create: function(
        backend:     TBackend;
        solver_type: TSolverType;
        prm:         Pointer;
        n:           Integer
        ): Pointer; stdcall;

    amgcl_solver_solve: procedure(
        hs:  Pointer;
        hp:  Pointer;
        rhs: PDouble;
        x:   PDouble
        ); stdcall;

    amgcl_solver_solve_mtx: procedure(
        hs:    Pointer;
        A_ptr: PInteger;
        A_col: PInteger;
        A_val: PDouble;
        hp:    Pointer;
        rhs:   PDouble;
        x:     PDouble
        ); stdcall;

    amgcl_solver_get_iters: function(hs: Pointer): Integer; stdcall;
    amgcl_solver_get_resid: function(hs: Pointer): Double;  stdcall;

    amgcl_solver_destroy: procedure(p: Pointer); stdcall;

constructor TParams.Create;
begin
    h := amgcl_params_create;
end;

procedure TParams.setprm(name: string; value: Integer);
begin
    amgcl_params_seti(h, PChar(name), value);
end;

procedure TParams.setprm(name: string; value: Single);
begin
    amgcl_params_setf(h, PChar(name), value);
end;

destructor TParams.Destroy;
begin
    amgcl_params_destroy(h);
end;

constructor TSolver.Create(
    backend:     TBackend;
    coarsening:  TCoarsening;
    relaxation:  TRelaxation;
    solver_type: TSolverType;
    prm:         TParams;
    n:           Integer;
    var ptr:     Array of Integer;
    var col:     Array of Integer;
    var val:     Array of Double
    );
begin
    hp := amgcl_precond_create(backend, coarsening, relaxation, prm.h, n, @ptr[0], @col[0], @val[0]);
    hs := amgcl_solver_create(backend, solver_type, prm.h, n);
end;

procedure TSolver.solve(
    var rhs: Array of Double;
    var x:   Array of Double
    );
begin
    amgcl_solver_solve(hs, hp, @rhs[0], @x[0]);
end;

function TSolver.iters: Integer;
begin
    iters := amgcl_solver_get_iters(hs);
end;

function TSolver.resid: Double;
begin
    resid := amgcl_solver_get_resid(hs);
end;

destructor TSolver.Destroy;
begin
    amgcl_precond_destroy(hp);
    amgcl_solver_destroy(hs);
end;

procedure load;
    function get_function(name: PChar): TFarProc;
    var
        res: TFarProc;
    begin
        res := GetProcAddress(hlib, name);
        if res = nil then
            raise EFuncNotFound.Create('Entry point to ' + name + ' not found');
        get_function := res;
    end;
begin
    hlib := LoadLibrary(DLLName);
    if hlib = 0 then raise Exception.Create('Failed to load ' + DLLName);

    try
        @amgcl_params_create    := get_function('amgcl_params_create');
        @amgcl_params_seti      := get_function('amgcl_params_seti');
        @amgcl_params_setf      := get_function('amgcl_params_setf');
        @amgcl_params_destroy   := get_function('amgcl_params_destroy');

        @amgcl_precond_create   := get_function('amgcl_precond_create');
        @amgcl_precond_destroy  := get_function('amgcl_precond_destroy');

        @amgcl_solver_create    := get_function('amgcl_solver_create');
        @amgcl_solver_solve     := get_function('amgcl_solver_solve');
        @amgcl_solver_solve_mtx := get_function('amgcl_solver_solve_mtx');
        @amgcl_solver_get_iters := get_function('amgcl_solver_get_iters');
        @amgcl_solver_get_resid := get_function('amgcl_solver_get_resid');
        @amgcl_solver_destroy   := get_function('amgcl_solver_destroy');
    except
        on e: Exception do begin
            FreeLibrary(hlib);
            hlib := 0;
            raise Exception.Create('Failed to load ' + DLLName +
                '. Reason: ' + e.Message);
        end;
    end;
end;

procedure unload;
begin
    if hlib <> 0 then begin
        FreeLibrary(hlib);
        hlib := 0;
    end;
end;

end.
