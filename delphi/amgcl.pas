{
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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

    TCoarsening = (
        coarseningRugeStuben          = 0,
        coarseningAggregation         = 1,
        coarseningSmoothedAggregation = 2,
        coarseningSmoothedAggrEMin    = 3
        );

    TRelaxation = (
        relaxationGaussSeidel  = 0,
        relaxationILU0         = 1,
        relaxationDampedJacobi = 2,
        relaxationSPAI0        = 3,
        relaxationChebyshev    = 4
        );

    TSolverType = (
        solverCG        = 0,
        solverBiCGStab  = 1,
        solverBiCGStabL = 2,
        solverGMRES     = 3
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

    TConvInfo = record
	iterations: Integer;
	residual:   Double;
    end;

    TSolver = class
        private
            h: Pointer;

        public
            constructor Create(
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

            function solve(
                var rhs: Array of Double;
                var x:   Array of Double
                ) : TConvInfo;
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

    amgcl_params_create: function: Pointer; cdecl;

    amgcl_params_seti: procedure(
        p:     Pointer;
        name:  PChar;
        value: Integer
        ); cdecl;

    amgcl_params_setf: procedure(
        p:     Pointer;
        name:  PChar;
        value: Single
        ); cdecl;

    amgcl_params_destroy: procedure(p: Pointer); cdecl;

    amgcl_solver_create: function(
        coarsening:  TCoarsening;
        relaxation:  TRelaxation;
        solver_type: TSolverType;
        prm:         Pointer;
        n:           Integer;
        ptr:         PInteger;
        col:         PInteger;
        val:         PDouble
        ): Pointer; cdecl;

    amgcl_solver_solve: function(
        h:   Pointer;
        rhs: PDouble;
        x:   PDouble
        ): TConvInfo; cdecl;

    amgcl_solver_solve_mtx: procedure(
        h:     Pointer;
        A_ptr: PInteger;
        A_col: PInteger;
        A_val: PDouble;
        rhs:   PDouble;
        x:     PDouble
        ); cdecl;

    amgcl_solver_destroy: procedure(h: Pointer); cdecl;

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
    h := amgcl_solver_create(
                coarsening, relaxation, solver_type, prm.h,
                n, @ptr[0], @col[0], @val[0]
                );
end;

function TSolver.solve(
    var rhs: Array of Double;
    var x:   Array of Double
    ): TConvInfo;
begin
    solve := amgcl_solver_solve(h, @rhs[0], @x[0]);
end;

destructor TSolver.Destroy;
begin
    amgcl_solver_destroy(h);
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

        @amgcl_solver_create    := get_function('amgcl_solver_create');
        @amgcl_solver_solve     := get_function('amgcl_solver_solve');
        @amgcl_solver_solve_mtx := get_function('amgcl_solver_solve_mtx');
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
