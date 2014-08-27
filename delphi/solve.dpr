program solve;

{$APPTYPE CONSOLE}

uses
    SysUtils,
    amgcl in 'amgcl.pas';

const
    m   = 256;
    n   = m * m;
    nnz = m * m + 4 * (m - 2) * (m - 2);
var
    i, j, head, idx: Integer;
    ptr, col: Array of Integer;
    val, rhs, x: Array of Double;

    prm:   amgcl.TParams;
    solver: amgcl.TSolver;
begin
    try
        amgcl.load;
    except
        on e: Exception do begin
            writeln(e.Message);
            exit;
        end;
    end;

    SetLength(ptr, n + 1);
    SetLength(col, nnz);
    SetLength(val, nnz);
    SetLength(rhs, n);
    SetLength(x,   n);

    ptr[0] := 0;
    idx    := 0;
    head   := 0;

    for j := 0 to m - 1 do begin
        for i := 0 to m - 1 do begin
            if (i = 0) or (i = m - 1) or (j = 0) or (j = m - 1) then
            begin
                col[head] := idx;
                val[head] := 1;
                rhs[idx]  := 0;

                head := head + 1;
            end else begin
                col[head+0] := idx - m;
                col[head+1] := idx - 1;
                col[head+2] := idx;
                col[head+3] := idx + 1;
                col[head+4] := idx + m;

                val[head+0] := -1;
                val[head+1] := -1;
                val[head+2] :=  4;
                val[head+3] := -1;
                val[head+4] := -1;

                rhs[idx] := 1;

                head := head + 5;
            end;

            x[idx] := 0;

            idx := idx + 1;
            ptr[idx] := head;
        end
    end;

    prm := amgcl.TParams.Create;

    solver := amgcl.TSolver.Create(
        amgcl.backendBuiltin,
        amgcl.coarseningSmoothedAggregation,
        amgcl.relaxationSPAI0,
        amgcl.solverBiCGStabL,
        prm, n, ptr, col, val
        );

    solver.solve(rhs, x);

    solver.free;
    prm.free;

    amgcl.unload;
end.

