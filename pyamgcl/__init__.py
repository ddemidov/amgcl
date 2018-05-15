import numpy
import scipy
from . import pyamgcl_ext
from scipy.sparse.linalg import LinearOperator

class solver(pyamgcl_ext.solver):
    """
    Iterative solver with preconditioning
    """
    def __init__(self, P, prm={}):
        self.P = P
        pyamgcl_ext.solver.__init__(self, self.P, prm)

    def __repr__(self):
        return self.P.__repr__()

    def __call__(self, *args):
        """
        Solves the system for the given system matrix and the right-hand side.

        In case single argument is given, it is considered to be the right-hand
        side. The matrix given at the construction is used for solution.

        In case two arguments are given, the first one should be a new system
        matrix, and the second is the right-hand side. In this case the
        preconditioner passed on construction of the solver is still used. This
        may be of use for solution of non-steady-state PDEs, where the
        discretized system matrix slightly changes on each time step, but the
        preconditioner built for one of previous time steps is still able to
        approximate the system matrix.  This saves time needed for rebuilding
        the preconditioner.

        Parameters
        ----------
        A : the new system matrix (optional)
        rhs : the right-hand side
        """
        if len(args) == 1:
            return pyamgcl_ext.solver.__call__(self, args[0])
        elif len(args) == 2:
            Acsr = args[0].tocsr()
            return pyamgcl_ext.solver.__call__(self, Acsr.indptr, Acsr.indices, Acsr.data, args[1])
        else:
            raise "Wrong number of arguments"

class amgcl(pyamgcl_ext.amgcl):
    """
    Algebraic multigrid hierarchy to be used as a preconditioner
    """
    def __init__(self, A, prm={}):
        """
        Creates algebraic multigrid hierarchy to be used as preconditioner.

        Parameters
        ----------
        A     The system matrix in scipy.sparse format
        prm   Dictionary with amgcl parameters
        """
        Acsr = A.tocsr()
        self.shape = A.shape

        pyamgcl_ext.amgcl.__init__(self, Acsr.indptr, Acsr.indices, Acsr.data, prm)
