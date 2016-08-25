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

class amg(pyamgcl_ext.amg):
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

        pyamgcl_ext.amg.__init__(self, Acsr.indptr, Acsr.indices, Acsr.data, prm)

class relaxation(pyamgcl_ext.relaxation):
    """
    Single-level relaxation to be used as a preconditioner
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

        pyamgcl_ext.relaxation.__init__(self, Acsr.indptr, Acsr.indices, Acsr.data, prm)
