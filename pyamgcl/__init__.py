import numpy
import scipy
from . import pyamgcl_ext
from scipy.sparse.linalg import LinearOperator

class make_solver:
    """
    Iterative solver preconditioned by algebraic multigrid

    The class builds algebraic multigrid hierarchy for the given matrix and
    uses the hierarchy as a preconditioner for the specified iterative solver.
    """

    def __init__(self, A, prm={}):
        """
        Class constructor.

        Creates algebraic multigrid hierarchy.

        Parameters
        ----------
        A     The system matrix in scipy.sparse format
        prm   Dictionary with amgcl parameters
        """
        Acsr = A.tocsr()

        self.S = pyamgcl_ext.make_solver(
                prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64)
                )

    def __repr__(self):
        """
        Provides information about the multigrid hierarchy.
        """
        return self.S.__repr__()

    def __call__(self, *args):
        """
        Solves the system for the given system matrix and the right-hand side.

        In case single argument is given, it is considered to be the right-hand
        side. The matrix given at the construction is used for solution.

        In case two arguments are given, the first one should be a new system
        matrix, and the second is the right-hand side. In this case the
        multigrid hierarchy initially built at construction is still used as a
        preconditioner. This may be of use for solution of non-steady-state
        PDEs, where the discretized system matrix slightly changes on each time
        step, but multigrid hierarchy built for one of previous time steps is
        still able to work as a decent preconditioner.  Thus time needed for
        hierarchy reconstruction is saved.

        Parameters
        ----------
        A : the new system matrix (optional)
        rhs : the right-hand side
        """
        if len(args) == 1:
            return self.S( args[0].astype(numpy.float64) )
        elif len(args) == 2:
            Acsr = args[0].tocsr()

            return self.S(
                    Acsr.indptr.astype(numpy.int32),
                    Acsr.indices.astype(numpy.int32),
                    Acsr.data.astype(numpy.float64),
                    args[1].astype(numpy.float64)
                    )
        else:
            raise "Wrong number of arguments"

    @property
    def iters(self):
        """
        Returns iterations made during last solve
        """
        return self.S.iters

    @property
    def error(self):
        """
        Returns relative error achieved during last solve
        """
        return self.S.error

class make_preconditioner(LinearOperator):
    """
    Algebraic multigrid hierarchy that may be used as a preconditioner with
    scipy iterative solvers.
    """
    def __init__(self, A, prm={}):
        """
        Class constructor.

        Creates algebraic multigrid hierarchy.

        Parameters
        ----------
        A : the system matrix in scipy.sparse format
        prm : dictionary with amgcl parameters
        """
        Acsr = A.tocsr()

        self.P = pyamgcl_ext.make_preconditioner(
                prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64)
                )

        if [int(v) for v in scipy.__version__.split('.')] < [0, 16, 0]:
            LinearOperator.__init__(self, A.shape, self.P)
        else:
            LinearOperator.__init__(self, dtype=numpy.float64, shape=A.shape)

    def __repr__(self):
        """
        Provides information about the multigrid hierarchy.
        """
        return self.P.__repr__()

    def __call__(self, x):
        """
        Preconditions the given vector.
        """
        return self.P(x.astype(numpy.float64))

    def _matvec(self, x):
        """
        Preconditions the given vector.
        """
        return self.__call__(x)

class make_cpr(LinearOperator):
    """
    CPR preconditioner.
    """
    def __init__(self, A, pmask, prm={}):
        """
        Class constructor.

        Parameters
        ----------
        A : the system matrix in scipy.sparse format
        prm : dictionary with amgcl parameters
        """
        Acsr = A.tocsr()

        self.P = pyamgcl_ext.make_cpr(
                prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64),
                pmask.astype(numpy.int32)
                )

        if [int(v) for v in scipy.__version__.split('.')] < [0, 16, 0]:
            LinearOperator.__init__(self, A.shape, self.P)
        else:
            LinearOperator.__init__(self, dtype=numpy.float64, shape=A.shape)

    def __call__(self, x):
        """
        Preconditions the given vector.
        """
        return self.P(x.astype(numpy.float64))

    def _matvec(self, x):
        """
        Preconditions the given vector.
        """
        return self.__call__(x)

class make_simple(LinearOperator):
    """
    SIMPLE preconditioner.
    """
    def __init__(self, A, pmask, prm={}):
        """
        Class constructor.

        Parameters
        ----------
        A : the system matrix in scipy.sparse format
        prm : dictionary with amgcl parameters
        """
        Acsr = A.tocsr()

        self.P = pyamgcl_ext.make_simple(
                prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64),
                pmask.astype(numpy.int32)
                )

        if [int(v) for v in scipy.__version__.split('.')] < [0, 16, 0]:
            LinearOperator.__init__(self, A.shape, self.P)
        else:
            LinearOperator.__init__(self, dtype=numpy.float64, shape=A.shape)

    def __call__(self, x):
        """
        Preconditions the given vector.
        """
        return self.P(x.astype(numpy.float64))

    def _matvec(self, x):
        """
        Preconditions the given vector.
        """
        return self.__call__(x)

