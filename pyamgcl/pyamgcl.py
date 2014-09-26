import numpy
import pyamgcl_ext
from pyamgcl_ext import coarsening, relaxation, solver_type
from scipy.sparse.linalg import LinearOperator

class make_solver:
    """
    Iterative solver preconditioned by algebraic multigrid

    The class builds algebraic multigrid hierarchy for the given matrix and
    uses the hierarchy as a preconditioner for the specified iterative solver.
    """

    def __init__(self,
            A,
            coarsening=pyamgcl_ext.coarsening.smoothed_aggregation,
            relaxation=pyamgcl_ext.relaxation.spai0,
            solver=pyamgcl_ext.solver_type.bicgstabl,
            prm={}
            ):
        """
        Class constructor.

        Creates algebraic multigrid hierarchy.

        Parameters
        ----------
        A : the system matrix in scipy.sparse format
        coarsening : {ruge_stuben, aggregation, *smoothed_aggregation*, smoothed_aggr_emin}
            The coarsening type to use for construction of the multigrid
            hierarchy.
        relaxation : {damped_jacobi, gauss_seidel, chebyshev, *spai0*, ilu0}
            The relaxation scheme to use for multigrid cycles.
        solver : {cg, bicgstab, *bicgstabl*, gmres}
            The iterative solver to use.
        prm : dictionary with amgcl parameters
        """
        Acsr = A.tocsr()

        self.S = pyamgcl_ext.make_solver(
                coarsening, relaxation, solver, prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64)
                )

    def __repr__(self):
        """
        Provides information about the multigrid hierarchy.
        """
        return self.S.__repr__()

    def __call__(self, rhs):
        """
        Solves the system for the given right-hand side.

        Parameters
        ----------
        rhs : the right-hand side
        """
        return self.S(rhs.astype(numpy.float64))

class make_preconditioner(LinearOperator):
    """
    Algebraic multigrid hierarchy that may be used as a preconditioner with
    scipy iterative solvers.
    """
    def __init__(self,
            A,
            coarsening=pyamgcl_ext.coarsening.smoothed_aggregation,
            relaxation=pyamgcl_ext.relaxation.spai0,
            prm={}
            ):
        """
        Class constructor.

        Creates algebraic multigrid hierarchy.

        Parameters
        ----------
        A : the system matrix in scipy.sparse format
        coarsening : {ruge_stuben, aggregation, *smoothed_aggregation*, smoothed_aggr_emin}
            The coarsening type to use for construction of the multigrid
            hierarchy.
        relaxation : {damped_jacobi, gauss_seidel, chebyshev, *spai0*, ilu0}
            The relaxation scheme to use for multigrid cycles.
        prm : dictionary with amgcl parameters
        """
        Acsr = A.tocsr()

        self.P = pyamgcl_ext.make_preconditioner(
                coarsening, relaxation, prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64)
                )

        LinearOperator.__init__(self, A.shape, self.P)

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

