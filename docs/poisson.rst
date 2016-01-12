Assembling matrix for Poisson's equation
----------------------------------------

The section provides an example of assembling the system matrix and the
right-hand side for a Poisson's equation in a unit square
:math:`\Omega=[0,1]\times[0,1]`:

.. math::

    -\Delta u = 1, \; u \in \Omega \quad u = 0, \; u \in \partial \Omega

The solution to the problem looks like this:

.. plot::

    from pylab import *
    from numpy import *
    h = linspace(-1, 1, 100)
    x,y = meshgrid(h, h)
    u = 0.5 * (1-x**2)
    for k in range(1,20,2):
        u -= 16/pi**3 * (sin(k*pi*(1+x)/2) / (k**3 * sinh(k * pi))
                * (sinh(k * pi * (1 + y) / 2) + sinh(k * pi * (1 - y)/2)))
    figure(figsize=(3,3))
    imshow(u, extent=(0,1,0,1))
    show()

Here is how the problem may be discretized on a uniform :math:`n \times n`
grid:

.. note: The CRS_ format [Saad03]_ is used for the discretized matrix.

.. _CRS: http://netlib.org/linalg/html_templates/node91.html

.. code-block:: cpp

    #include <vector>

    // Assembles matrix for Poisson's equation with homogeneous
    // boundary conditions on a n x n grid.
    // Returns number of rows in the assembled matrix.
    // The matrix is returned in the CRS components ptr, col, and val.
    // The right-hand side is returned in rhs.
    int poisson(
        int n,
        std::vector<int>    &ptr,
        std::vector<int>    &col,
        std::vector<double> &val,
        std::vector<double> &rhs
        )
    {
        int    n2 = n * n;        // Number of points in the grid.
        double h = 1.0 / (n - 1); // Grid spacing.

        ptr.clear(); ptr.reserve(n2 + 1); ptr.push_back(0);
        col.clear(); col.reserve(n2 * 5); // We use 5-point stencil, so the matrix
        val.clear(); val.reserve(n2 * 5); // will have at most n2 * 5 nonzero elements.

        rhs.resize(n2);

        for(int j = 0, k = 0; j < n; ++j) {
            for(int i = 0; i < n; ++i, ++k) {
                if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                    // Boundary point. Use Dirichlet condition.
                    col.push_back(k);
                    val.push_back(1.0);

                    rhs[k] = 0.0;
                } else {
                    // Interior point. Use 5-point finite difference stencil.
                    col.push_back(k - n);
                    val.push_back(-1.0 / (h * h));

                    col.push_back(k - 1);
                    val.push_back(-1.0 / (h * h));

                    col.push_back(k);
                    val.push_back(4.0 / (h * h));

                    col.push_back(k + 1);
                    val.push_back(-1.0 / (h * h));

                    col.push_back(k + n);
                    val.push_back(-1.0 / (h * h));

                    rhs[k] = 1.0;
                }

                ptr.push_back(col.size());
            }
        }

        return n2;
    }

