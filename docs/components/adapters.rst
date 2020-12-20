Matrix Adapters
===============

A matrix adapter allows AMGCL to construct a solver from some common matrix
formats. Internally, the CRS_ format is used, but it is easy to adapt any
matrix format that allows row-wise iteration over its non-zero elements.

.. _CRS: http://netlib.org/linalg/html_templates/node91.html

Tuple of CRS arrays
-------------------

Zero copy
---------

Block matrix
------------

Scaled system
-------------

Reordered system
----------------

Eigen matrix
------------

Epetra matrix
-------------

uBlas matrix
------------
