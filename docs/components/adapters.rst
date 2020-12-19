Matrix Adapters
===============

A matrix adapter allows AMGCL to construct a solver from some common matrix
formats. Internally, the CRS_ format is used, but it is easy to adapt any
matrix format that allows row-wise iteration over its non-zero elements.

.. _CRS: http://netlib.org/linalg/html_templates/node91.html
