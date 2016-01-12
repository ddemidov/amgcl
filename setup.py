from setuptools import setup, Extension
import os, sys
import numpy
from git_version import git_version

pyver = sys.version_info

#----------------------------------------------------------------------------
def find_file(filename, search_dirs):
    for dirname in search_dirs:
        for root, dirs, files in os.walk(dirname):
            for f in files:
                if filename in f:
                    return dirname
            for d in dirs:
                if filename in d:
                    return dirname
            if filename in root:
                return dirname
    return False


def boost_python_lib():
    library_dirs = [
            '/usr/local/lib64/',
            '/usr/lib64/',
            '/usr/local/lib/',
            '/usr/lib/',
            '/opt/local/lib/'
            ]

    boost_python = "boost_python-%s.%s" % (pyver[0], pyver[1])
    if find_file("lib" + boost_python, library_dirs):
        return boost_python

    if pyver >= (3, ):
        boost_python = "boost_python-py%s%s" % (pyver[0], pyver[1])
        if find_file("lib" + boost_python, library_dirs):
            return boost_python
        boost_python = "boost_python%s" % pyver[0]
        if find_file("lib" + boost_python, library_dirs):
            return boost_python

    return "boost_python"


setup(
        name='pyamgcl',
        version=git_version(),
        description='Solution of large sparse linear systems with Algebraic Multigrid Method',
        author='Denis Demidov',
        author_email='dennis.demidov@gmail.com',
        license='MIT',
        url='https://github.com/ddemidov/amgcl',
        include_package_data=True,
        zip_safe=False,
        packages=['pyamgcl'],
        ext_modules=[
            Extension('pyamgcl.pyamgcl_ext', ['pyamgcl/pyamgcl.cpp'],
                include_dirs=['.', numpy.get_include()],
                libraries=[boost_python_lib()],
                extra_compile_args=['-O3']
                )
            ]
)
