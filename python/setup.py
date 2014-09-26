from setuptools import setup, Extension
import os, sys
import numpy

pyver = sys.version_info

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

    path = find_file("libboost_python-%s.%s" % (pyver[0], pyver[1]), library_dirs)
    if path: return path

    if _version >= (3, ):
        path = find_file("libboost_python-py%s%s" % (pyver[0], pyver[1]), library_dirs)
        if path: return path

    return "boost_python"

setup(
        name='pyamgcl',
        version='0.5.0',
        description='Solution of large sparse linear systems with Algebraic Multigrid Method',
        author='Denis Demidov',
        author_email='dennis.demidov@gmail.com',
        url='https://github.com/ddemidov/amgcl',
        include_package_data=True,
        zip_safe=False,
        ext_modules=[
            Extension('pyamgcl',
                ['pyamgcl.cpp'],
                include_dirs=['..', numpy.get_include()],
                libraries=[boost_python_lib()]
                )
            ]
)
