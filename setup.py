from __future__ import absolute_import
from __future__ import print_function

import io
import re
import os
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()

setup_requirements = ["pytest-runner", "setuptools_scm"]


# Install requirements
install_requires = [
    "torch"
]

test_deps = [
    "torch",
    "pytest==4.5.0",
    "mock==3.0.5"
          ]
# Test requirements
extras_require = {
    'test': test_deps
    }


setup(
    author='Claire Emma Birnie',
    author_email="claire.birnie@kaust.edu.sa",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Unlicense",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6"
    ],
    description="A python package for general microseismic modelling, monitoring and analysis",
    install_requires=install_requires,
    license="Unlicense",
    name="pyfrac",
    packages=find_packages(exclude=['pytests']),
    include_package_data=True,
    zip_safe=True,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_deps,
    extras_require=extras_require,
    long_description=open('README.md').read(),
    version='0.0.1',
    #use_scm_version={
    #    "write_to": "pydenoise/_version.py",
    #    "relative_to": __file__,
    #},
)
