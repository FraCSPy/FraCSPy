Overview
========
FraCSPy (Framework for Conventional microSeismic Processing) is an open-source Python library focused on providing conventional microseismic monitoring tools, particularly for the purpose of benchmarking
newly proposed algorithms and workflows. 

We invite you to visit our `GitHub project page <https://github.com/FraCSPy>`_


Available Tools
---------------
In this first version, FraCSPy focuses on traditional monitoring tools (i.e., non-Machine Learning procedures). 
It has routines that cover the following elements of the microseismic monitoring pipeline: 

- detection,
- localisation, and
- characterisation: moment tensor determination and moment magnitude computation.

Alongside these core features, there are a number of functions available for data processing and for visualisation. 


Package Design
--------------
The package is designed with core classes for each element of the microseismic pipeline (e.g., detection, location, etc.) with
subroutines contained in these classes that follow conventional methods. For example, the location class has subroutines for 
diffraction stack imaging, least-squares imaging and cross-correlation-based Kirchhoff imaging, among others. First, the 
location object is created

.. code-block:: python

    >> L = fracspy.location.Location(x, y, z)

then the selected methodology is applied, using the `kind` feature,

.. code-block:: python

    >> inv, inv_hc = L.apply(vz, kind="lsi", Op=Op, nforhc=10, verbose=False)


Building the package in this manner keeps the codebase relatively clean and allows for easy adaptation/inclusion of different methodologies. 

The ability to include multiple methodologies is a core concept behind which FraCSPy is created. To date, there is no *one* 
detection/location/characterisation methodology which is prized above all others. And, therefore, we wish to include as many conventionally 
used methodologies as possible - allowing the users to choose which is right for the specific task and dataset.



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started

   self
   installation.rst
   gallery/index.rst
   tutorials/index.rst
   
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation

   api/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved

   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Citing <citing.rst>
   Credits <credits.rst>
