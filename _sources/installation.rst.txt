.. _installation:

Installation
############

Requirements
************
Installation requires [Conda](https://conda.io) package manager, e.g. one can use [miniforge](https://github.com/conda-forge/miniforge).

Dependencies
************
To avoid reinventing the wheel and allow the developers to focus on implementing specific microseismic monitoring routines, FraCSPy leverages 
numerous python packages, some generic and some that are geophysics-focussed. Below are some of the key dependencies:

* Python 3.9 or greater
* `NumPy <http://www.numpy.org>`_
* `SciPy <http://www.scipy.org/scipylib/index.html>`_
* `PyLops <https://pylops.readthedocs.io/en/stable/>`_
* `ObsPy <https://docs.obspy.org/>`_

We are incredibly grateful to the developers of these packages, giving us a higher starting point for the development of FraCSPy.


Step-by-step installation for users
***********************************
There are multiple ways in which FraCSPy can be installed. The recommended way is directly via the pip platform,

.. code-block:: bash

    >> pip install fracspy


within whatever coding environment you wish. This will pull the latest stable release and install it correctly within your working environment.

If you wish to install the package in a more manual fashion, the source files can be pulled from github and then installed directly. Below we explain how to
do this on both a Linux/Mac environment and on a Windows environment.

**Linux & Mac**

For Linux/Mac, you can leverage the MakeFile and simply run from the top directory of FraCSPy.

.. code-block:: bash

    >> make install


It will create a new conda environment `fracspy` with all the required packages:

Similarly, on Linux you can run:

.. code-block:: bash

    >> ./install.sh


**Windows**

On Windows, the best way is to use [miniforge](https://github.com/conda-forge/miniforge) prompt and run:

.. code-block:: bash

    >> install.bat


It will install the package to environment `fracspy` and activate it.



Now you are ready to use the package.

.. _DevInstall:

Step-by-step installation for developers
****************************************
For developers, FraCSPy can be installed such that any changes made are directly implemented in your local workspace. As before,
we provide instructions on how this can be done in Linux/Mac and Windows. For both options, you will need to clone the repository
from github. (It may be that you want to go off the *-dev branch, such that you have all the latest changes.)

**Linux & Mac**

Option One: Local installation into the current working environment

.. code-block:: bash

    >> make dev-install

Option Two: Create a specific conda environment and install FraCSPy (and dependencies) into that environment

.. code-block:: bash

    >> make dev-install_conda


**Windows**

.. code-block:: bash

    >> install-dev.bat



Uninstall Package
*****************
If you need to add/change packages:

.. code-block:: bash

    >> conda deactivate
    >> conda remove -n fracspy -all

