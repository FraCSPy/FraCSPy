.. _api:

FraCSPy API
===========

The Application Programming Interface (API) of FraCSPy mirrors the usual microseismic workflow
from pre-processing to source mechanism inversion. It is composed of the following modules:

* Location Determination
* Moment Tensor Inversion


Location Determination
----------------------

Modelling
~~~~~~~~~

.. currentmodule:: fracspy.modelling.kirchhoff

.. autosummary::
   :toctree: generated/

    Kirchhoff

.. currentmodule:: fracspy.modelling.trueamp_kirchhoff

.. autosummary::
   :toctree: generated/

    TAKirchhoff


Imaging/Inversion
~~~~~~~~~~~~~~~~~

.. currentmodule:: fracspy.location.migration

.. autosummary::
   :toctree: generated/

    diffstack
    semblancediffstack

.. currentmodule:: fracspy.location.imaging

.. autosummary::
   :toctree: generated/

    lsi
    sparselsi

.. currentmodule:: fracspy.location.xcorri

.. autosummary::
   :toctree: generated/

    xcorri

.. currentmodule:: fracspy.location.location

.. autosummary::
   :toctree: generated/

    Location


Utilities
~~~~~~~~~

.. currentmodule:: fracspy.location.utils

.. autosummary::
   :toctree: generated/

    get_max_locs


Moment Tensor Inversion
-----------------------

Inverse engines
~~~~~~~~~~~~~~~

.. currentmodule:: fracspy.mtinversion.mtai

.. autosummary::
   :toctree: generated/

    mtamplitude_modelling
    mtamplitude_inversion


Utilities
~~~~~~~~~

.. currentmodule:: fracspy.mtinversion.greensfunction

.. autosummary::
   :toctree: generated/

    collect_source_angles
    pwave_zcomp_Greens