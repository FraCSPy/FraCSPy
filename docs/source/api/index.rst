.. _api:

FraCSPy API
===========

The Application Programming Interface (API) of FraCSPy mirrors the usual microseismic workflow
from pre-processing to source mechanism inversion. It is composed of the following modules:

* *XX*: xxx
* *XX*: xxx
* *XX*: xxx


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

.. currentmodule:: fracspy.locationsolvers.imaging

.. autosummary::
   :toctree: generated/

    migration
    lsqr_migration
    fista_migration

.. currentmodule:: fracspy.locationsolvers.crosscorr_imaging

.. autosummary::
   :toctree: generated/

    XcorrObjFunc
    xcorr_imaging

.. currentmodule:: fracspy.locationsolvers.localisationutils


Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated/

    get_max_locs


Moment Tensor Inversion
-----------------------

Source Location Known
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: fracspy.mtsolvers.mtai

.. autosummary::
   :toctree: generated/

    frwrd_mtmodelling
    lsqr_mtsolver

