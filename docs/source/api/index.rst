.. _api:

PyPrac API
==========

The Application Programming Interface (API) of PyPrac mirrors the usual microseismic workflow
from pre-processing to source mechanism inversion. It is composed of the following modules:

* *XX*: xxx
* *XX*: xxx
* *XX*: xxx


Location Determination
----------------------

Modelling
~~~~~~~~~

.. currentmodule:: pyfrac.modelling.kirchhoff

.. autosummary::
   :toctree: generated/

    Kirchhoff

Imaging/Inversion
~~~~~~~~~~~~~~~~~

.. currentmodule:: pyfrac.locationsolvers.imaging

.. autosummary::
   :toctree: generated/

    migration
    lsqr_migration
    fista_migration

.. currentmodule:: pyfrac.locationsolvers.crosscorr_imaging

.. autosummary::
   :toctree: generated/

    XcorrObjFunc
    xcorr_imaging

.. currentmodule:: pyfrac.locationsolvers.localisationutils


Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated/

    get_max_locs


