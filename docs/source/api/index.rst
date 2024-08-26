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
    absdiffstack
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

.. currentmodule:: fracspy.utils.synthutils

.. autosummary::
   :toctree: generated/

    add_noise

.. currentmodule:: fracspy.location.utils

.. autosummary::
   :toctree: generated/

    get_max_locs
    dist2rec
    moveout_correction
    semblance_stack
    


Moment Tensor Inversion
-----------------------

Source Location Known
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: fracspy.mtsolvers.mtai

.. autosummary::
   :toctree: generated/

    frwrd_mtmodelling
    lsqr_mtsolver

