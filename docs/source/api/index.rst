.. _api:

FraCSPy API
===========

The Application Programming Interface (API) of FraCSPy mirrors the usual microseismic workflow
from pre-processing to source mechanism inversion. It is composed of the following modules:

* Detection
* Localisation
* Moment Tensor Inversion

Detection
---------

.. currentmodule:: fracspy.detection.stacking

.. autosummary::
   :toctree: generated/

    maxdiffstack
    stalta
    detect_peaks
    estimate_origin_times
    diffstack_detect

Localisation
------------

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

.. currentmodule:: fracspy.modelling.mt_kirchhoff

.. autosummary::
   :toctree: generated/

    MTSKirchhoff
    MTMKirchhoff


Imaging/Inversion
~~~~~~~~~~~~~~~~~

.. currentmodule:: fracspy.location.migration

.. autosummary::
   :toctree: generated/

    kmigration
    diffstack    

.. currentmodule:: fracspy.location.imaging

.. autosummary::
   :toctree: generated/

    lsi
    sparselsi
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
    polarity_correction
    semblance_stack


Moment Tensor Inversion
-----------------------

Inverse engines
~~~~~~~~~~~~~~~

.. currentmodule:: fracspy.mtinversion.mtai

.. autosummary::
   :toctree: generated/

    MTA

.. currentmodule:: fracspy.mtinversion.mtwi

.. autosummary::
   :toctree: generated/

    MTW

.. currentmodule:: fracspy.mtinversion.mtinversion

.. autosummary::
   :toctree: generated/

    MTInversion


Utilities
~~~~~~~~~

.. currentmodule:: fracspy.mtinversion.greensfunction

.. autosummary::
   :toctree: generated/

    collect_source_angles
    pwave_greens_comp
    mt_pwave_greens_comp
    mt_pwave_greens_multicomp


.. currentmodule:: fracspy.mtinversion.utils

.. autosummary::
   :toctree: generated/

    get_mt_at_loc
    get_mt_max_locs
    get_magnitude
