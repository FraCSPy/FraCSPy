from . import location
from . import modelling
from . import mtinversion
from . import processing
from . import utils
from . import visualisation


try:
    from .version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. Fracspy should be installed
    # properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")