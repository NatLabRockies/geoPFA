"""
GeoPFA modeling software.
"""

import numpy as np
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geopfa")
except PackageNotFoundError:
    __version__ = "0+unknown"


np.seterr(all="ignore")
