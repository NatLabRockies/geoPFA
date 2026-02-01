"""Transition module

All functionalities from this module were moved to
:mod:`~geopfa.data_readers`.
"""

import warnings

import geoPFA.data_readers


class GeospatialDataReaders(geoPFA.data_readers.GeospatialDataReaders):
    """Alias for geoPFA.data_readers.GeospatialDataReaders

    .. deprecated:: 0.1.0
       :class:`~geoPFA.data_readers.GeospatialDataReaders` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the GeospatialDataReaders class"""
        warnings.warn(
            "The geopfa3d.data_readers.GeospatialDataReaders class is deprecated and will be removed in a future version. "
            "Please use the geoPFA.data_readers module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
