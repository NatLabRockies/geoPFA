"""
Transition module

All functionalities from this module were moved to
:module:`~geopfa.plotters`.
"""

import warnings

from geopfa.plotters import (
    GeospatialDataPlotters as _UnifiedGeospatialDataPlotters
)


class GeospatialDataPlotters(_UnifiedGeospatialDataPlotters):
    """Alias for geopfa.plotters.GeospatialDataPlotters

    .. deprecated:: 0.1.0
       Use :class:`~geopfa.plotters.GeospatialDataPlotters` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The geopfa3d.plotters.GeospatialDataPlotters class is deprecated "
            "and will be removed in a future version. Please use "
            "geopfa.plotters.GeospatialDataPlotters instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)