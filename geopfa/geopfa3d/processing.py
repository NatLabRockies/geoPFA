"""Transition module

All functionalities from this module were moved to
:mod:`~geopfa.processing`.
"""

import warnings

import geopfa.processing


class Cleaners(geopfa.processing.Cleaners):
    """Alias for geopfa.processing.Cleaners

    .. deprecated:: 0.1.0
       :class:`~geopfa.processing.Cleaners` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Cleaners class"""
        warnings.warn(
            "The geopfa3d.processing.Cleaners class is deprecated"
            " and will be removed in a future version."
           " Please use the geopfa.processing module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class Exclusions(geopfa.processing.Exclusions):
    """Alias for geopfa.processing.Exclusions

    .. deprecated:: 0.1.0
       :class:`~geopfa.processing.Exclusions` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Exclusions class"""
        warnings.warn(
            "The geopfa3d.processing.Exclusions class is deprecated"
            " and will be removed in a future version."
           " Please use the geopfa.processing module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class Processing(geopfa.processing.Processing):
    """Alias for geopfa.processing.Processing

    .. deprecated:: 0.1.0
       :class:`~geopfa.processing.Processing` instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Processing class"""
        warnings.warn(
            "The geopfa3d.processing.Processing class is deprecated"
            " and will be removed in a future version."
           " Please use the geopfa.processing module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
