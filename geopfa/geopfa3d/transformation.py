"""
Transition module

All functionalities from this module were moved to
:mod:`~geopfa.transformation`.
"""

import warnings

from geopfa.transformation import (
    VoterVetoTransformation as _UnifiedVoterVetoTransformation
)


class VoterVetoTransformation3D(_UnifiedVoterVetoTransformation):
    """Alias for geopfa.transformation.VoterVetoTransformation

    .. deprecated:: 0.1.0
       Use :class:`~geopfa.transformation.VoterVetoTransformation` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The geopfa3d.transformation.VoterVetoTransformation3D class is deprecated "
            "and will be removed in a future version. Please use "
            "geopfa.transformation.VoterVetoTransformation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
