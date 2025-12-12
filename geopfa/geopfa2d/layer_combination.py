"""Transition module

All functionalities from this module were moved to
:module:`~geopfa.data_readers`.
"""

import warnings

from geopfa.layer_combination import VoterVeto as _UnifiedVoterVeto


class VoterVeto(_UnifiedVoterVeto):
    """Alias for geopfa.layer_combination.VoterVeto

    .. deprecated:: 0.1.0
       Use :class:`~geopfa.layer_combination.VoterVeto` instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The geopfa2d.layer_combination.VoterVeto class is deprecated and will be removed "
            "in a future version. Please use geopfa.layer_combination.VoterVeto instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
