from enum import IntEnum

import torch


class PitchType(IntEnum):
    """
    Someone with more baseball knowledge can feel free to modify this enum, however
    that would also require modifying the dict in code_mappings.py accordingly, and
    retraining each model.

    This is in its own file to avoid circular imports.
    """

    FOUR_SEAM = 0
    TWO_SEAM = 1
    SLIDER = 2
    CHANGEUP = 3
    CURVE = 4
    CUTTER = 5

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch type."""

        one_hot = torch.zeros(len(PitchType))
        one_hot[self] = 1
        return one_hot
