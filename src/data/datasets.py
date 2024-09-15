import random
from collections import defaultdict
from enum import IntEnum
from typing import Self, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.data_loading import BaseballData
from src.model.state import PitchResult
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Pitcher


class SwingResult(IntEnum):
    """
    The SwingResult enum represents the result of a batter's swing at a pitch, for use in training
    the SwingOutcomeDistribution model. It provides methods for converting to and from the full PitchResult.
    """

    STRIKE = 0
    FOUL = 1
    OUT = 2
    SINGLE = 3
    DOUBLE = 4
    TRIPLE = 5
    HOME_RUN = 6

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch type."""

        one_hot = torch.zeros(len(SwingResult))
        one_hot[self] = 1
        return one_hot

    def to_pitch_result(self) -> PitchResult:
        if self == self.STRIKE:
            return PitchResult.SWINGING_STRIKE
        elif self == self.FOUL:
            return PitchResult.SWINGING_FOUL
        elif self == self.OUT:
            return PitchResult.HIT_OUT
        elif self == self.SINGLE:
            return PitchResult.HIT_SINGLE
        elif self == self.DOUBLE:
            return PitchResult.HIT_DOUBLE
        elif self == self.TRIPLE:
            return PitchResult.HIT_TRIPLE
        elif self == self.HOME_RUN:
            return PitchResult.HIT_HOME_RUN

    @classmethod
    def from_pitch_result(cls, pitch_result: PitchResult):
        assert pitch_result.batter_swung()

        if pitch_result == PitchResult.SWINGING_STRIKE:
            return cls.STRIKE
        elif pitch_result == PitchResult.SWINGING_FOUL:
            return cls.FOUL
        elif pitch_result == PitchResult.HIT_OUT:
            return cls.OUT
        elif pitch_result == PitchResult.HIT_SINGLE:
            return cls.SINGLE
        elif pitch_result == PitchResult.HIT_DOUBLE:
            return cls.DOUBLE
        elif pitch_result == PitchResult.HIT_TRIPLE:
            return cls.TRIPLE
        elif pitch_result == PitchResult.HIT_HOME_RUN:
            return cls.HOME_RUN


class PitchDataset(Dataset):
    """
    A versatile pitch dataset, wrapping the BaseballData source. Options are available
    for filtering and mapping.
    """

    def __init__(self, data_source: BaseballData | None = None, valid_only: bool = True,
                 filter_on: Callable[[Pitch], bool] | None = None,
                 map_to: Callable[[int, Pitch], any] | None = None,
                 map_lazy: bool = True,
                 pitches: list[Pitch] | None = None, indices: list[int] | None = None):
        """
        :param data_source: The data source for the dataset
        :param valid_only: Whether to only include valid pitches
        :param filter_on: A function that filters the pitches
        :param map_to: A function that maps the pitches (with their index in the source) to a different type
        :param map_lazy: Whether to map the pitches lazily
        :param pitches: A list of pitches to use instead of the data source
        :param indices: A list of indices to use as an ordered subset of the data source
        """

        self.data_source = pitches if pitches is not None else data_source.pitches
        if indices is None:
            self.indices = [idx for idx, pitch in enumerate(self.data_source) if (not valid_only or pitch.is_valid()) and (filter_on is None or filter_on(pitch))]
        else:
            self.indices = [idx for idx in indices if (not valid_only or self.data_source[idx].is_valid()) and (filter_on is None or filter_on(self.data_source[idx]))]
        self.map_to = map_to

        if not map_lazy and map_to is not None:
            self.data = [map_to(idx, self.data_source[idx]) for idx in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> any:
        if hasattr(self, 'data'):
            return self.data[idx]
        else:
            return (self.map_to(self.indices[idx], self.data_source[self.indices[idx]]) if self.map_to is not None
                    else self.data_source[self.indices[idx]])

    @classmethod
    def get_split_on_attribute(cls, data_source: BaseballData | None, val_split: float = 0.2,
                               attribute: Callable[[Pitch], any] | None = None,
                               valid_only: bool = True,
                               filter_on: Callable[[Pitch], bool] | None = None,
                               map_to: Callable[[int, Pitch], any] | None = None,
                               map_lazy: bool = True,
                               seed: int | None = None) -> tuple[Self, Self]:
        """
        Splits the data into training and validation sets, with an option to split on a custom attribute.
        Note that the split is not guaranteed to be perfect across the attribute.
        """

        indices = list(range(len(data_source.pitches)))
        random.seed(seed)
        random.shuffle(indices)

        # Group by attribute
        if attribute is not None:
            attribute_map = defaultdict(list)
            for idx in indices:
                attribute_map[attribute(data_source.pitches[idx])].append(idx)
            indices = [idx for group in attribute_map.values() for idx in group]

        split_idx = int(len(indices) * (1 - val_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        return (cls(data_source, valid_only, filter_on, map_to, map_lazy, indices=train_indices),
                cls(data_source, valid_only, filter_on, map_to, map_lazy, indices=val_indices))


class PitchControlDataset(Dataset):
    """
    A dataset for training a model to predict the control of a pitch based on the pitcher and pitch type.
    Each sample contains a pitcher embedding, a one-hot encoding of the pitch type, and the variables
    of a fitted bivariate normal distribution.

    The empty_data flag is used for testing purposes

    This class is still messy but suffices for pitch control. If we work on improving that then this can be refactored.
    """

    def __init__(self, data_source: BaseballData | None = None, pitchers: list[Pitcher] | None = None, empty_data: bool = False):
        self.data = []
        for i, pitcher in enumerate(data_source.pitchers.values() if pitchers is None else pitchers):
            if not empty_data:
                for pitch_type, distribution in pitcher.estimated_control.items():
                    self.data.append((pitcher.obp_percentile, (pitcher.data, pitch_type.get_one_hot_encoding()),
                                      torch.tensor([distribution.mean[0], distribution.mean[1],
                                                    distribution.covariance_matrix[0, 0], distribution.covariance_matrix[1, 1],
                                                    distribution.covariance_matrix[0, 1]])))
            else:
                for pitch_type in PitchType:
                    self.data.append((i, pitch_type, (pitcher.data, pitch_type.get_one_hot_encoding()), torch.zeros(5)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[float, tuple[Tensor, Tensor], Tensor]:
        return self.data[idx]

    @classmethod
    def get_random_split(cls, data_source: BaseballData, val_split: float = 0.2, seed: int | None = None) -> tuple[Self, Self]:
        """Splits the data into training and validation sets, keeping the pitchers together."""

        pitchers = list(data_source.pitchers.values())
        random.seed(seed)
        random.shuffle(pitchers)

        split_idx = int(len(pitchers) * (1 - val_split))
        train_pitchers = pitchers[:split_idx]
        validation_pitchers = pitchers[split_idx:]

        return (cls(pitchers=train_pitchers),
                cls(pitchers=validation_pitchers))
