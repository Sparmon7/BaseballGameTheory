import enum
from collections import defaultdict

import torch
from torch.distributions import MultivariateNormal

from src.model.pitch_type import PitchType
from src.model.zones import Zones


min_obp_cutoff = 167  # Minimum number of at-bats to consider a player for OBP (chosen to represent 95% of pitches)
min_run_attempts = 10 # Minimum number of run attempts to consider a player for running

class Runner:
    """
    A Runner is represented by 3 probabilites:
    - The probability of scoring from second on a single (otherwise assumed to reach third)
    - The probability of reaching third from first on a single (otherwise assumed to reach second)
    - The probability of scoring from first on a double (otherwise assumed to reach third)
    
    Assumptions:
    - Does not account for the possibility of being thrown out
    - Runner at first will never score on a single
    - Runner will always advance at least as many bases as the hitter
    """
    
    __slots__ = ['second_to_home', 'first_to_third', 'first_to_home']
    
    def __init__(self, second_to_home: float | None = None, first_to_third: float | None = None, first_to_home: float| None = None):
        self.second_to_home = second_to_home
        self.first_to_third = first_to_third
        self.first_to_home = first_to_home
        

class Batter:
    """
    A Batter is represented as a 3D tensor of shape (2 * len(PitchType), Zones.DIMENSION, Zones.DIMENSION).
    Each pitch type/location combination has a corresponding relative swinging frequency and batting average.

    Attributes:
        data (torch.Tensor): The tensor representing the batter's relative swinging frequency and batting average.
        obp (float): The batter's on-base percentage.
        obp_percentile (float): The batter's percentile in on-base percentage.
        num_at_bats (int): The number of at-bats the batter has had.

    We could consider changing this representation to add more information about the batter, such as handedness, etc.
    That would require reworking the data loading code and the models.
    """

    __slots__ = ['data', 'obp', 'obp_percentile', 'num_at_bats']

    def __init__(self, obp: float | None = None, obp_percentile: float | None = None,
                 data: torch.Tensor = None):
        if data:
            assert (data.size(0) == 2 * len(PitchType) and
                    data.size(1) == Zones.DIMENSION and
                    data.size(2) == Zones.DIMENSION)

            self.data = data
        else:
            self.data = torch.zeros(2 * len(PitchType), Zones.DIMENSION, Zones.DIMENSION)
            # self.data[:len(PitchType), :, :] = 1 / len(PitchType)

        self.obp = obp
        self.obp_percentile = obp_percentile
        self.num_at_bats = 0

    def set_swinging_frequency_data(self, data: torch.Tensor):
        """
        Sets the swinging data for the batter. You can provide relative frequencies or provide total pitches for each type
        since the method normalizes the values.

        :param data: The swinging frequency data, a 3D tensor of shape (len(PitchType), Zones.DIMENSION, Zones.DIMENSION) to correspond to each pitch type.
        """

        # Normalize the swinging frequencies
        if data.sum() > 0:
            self.data[len(PitchType):, :, :] = data / data.sum()

    def set_batting_average_data(self, data: torch.Tensor):
        """Sets the batting average data for the batter."""

        self.data[:len(PitchType), :, :] = data


class Pitcher:
    """
    A Pitcher is represented as a 3D tensor of shape (2 * len(PitchType), Zones.DIMENSION, Zones.DIMENSION).
    Each pitch type/location combination has a corresponding relative throwing frequency and average velocity.
    Note that for now this tensor is structurally identical to a batter's.

    Attributes:
        data (torch.Tensor): The tensor representing the pitcher's relative throwing frequency and average velocity.
        obp (float): The pitcher's on-base percentage (against).
        obp_percentile (float): The pitcher's percentile in on-base percentage (where lower obp is higher percentile).
        estimated_control (defaultdict[PitchType, MultivariateNormal | None]): A bivariate normal distribution
            representing the pitcher's control. Note that BaseballData learns this from empirical data, and it is used
            to train the PitcherControl network.
        num_batters_faced (int): The number of batters faced by the pitcher.
    """

    __slots__ = ['data', 'obp', 'obp_percentile', 'estimated_control', 'num_batters_faced']

    def __init__(self, obp: float | None = None, obp_percentile: float | None = None,
                 data: torch.Tensor = None):
        if data:
            assert (data.size(0) == 2 * len(PitchType) and
                    data.size(1) == Zones.DIMENSION and
                    data.size(2) == Zones.DIMENSION)

            # Normalize the throwing frequencies
            data[:len(PitchType), :, :] /= data[:len(PitchType), :, :].sum(dim=0, keepdim=True)
            self.data = data
        else:
            self.data = torch.zeros(2 * len(PitchType), Zones.DIMENSION, Zones.DIMENSION)
            # self.data[:len(PitchType), :, :] = 1 / len(PitchType)

        self.obp = obp
        self.obp_percentile = obp_percentile
        self.estimated_control: dict[PitchType, MultivariateNormal | None] = {}
        self.num_batters_faced = 0

    def set_throwing_frequency_data(self, data: torch.Tensor):
        """
        Sets the throwing data for the pitcher. You can provide relative frequencies or provide total pitches for each type
        since the method normalizes the values.

        :param data: The throwing frequency data, a 3D tensor of shape (len(PitchType), Zones.DIMENSION, Zones.DIMENSION) to correspond to each pitch type. The tensor is normalized to sum to 1 along the pitch type dimension.
        """

        # Normalize the throwing frequencies
        if data.sum() > 0:
            self.data[:len(PitchType), :, :] = data / data.sum()

    def set_average_velocity_data(self, data: torch.Tensor):
        """Sets the average velocity data for the pitcher."""

        self.data[len(PitchType):, :, :] = data


class MLBTeam(enum.IntEnum):
    DIAMONDBACKS = 0
    BRAVES = 1
    ORIOLES = 2
    RED_SOX = 3
    WHITE_SOX = 4
    CUBS = 5
    REDS = 6
    INDIANS = 7
    ROCKIES = 8
    TIGERS = 9
    ASTROS = 10
    ROYALS = 11
    ANGELS = 12
    DODGERS = 13
    MARLINS = 14
    BREWERS = 15
    TWINS = 16
    METS = 17
    YANKEES = 18
    ATHLETICS = 19
    PHILLIES = 20
    PIRATES = 21
    PADRES = 22
    GIANTS = 23
    MARINERS = 24
    CARDINALS = 25
    RAYS = 26
    RANGERS = 27
    BLUE_JAYS = 28
    NATIONALS = 29
