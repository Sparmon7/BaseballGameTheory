from enum import IntEnum
from typing import Self


class PitchResult(IntEnum):
    """An enum representing the possible results of a pitch, used to transition the state."""

    CALLED_BALL = 0
    CALLED_STRIKE = 1

    SWINGING_STRIKE = 2
    SWINGING_FOUL = 3

    HIT_SINGLE = 4
    HIT_DOUBLE = 5
    HIT_TRIPLE = 6
    HIT_HOME_RUN = 7
    HIT_OUT = 8

    def batter_swung(self):
        """Returns whether the batter swung at the pitch (or if he was hit by the pitch)."""

        return (self == PitchResult.SWINGING_STRIKE or self == PitchResult.SWINGING_FOUL or self == PitchResult.HIT_OUT or
                self.batter_hit())

    def batter_hit(self):
        """Returns whether the batter hit the pitch (or was hit by the pitch)."""

        return (self == PitchResult.HIT_SINGLE or self == PitchResult.HIT_DOUBLE or self == PitchResult.HIT_TRIPLE or
                self == PitchResult.HIT_HOME_RUN)


class Rules:
    """
    This class decouples some of the game rules from the game state, to make it easier to swap
    between different rule sets. This is useful for debugging and testing things quickly.
    """

    num_innings = 9
    num_balls = 4
    num_strikes = 3
    num_outs = 3
    num_batters = 9
    max_runs = 9

    fouls_end_at_bats = False
    two_base_game = False


class DebugRules(Rules):
    """Here's a set of rules with much less game states"""

    num_innings = 3
    num_balls = 3
    num_strikes = 2
    num_outs = 2
    num_batters = 9

    fouls_end_at_bats = True  # This speeds up convergence significantly
    two_base_game = True  # This is a game with only two bases


class GameState:
    """Represents the changing state of a game."""

    __slots__ = ['inning', 'bottom', 'balls', 'strikes', 'num_runs', 'num_outs', 'first', 'second', 'third', 'batter']

    def __init__(self, inning=0, bottom=True, balls=0, strikes=0, runs=0, outs=0, first=False, second=False, third=False, batter=0):
        self.inning = inning
        self.bottom = bottom
        self.balls = balls
        self.strikes = strikes
        self.num_runs = runs
        self.num_outs = outs
        self.first = first      # True if runner on first
        self.second = second    # True if runner on second
        self.third = third      # True if runner on third
        self.batter = batter

    def transition_from_pitch_result(self, result: PitchResult, rules: type[Rules] = Rules) -> tuple[Self, int]:
        """Return the next state, transitioning according to result and the supplied rule config class."""

        next_state = GameState(inning=self.inning, balls=self.balls, strikes=self.strikes, runs=self.num_runs,
                               outs=self.num_outs, first=self.first, second=self.second, third=self.third, batter=self.batter)

        if next_state.inning >= rules.num_innings or next_state.num_runs >= rules.max_runs:
            return next_state, 0

        if (result == PitchResult.SWINGING_STRIKE or result == PitchResult.CALLED_STRIKE or
                (result == PitchResult.SWINGING_FOUL and (next_state.strikes < rules.num_strikes - 1 or rules.fouls_end_at_bats))):
            next_state.strikes += 1
        elif result == PitchResult.CALLED_BALL:
            next_state.balls += 1
        elif result == PitchResult.HIT_SINGLE:
            next_state.move_batter(1, rules)
        elif result == PitchResult.HIT_DOUBLE:
            next_state.move_batter(2, rules)
        elif result == PitchResult.HIT_TRIPLE:
            next_state.move_batter(3, rules)
        elif result == PitchResult.HIT_HOME_RUN:
            next_state.move_batter(4, rules)
        elif result == PitchResult.HIT_OUT:
            next_state.num_outs += 1
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % rules.num_batters

        if next_state.balls == rules.num_balls:  # Walk
            next_state.move_batter(1, rules)
        if next_state.strikes == rules.num_strikes:
            next_state.num_outs += 1
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % rules.num_batters

        if next_state.num_runs > rules.max_runs:
            next_state.num_runs = rules.max_runs

        if next_state.num_outs >= rules.num_outs:
            next_state.inning += 1
            next_state.num_outs = 0
            next_state.first = next_state.second = next_state.third = False

        return next_state, next_state.num_runs - self.num_runs

    def move_batter(self, num_bases: int, rules: type[Rules] = Rules):
        """A helper method, advances runners and resets count"""

        if num_bases >= 4:
            self.num_runs += 1 + int(self.first) + int(self.second) + int(self.third)
            self.first = self.second = self.third = False
        elif num_bases == 3:
            self.num_runs += int(self.first) + int(self.second) + int(self.third)
            self.first = self.second = False
            self.third = True
        elif num_bases == 2:
            self.num_runs += int(self.second) + int(self.third)
            self.third = self.first
            self.first = False
            self.second = True
        elif num_bases == 1:
            self.num_runs += int(self.third)
            self.third = self.second
            self.second = self.first
            self.first = True

        if rules.two_base_game:
            self.num_runs += int(self.third)
            self.third = False

        self.balls = self.strikes = 0
        self.batter = (self.batter + 1) % rules.num_batters

    def value(self) -> int:
        """
        Returns the "value" of the current state. Of course, value is more complicated than a single integer,
        this is a target for the value iteration algorithm
        """

        return self.num_runs

    def __repr__(self):
        return (f"GameState(i{self.inning} b{self.batter}: {self.balls}/{self.strikes}, {self.num_outs}, "
                f"{'x' if self.first else '-'}{'x' if self.second else '-'}{'x' if self.third else '-'})")

    def __hash__(self):
        return hash((self.inning, self.balls, self.strikes, self.num_outs, self.first, self.second, self.third, self.batter))

    def __eq__(self, other):
        return hash(self) == hash(other)
