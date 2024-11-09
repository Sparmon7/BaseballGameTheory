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
    num_balls = 2
    num_strikes = 1
    num_outs = 1
    num_batters = 9
    max_runs = 5
    fouls_end_at_bats = False


class GameState:
    """Represents the changing state of a game."""

    __slots__ = ['inning', 'bottom', 'balls', 'strikes', 'num_runs', 'num_outs', 'first', 'second', 'third', 'batter']

    def __init__(self, inning=0, bottom=True, balls=0, strikes=0, runs=0, outs=0, first=-1, second=-1, third=-1, batter=0):
        self.inning = inning
        self.bottom = bottom
        self.balls = balls
        self.strikes = strikes
        self.num_runs = runs
        self.num_outs = outs
        self.first = first   
        self.second = second  
        self.third = third     
        self.batter = batter

    def checkValidity(self, rules=Rules):
        if (self.first!=self.second or self.first==-1) and (self.first!=self.third or self.first==-1) and (self.second!=self.third or self.second==-1) and \
        ((self.batter-self.first +8) % rules.num_batters < (self.num_outs + 1) or self.first==-1) and ((self.batter-self.second +8) % rules.num_batters < (self.num_outs + 1 + int(self.first!=-1)) or self.second==-1) and ( (self.batter - self.third  +8) % rules.num_batters < (self.num_outs + 1 + int(self.second!=-1) + int(self.first!=-1)) or self.third==-1):
            return True
        else:
            return False
    
    def checkTransValidity(self, result: PitchResult, firstBase, secondBase, thirdBase):
        #checking that inputted transition bases are valid, going case by case with all 8 combos of runners on base
        if firstBase < -1 or firstBase > 2 or secondBase < -1 or secondBase > 2 or thirdBase < -1 or thirdBase > 2 or thirdBase == 0:
            return False
        
        if self.third!=-1:
            if self.second!=-1:
                if self.first!=-1:
                    if firstBase > secondBase or secondBase > thirdBase:
                        return False
                    if firstBase< 2 and firstBase >=secondBase:
                        return False
                    if secondBase<2 and secondBase >= thirdBase:
                        return False
                    if firstBase==-1 or (firstBase==0 and result == PitchResult.HIT_DOUBLE):
                        return False
                    
                else:
                    if secondBase > thirdBase:
                        return False
                    if secondBase<2 and secondBase >= thirdBase:
                        return False
                    if secondBase==-1 or (secondBase==0 and result == PitchResult.HIT_DOUBLE):
                        return False 
                    
            else:
                if self.first!=-1:
                    if firstBase > thirdBase:
                        return False
                    if firstBase< 2 and firstBase >= thirdBase:
                        return False
                    if firstBase==-1 or (firstBase==0 and result == PitchResult.HIT_DOUBLE):
                        return False
                    
                else:
                    if thirdBase == -1:
                        return False
                
        else:
            if self.second!=-1:
                if self.first!=-1:
                    if firstBase > secondBase:
                        return False
                    if firstBase< 2 and firstBase >=secondBase:
                        return False
                    if firstBase==-1 or (firstBase==0 and result == PitchResult.HIT_DOUBLE):
                        return False
                    
                else:
                    if secondBase == -1 or (secondBase==0 and result == PitchResult.HIT_DOUBLE):
                        return False
                
            else:
                if self.first!=-1:
                    if firstBase == -1 or (firstBase==0 and result == PitchResult.HIT_DOUBLE):
                        return False
                    
            
        return True 
               
    def transition_from_pitch_result(self, result: PitchResult, rules: type[Rules] = Rules, firstBase: int = -1, secondBase: int = -1, thirdBase: int = -1) -> tuple[Self, int]:
        """Return the next state, transitioning according to result and the supplied rule config class."""
        """The three base variables indicate what base the player ends at [-1,0,1,2]=[None,second,third,home]"""

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
            if not self.checkTransValidity(result,firstBase,secondBase,thirdBase):
                return None, 0
            next_state.move_batter(1, rules, firstBase, secondBase, thirdBase)
        elif result == PitchResult.HIT_DOUBLE:
            if not self.checkTransValidity(result,firstBase,secondBase,thirdBase):
                    return None, 0
            next_state.move_batter(2, rules, firstBase, secondBase, thirdBase)
        elif result == PitchResult.HIT_TRIPLE:
            next_state.move_batter(3, rules)
        elif result == PitchResult.HIT_HOME_RUN:
            next_state.move_batter(4, rules)
        elif result == PitchResult.HIT_OUT:
            next_state.num_outs += 1
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % rules.num_batters

        if next_state.balls == rules.num_balls:  # Walk
            if next_state.first!=-1:
                if next_state.second!=-1:
                    if next_state.third!=-1:
                        next_state.num_runs += 1
                    next_state.third = next_state.second
                next_state.second = next_state.first
            next_state.first = next_state.batter
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % rules.num_batters
        if next_state.strikes == rules.num_strikes:
            next_state.num_outs += 1
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % rules.num_batters

        if next_state.num_runs > rules.max_runs:
            next_state.num_runs = rules.max_runs

        if next_state.num_outs >= rules.num_outs:
            next_state.inning += 1
            next_state.num_outs = 0
            next_state.first = next_state.second = next_state.third = -1

        return next_state, next_state.num_runs - self.num_runs

    def move_batter(self, num_bases: int, rules: type[Rules] = Rules, firstBase: int = -1, secondBase: int = -1, thirdBase: int = -1):
        """A helper method, advances runners and resets count"""
        if num_bases >= 4:
            self.num_runs += 1 + int(self.first != -1) + int(self.second != -1) + int(self.third != -1)
            self.first = self.second = self.third = -1
        elif num_bases == 3:
            self.num_runs += int(self.first!=-1) + int(self.second!=-1) + int(self.third!=-1)
            self.first = self.second = -1
            self.third = self.batter
        elif num_bases == 2:
            if self.third!=-1:
                if thirdBase ==2:
                    self.num_runs += 1
                    self.third = -1
            if self.second!=-1:
                if secondBase == 2:
                    self.num_runs += 1
                    self.second = -1
                elif secondBase == 1:
                    self.third = self.second
                    self.second=-1
            if self.first!=-1:
                if firstBase == 2:
                    self.num_runs += 1
                    self.first = -1
                elif firstBase == 1:
                    self.third = self.first
                    self.first=-1
            self.second = self.batter
                        
           
        elif num_bases == 1:
            if firstBase!=-1 or secondBase!=-1 or thirdBase!=-1:
                if self.third!=-1:
                    if thirdBase ==2:
                        self.num_runs += 1
                        self.third = -1
                if self.second!=-1:
                    if secondBase == 2:
                        self.num_runs += 1
                        self.second = -1
                    elif secondBase == 1:
                        self.third = self.second
                        self.second=-1
                if self.first!=-1:
                    if firstBase == 2:
                        self.num_runs += 1
                        self.first = -1
                    elif firstBase == 1:
                        self.third = self.first
                        self.first=-1
                    elif firstBase == 0:
                        self.second = self.first
                        self.first=-1
                self.first=self.batter
                        
            else:
                self.num_runs += int(self.third!=-1)
                self.third = self.second
                self.second = self.first
                self.first = self.batter


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
                f"{self.first}{self.second}{self.third})")

    def __hash__(self):
        return hash((self.inning, self.balls, self.strikes, self.num_outs, self.first, self.second, self.third, self.batter))

    def __eq__(self, other):
        return hash(self) == hash(other)