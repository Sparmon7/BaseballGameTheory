from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.getenv('FOLDER'))

import copy
import random
import warnings
from collections import defaultdict
from typing import Self

import cvxpy as cp
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader



try:
    # noinspection PyUnresolvedReferences
    ipython_name = get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
import sys
sys.path.append('C:/Users/sparm/BaseballGameTheory')
from src.policy.rosters import rosters
from src.data.data_loading import BaseballData, load_blosc2, save_blosc2
from src.data.datasets import SwingResult, PitchDataset, PitchControlDataset
from src.distributions.batter_patience import BatterSwings, batter_patience_map
from src.distributions.pitcher_control import PitcherControl
from src.distributions.swing_outcome import SwingOutcome, map_swing_outcome
from src.model.state import GameState, PitchResult, Rules
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.zones import default


# To simplify typing, we define some types in accordance with the reference paper's notation
type A = tuple[PitchType, int]  # (PitchType, ZONE_i)
type O = bool
type S = GameState

# These distributions can be indexed into according to the lists defined above
type SwingOutcomeDistribution = np.ndarray  # [S_i][PitchType][ZONES_i][SwingResult] -> swing result probability
type PitcherControlDistribution = np.ndarray  # [A_i][COMBINED_ZONES_i] -> pitch outcome zone probability
type BatterPatienceDistribution = np.ndarray  # [S_i][PitchType][ZONES_i] -> batter swing probability


class PolicySolver:
    """
    Given a pitcher and batter, this class aims to calculate the optimal policy for the pitcher.
    The solution to the game is dependent on the setup of GameState and its transition rules.
    Depending on your use case, you will want to modify the hash function in GameState.

    Note that we're not currently using the num_runs attribute in the GameState, but it can be used to limit the number
    of runs, and the models are also trained with runs as a parameter. However, this increases the number of states.
    """

    # We define some type aliases to improve readability, note how they are indexed into
    type Policy = np.ndarray  # [S_i][A_i] -> float

    # We can define the transition distribution with reasonably sized ndarrays, since the number of states
    # that can be reached from a single state is limited
    type Transitions = np.ndarray  # [S_i][0-7] -> (next state index, reward)
    type TransitionDistribution = np.ndarray  # [S_i][A_i][O_i][0-7] -> probability
    max_transitions = 12

    default_batch: int = 512

    def __init__(self, bd: BaseballData, pitcher_id: int, batter_lineup: list[int], rules: type[Rules] = Rules):
        """
        Initializes the policy solver with the given pitcher and batter, and some optional parameters.
        This does not do any actual calculations, but sets up the necessary data structures and config.
        """

        # BatterPatienceDistribution indexing relies on COMBINED_ZONES = ZONES + BORDERLINE_ZONES in that order
        assert default.COMBINED_ZONES[len(default.ZONES)] == default.BORDERLINE_ZONES[0]

        self.bd = bd
        self.pitcher_id = pitcher_id
        self.batter_lineup = batter_lineup

        self.pitcher_actions: list[A] = [(pitch_type, zone_i) for zone_i in range(len(default.ZONES)) for pitch_type in PitchType]
        self.batter_actions: list[O] = [False, True]  # Order is important!

        third_options = [False, True] if not rules.two_base_game else [False]
        base_options = range(rules.num_batters) if rules.runner_stochastic else [False, True]
        if not rules.two_base_game and rules.runner_stochastic:
            third_options = base_options
        

        self.game_states: list[S] = [
            GameState(inning=inning, balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third, batter=batter)
            for inning in range(rules.num_innings) for balls in range(rules.num_balls) for strikes in range(rules.num_strikes) for outs in range(rules.num_outs)
            for first in base_options for second in base_options for third in third_options for batter in range(rules.num_batters)
        ]

        # Terminal states are stored separately for easier indexing
        self.final_states: list[S] = [
            GameState(inning=rules.num_innings, balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third, batter=batter)
            for balls in range(rules.num_balls) for strikes in range(rules.num_strikes) for outs in range(rules.num_outs)
            for first in base_options for second in base_options for third in third_options for batter in range(rules.num_batters)
        ]
    

        self.total_states = self.game_states + self.final_states
        self.total_states_dict = {state: i for i, state in enumerate(self.total_states)}

        self.rules = rules

        self.transitions = None
        self.transition_distribution = None

        self.policy_problem = None
        self.num_programs = 0
        self.raw_values: np.ndarray | None = None
        self.raw_policy: PolicySolver.Policy | None = None

        self.batter_lineup_permutation = range(rules.num_batters)
        self.permuted_transitions = None

    @classmethod
    def from_saved(cls, path: str, bd: BaseballData | None = None) -> Self:
        """Loads a saved policy solver from a blosc2 file"""

        solver: Self = load_blosc2(path)
        solver.bd = bd
        return solver

    def save(self, path: str):
        """Saves the policy solver to a blosc2 file"""

        save_blosc2(self, path)

    def __getstate__(self):
        """We shouldn't pickle bd (redundant) or policy_problem (not serializable)"""

        return {k: v for k, v in self.__dict__.items() if k not in ['bd', 'policy_problem']}

    def initialize_distributions(self, batch_size: int = default_batch, save_distributions: bool = False,
                                 load_distributions: bool = False, load_transition: bool = False, path: str = ""):
        """
        Initializes the transition distributions for the given pitcher and batter pair. Note that
        the individual calculate distribution methods support batching for multiple pairs, but this 
        is not currently exposed.
        """

        if load_distributions and load_transition:
            distributions = load_blosc2(f'{path}transition_distribution.blosc2')
            self.transitions, self.transition_distribution = distributions['transitions'], distributions['transition_distribution']
            return

        distributions = defaultdict(lambda: None)
        if load_distributions:
            distributions = load_blosc2(f'{path}distributions.blosc2')

        self.transitions, self.transition_distribution = self.precalculate_transition_distribution(
            batch_size=batch_size, batter_patiences=distributions['batter_patience'],
            swing_outcomes=distributions['swing_outcomes'], pitcher_control=distributions['pitcher_control'],
            save_distributions=save_distributions, path=path
        )

        if save_distributions:
            save_blosc2({'transition_distribution': self.transition_distribution, 'transitions': self.transitions}, f'{path}transition_distribution.blosc2')

    def precalculate_transition_distribution(self, batch_size: int = default_batch,
                                             batter_patiences: BatterPatienceDistribution | None = None,
                                             swing_outcomes: SwingOutcomeDistribution | None = None,
                                             pitcher_control: PitcherControlDistribution | None = None,
                                             save_distributions: bool = False, path: str = "") -> tuple[Transitions, TransitionDistribution]:
        """
        Precalculates the transition probabilities for a given pitcher and batter pair.
        This method is complicated by the fact that both the pitch outcome and the swing outcome
        are stochastic.
        """

        swing_outcomes = self.calculate_swing_outcome_distribution([(self.pitcher_id, batter_id) for batter_id in self.batter_lineup], batch_size=batch_size) \
            if swing_outcomes is None else swing_outcomes
        pitcher_control = self.calculate_pitcher_control_distribution([self.pitcher_id], batch_size=batch_size)[self.pitcher_id] \
            if pitcher_control is None else pitcher_control
        batter_patiences = self.calculate_batter_patience_distribution(self.batter_lineup, batch_size=batch_size) \
            if batter_patiences is None else batter_patiences

        if save_distributions:
            save_blosc2({'swing_outcomes': swing_outcomes, 'pitcher_control': pitcher_control, 'batter_patience': batter_patiences}, f'{path}distributions.blosc2')

        def map_t(t):
            return self.total_states_dict[t[0]], t[1]

        # Remove duplicates maintaining order, and pad
        def pad(s):
            s = list(dict.fromkeys(s).keys())
            return s + [(-1, 0)] * (self.max_transitions - len(s))



        '''        

        # Since a state can only transition to a limited number of states, we're able to store the transition indexes in
        # a fixed size array. Empty transitions are padded with (-1, 0)
        transitions = np.asarray([pad([map_t(state.transition_from_pitch_result(result, rules=self.rules))
                                       for result in PitchResult]) for state in self.game_states], dtype=np.int32)
        probabilities = np.zeros((len(self.game_states), len(self.pitcher_actions), len(self.batter_actions), self.max_transitions), dtype=np.float32)

        # Used to transform swing outcome probabilities to transition probabilities. A matrix like this is necessary since
        # multiple swing outcomes can lead to the same state
        swing_to_transition_matrix = np.asarray([
            np.asarray([transitions[state_i, :, 0] == self.total_states_dict[self.total_states[state_i].transition_from_pitch_result(result.to_pitch_result(), rules=self.rules)[0]]
                        for result in SwingResult]).transpose()
            for state_i in range(len(self.game_states))
        ])
    '''
        borderline_mask = np.asarray([zone.is_borderline for zone in default.COMBINED_ZONES])
        strike_mask = np.asarray([zone.is_strike for zone in default.COMBINED_ZONES])

        # It's important for indexing that these are at the start
        called_ball_i = 0
        called_strike_i = 1
        assert PitchResult.CALLED_BALL == called_ball_i
        assert PitchResult.CALLED_STRIKE == called_strike_i

        # Iterate over each state
        # At the cost of readability, we use numpy operations to speed up the calculations (if necessary, even the remaining for loops can be removed)
        for state_i, state in tqdm(enumerate(self.game_states), desc='Calculating transition distribution', total=len(self.game_states)):
            for action_i, action in enumerate(self.pitcher_actions):
                pitch_type, intended_zone_i = action
                for batter_swung in range(len(self.batter_actions)):
                    # Given an intended pitch, we get the actual outcome distribution
                    outcome_zone_probs = pitcher_control[action_i]

                    swing_probs = np.zeros(len(default.COMBINED_ZONES)) + batter_swung

                    # On obvious balls, the batter will not swing
                    swing_probs[~strike_mask] = 0

                    # On borderline balls, if the batter has chosen to swing, we override the decision with the batter's patience
                    if batter_swung:
                        swing_probs[borderline_mask] = batter_patiences[self.batter_lineup[state.batter]][state_i, pitch_type]
                    take_probs = 1 - swing_probs
        '''
                    # Handle swing outcomes (stochastic)
                    result_probs = swing_outcomes[(self.pitcher_id, self.batter_lineup[state.batter])][state_i, pitch_type]
                    transition_probs = np.dot(swing_to_transition_matrix[state_i], result_probs.transpose())
                    zone_swing_probs = swing_probs * outcome_zone_probs
                    probabilities[state_i, action_i, batter_swung] += np.dot(transition_probs, zone_swing_probs[0:len(default.ZONES)] + zone_swing_probs[len(default.ZONES):])

                    # Handle take outcome (deterministic)
                    probabilities[state_i, action_i, batter_swung, called_strike_i] += np.dot(take_probs, outcome_zone_probs * strike_mask)
                    probabilities[state_i, action_i, batter_swung, called_ball_i] += np.dot(take_probs, outcome_zone_probs * ~strike_mask)


        '''

        return transitions, probabilities

    def calculate_swing_outcome_distribution(self, matchups: list[tuple[int, int]], batch_size=default_batch) -> dict[tuple[int, int], SwingOutcomeDistribution]:
        """
        Calculates the distribution of swing outcomes for a list of pitcher and batter pairs, given the current game state.
        This method takes in a list to allow for batch processing.

        :return: A dictionary mapping a state, pitcher action and batter action to a distribution of swing outcomes
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        swing_outcome_model = SwingOutcome().to(device)
        swing_outcome_model.load_state_dict(torch.load('../../model_weights/swing_outcome.pth', map_location=device, weights_only=True))
        swing_outcome_model.eval()

        swing_outcome = {}

        # We only care about calculating the results for states that are unique to the model (which does not consider every variable)
        interested_states = [self.total_states_dict[GameState(balls=balls, strikes=strikes, outs=outs)]
                             for balls in range(self.rules.num_balls) for strikes in range(self.rules.num_strikes) for outs in range(self.rules.num_outs)]
        interested_pitches = [(state_i, pitch_i) for state_i in interested_states for pitch_i in range(len(self.pitcher_actions))]

        for pitcher_id, batter_id in tqdm(matchups, desc='Calculating swing outcomes'):
            swing_outcome[(pitcher_id, batter_id)] = np.zeros((len(self.game_states), len(PitchType), len(default.ZONES), len(SwingResult)), dtype=np.float32)

            pitch_data = [Pitch(self.game_states[state_i], batter_id=batter_id, pitcher_id=pitcher_id,
                                location=self.pitcher_actions[pitch_i][1], pitch_type=self.pitcher_actions[pitch_i][0], pitch_result=PitchResult.HIT_SINGLE)
                          for state_i, pitch_i in interested_pitches]
            pitch_dataset = PitchDataset(pitches=pitch_data, map_to=lambda idx, p: map_swing_outcome(idx, p, self.bd), valid_only=False)
            pitch_dataloader = DataLoader(pitch_dataset, batch_size=batch_size, shuffle=False)

            for batch, (pitch_idx, data, target) in enumerate(pitch_dataloader):
                data = [d.to(device) for d in data]
                outcome_tensor = swing_outcome_model(*data, softmax=True)

                result_distributions = outcome_tensor.squeeze().tolist()
                if not isinstance(result_distributions[0], list):  # In case the batch is a single element
                    result_distributions = [result_distributions]

                for i, result_distribution in enumerate(result_distributions):
                    state_i, pitch_i = interested_pitches[batch * batch_size + i]
                    pitch_type, zone_i = self.pitcher_actions[pitch_i]
                    swing_outcome[(pitcher_id, batter_id)][state_i, pitch_type, zone_i] = result_distribution

            for state_i in range(len(self.game_states)):
                if state_i not in interested_states:
                    for pitch_i in range(len(self.pitcher_actions)):
                        state = self.game_states[state_i]
                        interested_state = self.total_states_dict[GameState(balls=state.balls, strikes=state.strikes, outs=state.num_outs)]
                        pitch_type, zone_i = self.pitcher_actions[pitch_i]
                        swing_outcome[(pitcher_id, batter_id)][state_i, pitch_type, zone_i] = swing_outcome[(pitcher_id, batter_id)][interested_state, pitch_type, zone_i]

        return swing_outcome

    def calculate_pitcher_control_distribution(self, pitchers: list[int], batch_size=default_batch) -> dict[int, PitcherControlDistribution]:
        """
        Calculates the distribution of actual pitch outcomes for a given pitcher, given the intended pitch type and zone

        :return: A dictionary mapping a pitcher action to a distribution of actual pitch outcomes over zones
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pitcher_control_model = PitcherControl().to(device)
        pitcher_control_model.load_state_dict(torch.load('../../model_weights/pitcher_control.pth', map_location=device, weights_only=True))
        pitcher_control_model.eval()

        pitcher_type_control = defaultdict(defaultdict)

        # This dataset class automatically iterates over pitch types, empty_data is a custom flag for this purpose
        pitch_control_dataset = PitchControlDataset(data_source=None,
                                                    pitchers=[self.bd.pitchers[pitcher_i] for pitcher_i in pitchers],
                                                    empty_data=True)
        dataloader = DataLoader(pitch_control_dataset, batch_size=batch_size, shuffle=False)

        num_batches = len(dataloader)
        for i, (p_i, p_type, p_data, distribution) in tqdm(enumerate(dataloader), desc='Calculating pitcher control',
                                                           leave=False, total=num_batches):
            p_data = [d.to(device) for d in p_data]
            control_tensor = pitcher_control_model(*p_data)

            control_list = control_tensor.squeeze().tolist()
            if not isinstance(control_list[0], list):
                control_list = [control_list]

            for k, control in enumerate(control_list):
                pitcher = pitchers[p_i[k]]
                pitch_type = PitchType(p_type[k].item())
                gaussian = MultivariateNormal(torch.tensor(control[:2]), torch.tensor([[control[2], control[4]],
                                                                                       [control[4], control[3]]]))
                pitcher_type_control[pitcher][pitch_type] = gaussian

        # To make things simple, we use random sampling to find a distribution of pitch outcomes
        # If there's an equation for doing this directly, we should use it
        pitcher_control = {}
        for pitcher in tqdm(pitchers, desc='Sampling pitcher control'):
            pitcher_control[pitcher] = np.zeros((len(self.pitcher_actions), len(default.COMBINED_ZONES)), dtype=np.float32)

            for pitch_i, pitch in enumerate(self.pitcher_actions):
                pitch_type, intended_zone_i = pitch
                zone_center = default.ZONES[intended_zone_i].center()
                gaussian = pitcher_type_control[pitcher][pitch_type]
                gaussian = MultivariateNormal(torch.tensor([zone_center[0], zone_center[1]]), gaussian.covariance_matrix)

                num_samples = 10000
                sample_pitches = gaussian.sample(torch.Size((num_samples,)))
                zones = default.get_zones_batched(sample_pitches[:, 0], sample_pitches[:, 1])
                for zone_i in zones:
                    pitcher_control[pitcher][pitch_i, zone_i] += 1 / num_samples

        return pitcher_control

    def calculate_batter_patience_distribution(self, batters: list[int], batch_size=default_batch) -> dict[int, BatterPatienceDistribution]:
        """
        Calculates the distribution of batter patience for a given batter, given the current game state and pitcher action

        :return: A dictionary mapping a state and pitcher action (on a borderline zone) to the probability that the batter will swing
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batter_patience_model = BatterSwings().to(device)
        batter_patience_model.load_state_dict(torch.load('../../model_weights/batter_patience.pth', map_location=device, weights_only=True))
        batter_patience_model.eval()

        # We only care about calculating the results for states that are unique to the model (which does not consider every variable)
        interested_states = [self.total_states_dict[GameState(balls=balls, strikes=strikes, outs=outs)]
                             for balls in range(self.rules.num_balls) for strikes in range(self.rules.num_strikes) for outs in range(self.rules.num_outs)]
        interested_pitches = [(state_i, type_i, borderline_zone_i) for state_i in interested_states
                              for type_i in range(len(PitchType)) for borderline_zone_i in range(len(default.BORDERLINE_ZONES))]

        batter_patience = {}
        for batter_i in tqdm(batters, desc='Calculating batter patience'):
            batter_patience[batter_i] = np.zeros((len(self.game_states), len(PitchType), len(default.ZONES)), dtype=np.float32)

            pitch_data = [Pitch(self.game_states[state_i], pitcher_id=-1, batter_id=batter_i,
                                location=zone_i, pitch_type=PitchType(type_i), pitch_result=PitchResult.HIT_SINGLE)
                          for state_i, type_i, zone_i in interested_pitches]

            patience_dataset = PitchDataset(pitches=pitch_data, map_to=lambda idx, p: batter_patience_map(self.bd, idx, p), valid_only=False)
            dataloader = DataLoader(patience_dataset, batch_size=batch_size, shuffle=False)

            for batch, (pitch_idx, data, swing) in enumerate(dataloader):
                data = [d.to(device) for d in data]

                swing_percent = batter_patience_model(*data).squeeze().tolist()
                if not isinstance(swing_percent, list):
                    swing_percent = [swing_percent]

                for i, swing_percent in enumerate(swing_percent):
                    state_i, pitch_type_i, zone_i = interested_pitches[batch * batch_size + i]
                    batter_patience[batter_i][state_i, pitch_type_i, zone_i] = swing_percent

            for state_i in range(len(self.game_states)):
                if state_i not in interested_states:
                    for pitch_type_i in range(len(PitchType)):
                        for zone_i in range(len(default.BORDERLINE_ZONES)):
                            interested_state = self.total_states_dict[GameState(balls=self.game_states[state_i].balls, strikes=self.game_states[state_i].strikes,
                                                                                outs=self.game_states[state_i].num_outs)]
                            batter_patience[batter_i][state_i, pitch_type_i, zone_i] = batter_patience[batter_i][interested_state, pitch_type_i, zone_i]

        return batter_patience

    @classmethod
    def tuple_help(cls):
        """Utility for default dict"""
    
        return 0, 0

    def set_batter_permutation(self, batter_lineup_permutation: np.ndarray | None):
        """
        Without modifying the distributions, this method allows you to change the effective batter lineup
        for the policy solver. The input should be a permutation of range(GameState.num_batters).
        """

        if batter_lineup_permutation is None:
            self.permuted_transitions = None
            return

        self.batter_lineup_permutation = list(batter_lineup_permutation)
        self.permuted_transitions = self.transitions.copy()
        for state_i, state in enumerate(self.game_states):
            transitions = self.permuted_transitions[state_i, :, 0]
            for idx, transition_i in enumerate(transitions):
                if transition_i > 0:
                    transition = self.total_states[transition_i]

                    # If the transition goes to a new batter, we override it with the next batter in the permutation
                    if transition.batter != state.batter:
                        transition = copy.copy(transition)
                        current_batter_i = batter_lineup_permutation.index(state.batter)
                        transition.batter = batter_lineup_permutation[(current_batter_i + 1) % self.rules.num_batters]
                        transitions[idx] = self.total_states_dict[transition]

    def calculate_optimal_policy(self, print_output: bool = False, beta: float = 1e-3,
                                 use_last_values: bool = False) -> tuple[Policy, list[float]]:
        """
        Uses value iteration to calculate the optimal policy for our model, given the pitcher and batter.

        A policy (or mixed strategy) defines a probability distribution over actions for each state for
        the pitcher.

        :param print_output: This flag prints the difference between iterations and keeps a progress bar
        :param beta: The minimum change in value to continue iterating
        :param use_last_values: If True, the last calculated values are used as the initial values for this run, which can speed up convergence by 50%
        :return: The optimal pitcher policy, assigning a probability to each action in each state and the value of each state.
            These structures are indexed according to the lists defined in the class, but methods are provided to view the data more easily.
            You can ignore the return value and use get_value or get_policy instead.
        """

        # Stores the "value" of each state, indexed according to total_states
        value = self.raw_values if self.raw_values is not None and use_last_values else (
            np.concatenate((np.random.rand(len(self.game_states)), np.zeros(len(self.final_states)))))

        # Stores the policy for each state, indexed according to game_states
        policy: PolicySolver.Policy = np.ones((len(self.game_states), len(self.pitcher_actions))) / len(self.pitcher_actions)

        if self.transition_distribution is None:
            self.initialize_distributions()

        # Instead of optimizing all states at once, we can do one out at a time
        # This is because the values of a later out are completely independent of the values of an earlier out
        # You can view this as mixing backward induction with value iteration
        num_inning_outs = self.rules.num_innings * self.rules.num_outs
        for inning_out in tqdm(reversed(range(num_inning_outs)), total=num_inning_outs, desc='Optimizing innings/outs', disable=not print_output):
            states = [state_i for state_i, state in enumerate(self.game_states)
                      if state.inning == inning_out // self.rules.num_outs and state.num_outs == inning_out % self.rules.num_outs]

            # Sort the states sequentially to speed up convergence, keeping at-bats together
            states.sort(key=lambda st: (self.batter_lineup_permutation.index(self.game_states[st].batter),
                                        int(self.game_states[st].third), int(self.game_states[st].second), int(self.game_states[st].first),
                                        self.game_states[st].balls, self.game_states[st].strikes), reverse=True)

            # Now run value iteration on these states, repeatedly calculating the policy and values until convergence
            difference = float('inf')
            iter_num = 0
            while difference > beta:
                new_policy = np.zeros((len(self.game_states), len(self.pitcher_actions)))
                new_value = value.copy()

                transitions_src = self.transitions if self.permuted_transitions is None else self.permuted_transitions

                # Optimize the policy for each state
                for state_i in states:
                    # The expected value (transition reward + value of next states) for each action pair
                    transitions, rewards = transitions_src[state_i].transpose()
                    action_quality = np.dot(self.transition_distribution[state_i], rewards + new_value[transitions])
                    new_policy[state_i], new_value[state_i] = self.update_policy(action_quality)

                    # We can improve convergence by running additional iterations on states with two strikes,
                    # since they are self-loops
                    # We tested out some schedules, but a fixed number of iterations appeared to work best
                    loop_schedule = [2]
                    if self.game_states[state_i].strikes == self.rules.num_strikes - 1 and not self.rules.fouls_end_at_bats:
                        for _ in range(loop_schedule[min(iter_num, len(loop_schedule) - 1)]):
                            action_quality = np.dot(self.transition_distribution[state_i], rewards + new_value[transitions])
                            new_policy[state_i], new_value[state_i] = self.update_policy(action_quality)

                # Update values
                difference = np.abs(new_value - value).max()
                policy = new_policy
                value = new_value

                iter_num += 1

        self.raw_values = value
        self.raw_policy = policy

        return policy, value

    def initialize_policy_problem(self, max_pitch_percentage: float = 0.7):
        """We only need to initialize the policy problem once, as the constraints always the same"""

        if len(self.pitcher_actions) == 1:
            max_pitch_percentage = 1.0

        policy = cp.Variable(len(self.pitcher_actions))
        policy_constraints = [policy >= 0, cp.sum(policy) == 1, policy <= max_pitch_percentage]  # Limit the maximum probability of any action

        action_quality = [cp.Parameter(len(self.pitcher_actions)) for _ in self.batter_actions]

        objective = cp.Minimize(cp.maximum(*[cp.sum([policy[a_i] * action_quality[o][a_i]
                                                     for a_i in range(len(self.pitcher_actions))]) for o in self.batter_actions]))

        problem = cp.Problem(objective, policy_constraints)
        self.policy_problem = policy, action_quality, problem

    def update_policy(self, action_quality: np.ndarray, print_warnings: bool = False) -> tuple[np.asarray, float]:
        """Optimizes a new policy using dynamic programming"""

        if self.policy_problem is None:
            self.initialize_policy_problem()

        self.num_programs += 1

        policy, action_quality_param, problem = self.policy_problem
        for o in range(len(self.batter_actions)):
            action_quality_param[o].value = action_quality[:, o]
        problem.solve()

        if problem.status != cp.OPTIMAL and problem.status != cp.OPTIMAL_INACCURATE:
            raise ValueError(f'Policy optimization failed, status {problem.status}')
        else:
            if problem.status == cp.OPTIMAL_INACCURATE and print_warnings:
                warnings.warn('Inaccurate optimization detected')
            new_policy = np.asarray([policy[a_i].value for a_i in range(len(self.pitcher_actions))])
            return new_policy, problem.value

    def get_value(self, state: GameState = GameState()) -> float:
        """After policy has been calculated, returns the value of a state"""

        return self.raw_values[self.total_states_dict[state]]

    def get_policy(self) -> dict[GameState, list[tuple[A, float]]]:
        """After policy has been calculated, returns the optimal policy as a dictionary, for easier access"""

        optimal_policy = {}
        for i, state in enumerate(self.game_states):
            optimal_policy[state] = []
            for j, prob in enumerate(self.raw_policy[i]):
                if prob > 0.001:
                    optimal_policy[state].append((self.pitcher_actions[j], prob))
        return optimal_policy


def seed(i: int = 0):
    """
    There's a bit of randomness in the distribution calculations, most likely from the way
    we're sampling with the pitcher control model. You can use this function to seed the randomness.
    """

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)


def test_era(bd: BaseballData, pitcher_id: int, batter_lineup: list[int], load=False, batter_permutation=None):
    print(f'Pitcher OBP: {bd.pitchers[pitcher_id].obp}, Batter (first) OBP: {bd.batters[batter_lineup[0]].obp}')

    solver = PolicySolver(bd, pitcher_id, batter_lineup, rules=Rules)
    solver.initialize_distributions(save_distributions=True, load_distributions=load, load_transition=True)
    solver.set_batter_permutation(batter_permutation)
    solver.calculate_optimal_policy(print_output=True, beta=2e-4)

    first_batter = 0 if batter_permutation is None else batter_permutation[0]
    print(f'ERA {solver.get_value(GameState(batter=first_batter))}\n')
    print(f'Solved in {solver.num_programs} programs')

    solver.save('solved_policy.blosc2')


def main(debug: bool = False, load=False):
    if not debug:
        bd = BaseballData(load_pitches=False)

        # The resulting ERA is highly dependent on the pitchers and batters chosen
        # For these kinds of tests we ought to only look at players with more appearances than min_obp_cutoff (167)

        # A Cardinals lineup vs Aaron Nola
        full_matchup = (605400, rosters['cardinals'])  # ERA 5.13

        test_era(bd, *full_matchup, load=load)
    else:
        distributions = load_blosc2('distributions.blosc2')
        transition_distribution = load_blosc2('transition_distribution.blosc2')
        solver = PolicySolver.from_saved('solved_policy.blosc2')
        print(f'ERA {solver.get_value()}')
        pass  # Do something with the data or just examine it


if __name__ == '__main__':
    seed()
    BaseballData()
    main(debug=False, load=False)