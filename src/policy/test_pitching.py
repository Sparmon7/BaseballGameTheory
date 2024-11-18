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


for i in default.ZONES:
    print(i)
    
print()
for i in default.COMBINED_ZONES: 
    print(i)


bd = BaseballData(load_pitches=False) 
type A = tuple[PitchType, int]  
pitcher_actions: list[A] = [(pitch_type, zone_i) for zone_i in range(len(default.ZONES)) for pitch_type in PitchType]  

def calculate_pitcher_control_distribution( pitchers: list[int], batch_size=528):
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
                                                pitchers=[bd.pitchers[pitcher_i] for pitcher_i in pitchers],
                                                empty_data=True)
    dataloader = DataLoader(pitch_control_dataset, batch_size=128, shuffle=False)

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
        pitcher_control[pitcher] = np.zeros((len(pitcher_actions), len(default.COMBINED_ZONES)), dtype=np.float32)

        for pitch_i, pitch in enumerate(pitcher_actions):
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



pitcher = calculate_pitcher_control_distribution(pitchers = [605400])
data = pitcher[605400]
for i in data:
    print(i)
    print(len(i))
    print()

#bd = BaseballData(load_pitches=False)
