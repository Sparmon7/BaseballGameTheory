from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.getenv('FOLDER'))

import json
import os.path
import pickle
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple

import blosc2
import numpy as np
import pandas
import torch
from pandas import DataFrame
from pybaseball import player_search_list
from torch import nan_to_num, Tensor
from torch.distributions import MultivariateNormal

try:
    # noinspection PyUnresolvedReferences
    ipython_name = get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm


from src.data.code_mappings import pitch_type_mapping, pitch_result_mapping, at_bat_event_mapping, team_code_mapping, \
    team_name_mapping
from src.model.state import GameState, PitchResult, Rules
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Batter, Pitcher, Runner, min_obp_cutoff, min_run_attempts
from src.model.zones import Zones, default


def load_blosc2(path: str):
    """Loads a file compressed with pickle and blosc2"""

    with open(path, 'rb') as f:
        return pickle.loads(blosc2.decompress(f.read()))


def save_blosc2(data, path: str):
    """Saves an object to a file compressed with pickle and blosc2, creating directories if necessary"""

    if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as f:
        f.write(blosc2.compress(pickle.dumps(data)))


def fill_partial_stat(stat: Tensor):
    """Utility function"""

    for zone in default.ZONES:
        if len(zone.coords) > 1:
            for coord in zone.coords[1:]:
                stat[:, *coord] = stat[:, *zone.coords[0]]
    return stat


class BaseballData:
    """
    This class loads, processes, and stores the baseball data used for training and evaluation.
    More specific datasets are created for training specific models.

    We use pickle and blosc2 to compress data very efficiently.
    """

    default_processed_data_dir = '../../processed_data/'
    
    year_range = range(2008, 2025)

    def __init__(self, load_pitches: bool = True, load_players: bool = True,
                 processed_data_dir: str = default_processed_data_dir):
        self.pitches = None
        self.players = None

        if load_pitches:
            self.pitches = []
            for year in tqdm(self.year_range, desc='Loading pitches'):
                self.pitches.extend(load_blosc2(processed_data_dir + f'{year}.blosc2'))

        if load_players:
            players = load_blosc2(processed_data_dir + 'players.blosc2')
            self.pitchers: defaultdict[int, Pitcher] = players['pitchers']
            self.batters: defaultdict[int, Batter] = players['batters']
            self.runners : defaultdict[int, Runner] = players['runners']

    @classmethod
    def process_data(cls, raw_data_dir: str = '../../raw_data/statcast/', processed_data_dir: str = default_processed_data_dir):
        """
        Load raw baseball data from the specified directory.
        """

        print('Generating baseball data (this will only happen once)...')

        # Keep track of players and statistics
        pitchers: defaultdict[int, Pitcher] = defaultdict(Pitcher)
        batters: defaultdict[int, Batter] = defaultdict(Batter)
        runners: defaultdict[int, Runner] = defaultdict(Runner)

        pitcher_all_at_bats = defaultdict(set)
        pitcher_hits_against = defaultdict(int)
        pitch_statistics_shape = (len(PitchType), Zones.DIMENSION, Zones.DIMENSION)
        pitcher_total_thrown = defaultdict(lambda: torch.zeros(pitch_statistics_shape))
        pitcher_total_velocity = defaultdict(lambda: torch.zeros(pitch_statistics_shape))
        pitch_locations_30 = defaultdict(lambda: defaultdict(list))  # To calculate pitcher control

        batter_all_at_bats = defaultdict(set)
        batter_hits = defaultdict(int)
        batter_total_encountered = defaultdict(lambda: torch.zeros(pitch_statistics_shape))
        batter_total_swung = defaultdict(lambda: torch.zeros(pitch_statistics_shape))
        batter_total_hits = defaultdict(lambda: torch.zeros(pitch_statistics_shape))
        
        runner_shape = (2,3,3) #2 bases, 3 possible starting bases, 3 ending bases
        runner_results = defaultdict(lambda: torch.zeros(runner_shape))

        velocities = []  # To normalize the velocity data

        # Load the pitches, year by year
        for year in tqdm(cls.year_range):
            pitch_data: DataFrame = load_blosc2(f'{raw_data_dir}{year}.blosc2')
            pitch_data = pitch_data.replace({np.nan: None})

            pitches = []
            #runner calculation variables
            prior_1 = None
            prior_2 = None
            prior_3 = None
            prior_res = None
            prior_game = None
            for row in pitch_data[::-1].itertuples(index=False):
                
                row: NamedTuple  # Consult https://baseballsavant.mlb.com/csv-docs for column names
                state = GameState(inning=row.inning - 1, bottom=row.inning_topbot == 'Bot',
                                  balls=row.balls, strikes=row.strikes, runs=row.bat_score,
                                  outs=row.outs_when_up, first=bool(row.on_1b),
                                  second=bool(row.on_2b), third=bool(row.on_3b))
                
            
                pitch_type = pitch_type_mapping.get(row.pitch_type, None)

                zone_idx = None
                plate_x, plate_z = None, None
                if row.sz_top is not None:
                    plate_x, plate_z = 12 * row.plate_x, 12 * row.plate_z  # Convert from feet to inches
                    zones = default if row.sz_bot < 0.3 else Zones(sz_bottom=12 * row.sz_bot, sz_top=12 * row.sz_top)
                    zone_idx, zone = zones.get_zone(plate_x, plate_z)

                pitch_outcome = pitch_result_mapping.get(row.description, None)
                
                if pitch_outcome == PitchResult.HIT_SINGLE:
                    pitch_outcome = at_bat_event_mapping.get(row.events, PitchResult.HIT_SINGLE)
               
                #updating runner counts
                if prior_res is not None:
                    if prior_game == row.game_pk: #checks to make sure same game in case of walk off or delay
                        hitType=1
                        if prior_res == PitchResult.HIT_SINGLE:
                            hitType=0
                        for index, runnerId in enumerate([prior_1, prior_2, prior_3]):
                            if runnerId is not None:
                                if row.on_2b == runnerId:
                                    runner_results[runnerId][hitType,index,0] += 1
                                elif row.on_3b == runnerId:
                                    runner_results[runnerId][hitType,index,1] += 1
                                else:
                                    runner_results[runnerId][hitType,index,2] += 1 
                    prior_res = None
                    prior_1 = None
                    prior_2= None
                    prior_3 = None
                    prior_game = None
                    
                if (pitch_outcome == PitchResult.HIT_SINGLE or pitch_outcome == PitchResult.HIT_DOUBLE) and (row.on_1b != None or row.on_2b != None or row.on_3b !=None):  
                    prior_res = pitch_outcome
                    prior_1 = row.on_1b
                    prior_2= row.on_2b
                    prior_3 = row.on_3b
                    prior_game = row.game_pk

                    

                pitch = Pitch(game_state=state, batter_id=row.batter, pitcher_id=row.pitcher,
                              pitch_type=pitch_type, location=zone_idx, pitch_result=pitch_outcome,
                              speed=row.release_speed, plate_x=plate_x, plate_z=plate_z,
                              game_id=row.game_pk, at_bat_num=row.at_bat_number,
                              pitch_num=row.pitch_number, home_team=team_code_mapping.get(row.home_team, None),
                              away_team=team_code_mapping.get(row.away_team, None))

                # Update the pitch entry
                pitches.append(pitch)

                # Update player statistics
                if pitch.is_valid() and pitch.speed is not None:
                    # Pitcher
                    pitcher_all_at_bats[pitch.pitcher_id].add((pitch.game_id, pitch.at_bat_num))
                    pitcher_hits_against[pitch.pitcher_id] += int(pitch.result.batter_hit())

                    zone_coord = default.COMBINED_ZONES[pitch.zone_idx].coords[0]
                    loc = (pitch.type, *zone_coord)
                    pitcher_total_thrown[pitch.pitcher_id][*loc] += 1
                    pitcher_total_velocity[pitch.pitcher_id][*loc] += pitch.speed
                    velocities.append(pitch.speed)

                    if pitch.game_state.balls == 3 and pitch.game_state.strikes == 0:
                        pitch_locations_30[pitch.pitcher_id][pitch.type].append((pitch.plate_x, pitch.plate_z))

                    # Batter
                    batter_all_at_bats[pitch.batter_id].add((pitch.game_id, pitch.at_bat_num))
                    batter_hits[pitch.batter_id] += int(pitch.result.batter_hit())

                    batter_total_encountered[pitch.batter_id][*loc] += 1
                    batter_total_swung[pitch.batter_id][*loc] += int(pitch.result.batter_swung())
                    batter_total_hits[pitch.batter_id][*loc] += int(pitch.result.batter_hit())

            save_blosc2(pitches, processed_data_dir + f'{year}.blosc2')

        # Add the aggregate statistics to the players, replacing blank statistics with zeros
        velocity_mean = torch.mean(torch.tensor(velocities))
        velocity_std = torch.std(torch.tensor(velocities))

        # Aggregate pitcher statistics
        for pitcher_id in pitcher_all_at_bats.keys():
            pitcher = pitchers[pitcher_id]

            pitcher.num_batters_faced = len(pitcher_all_at_bats[pitcher_id])
            pitcher.obp = pitcher_hits_against[pitcher_id] / pitcher.num_batters_faced
            pitcher.set_throwing_frequency_data(fill_partial_stat(pitcher_total_thrown[pitcher_id]))

            avg_velocity = fill_partial_stat(pitcher_total_velocity[pitcher_id] / pitcher_total_thrown[pitcher_id])
            normalized_velocity = nan_to_num((avg_velocity - velocity_mean) / velocity_std)

            pitcher.set_average_velocity_data(normalized_velocity)

        # Pitch control distributions
        jitter = torch.eye(2) * 1e-5  # Helps with positive definiteness
        for pitcher_id, pitch_counts in pitch_locations_30.items():
            for pitch_type, locations in pitch_counts.items():
                if len(locations) > 5:
                    locations_tensor = torch.tensor(locations, dtype=torch.float32)
                    mean = torch.mean(locations_tensor, dim=0)
                    covar = torch.cov(locations_tensor.T) + jitter
                    try:
                        pitchers[pitcher_id].estimated_control[pitch_type] = MultivariateNormal(mean, covar)
                    except ValueError:  # If the covariance matrix is not positive definite
                        pass

        # Aggregate batter statistics
        for batter_id in batter_all_at_bats.keys():
            batter = batters[batter_id]
            batter.num_at_bats = len(batter_all_at_bats[batter_id])
            batter.obp = batter_hits[batter_id] / batter.num_at_bats
            batter.set_swinging_frequency_data(nan_to_num(fill_partial_stat(batter_total_swung[batter_id])))
            batter.set_batting_average_data(nan_to_num(fill_partial_stat(batter_total_hits[batter_id] / batter_total_encountered[batter_id])))

        # Add the OBP statistics to the players
        batter_by_obp = sorted(filter(lambda b: b.num_at_bats > min_obp_cutoff, batters.values()), key=lambda b: b.obp)
        pitcher_by_obp = sorted(filter(lambda p: p.num_batters_faced > min_obp_cutoff, pitchers.values()), key=lambda p: p.obp, reverse=True)
        for idx, batter in enumerate(batter_by_obp):
            batter.obp_percentile = idx / len(batter_by_obp)
        for idx, pitcher in enumerate(pitcher_by_obp):
            pitcher.obp_percentile = idx / len(pitcher_by_obp)

       
        # Aggregate runner statistics     
        runner_totals = torch.zeros(runner_shape)
        for value in runner_results.values():
            runner_totals += value
        
        #change counts to probabilities
        for hitType in range(2):
            for starting_base in range(3):
                runner_totals[hitType,starting_base] /= sum(runner_totals[hitType,starting_base]) 
        for runner_id in batter_all_at_bats.keys():
            runner = runner_results[runner_id]
            for hitType in range(2):
                for starting_base in range(3):
                    if sum(runner[hitType,starting_base]) > min_run_attempts:
                        runner[hitType,starting_base] /= sum(runner[hitType,starting_base]) 
                    else:
                        runner[hitType,starting_base] = runner_totals[hitType,starting_base] 
            runners[runner_id] = Runner(runner)


        players = {
            'pitchers': pitchers,
            'batters': batters,
            'runners': runners
        }

        save_blosc2(players, processed_data_dir + 'players.blosc2')

        print('Done')

    def get_lineups(self, require_percentile=True, rules=Rules) -> list[tuple[int, tuple[int, ...]]]:
        """
        Assuming pitches are loaded in order, returns the lineups for each game.
        TODO, this is currently inaccurate since it doesn't consider which team is batting.
        What we ought to do is remove the game level info from Pitch and make a separate Game class.
        Then, in our process method, we can obtain these lineups and store it in that class.
        """

        lineups: dict[int, dict[int, Batter]] = defaultdict(dict)
        for pitch in self.pitches:
            lineup = lineups[pitch.game_id]
            if pitch.batter_id not in lineup and len(lineup) < rules.num_batters:
                lineup[pitch.batter_id] = self.batters[pitch.batter_id]

        result = []
        for game_id, lineup in lineups.items():
            if len(lineup) == rules.num_batters and (not require_percentile or all(b.obp_percentile is not None for b in lineup.values())):
                result.append((game_id, tuple(lineup.keys())))
        return result

    def parse_trades(self, raw_transactions_file: str = '../../raw_data/transactions.json'):
        """
        Assuming players are loaded in order, returns the single player trades which we have data for.
        Currently, this only outputs 67 trades, since most trades are not for players with enough at-bats.
        """

        with open(raw_transactions_file, 'r', encoding='utf-8') as f:
            transactions = json.load(f)

        # Fetch the player names
        player_names = []
        for transaction in transactions:
            for player in transaction['team1']['acquires'] + transaction.get('team2', {}).get('acquires', []):
                name_parts = player.split()
                if len(name_parts) >= 2:
                    player_names.append((name_parts[1], name_parts[0]))

        # This might use an older Pandas version (and you'll need to change append to _append)
        search_results: pandas.DataFrame = player_search_list(player_names)

        # Find valid single player trades
        single_player_trades = []
        for transaction in transactions:
            if len(transaction['team1']['acquires']) == 1 and len(transaction.get('team2', {}).get('acquires', [])) == 1:
                player1_parts = transaction['team1']['acquires'][0].split()
                player2_parts = transaction['team2']['acquires'][0].split()

                if len(player1_parts) < 2 or len(player2_parts) < 2:
                    continue

                first_name1, last_name1 = player1_parts[0:2]
                first_name2, last_name2 = player2_parts[0:2]
                player1_id = search_results[(search_results['name_first'] == first_name1.lower()) & (search_results['name_last'] == last_name1.lower())]['key_mlbam'].values
                player2_id = search_results[(search_results['name_first'] == first_name2.lower()) & (search_results['name_last'] == last_name2.lower())]['key_mlbam'].values

                if not len(player1_id) or not len(player2_id):
                    continue

                player1_id = player1_id[0]
                player2_id = player2_id[0]

                if (player1_id in self.batters and self.batters[player1_id].num_at_bats > 50 and
                        player2_id in self.batters and self.batters[player2_id].num_at_bats > 50):
                    team1 = team_code_mapping[team_name_mapping[transaction['team1']['name']]]
                    team2 = team_code_mapping[team_name_mapping[transaction.get('team2', {}).get('name', '')]]
                    date = datetime.strptime(transaction['date'], '%b %d, %Y').date()

                    single_player_trades.append((date, team1, player2_id, team2, player1_id))

        save_blosc2(single_player_trades, f'{self.default_processed_data_dir}trades.blosc2')


if __name__ == '__main__':
    BaseballData.process_data()
