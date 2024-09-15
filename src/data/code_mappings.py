"""
In this file, we define some mappings from codes in the datasets to our enum values.
Note that these mappings are not perfect and someone with more knowledge ought to look at them.
"""
from src.model.players import MLBTeam
from src.model.state import PitchResult
from src.model.pitch_type import PitchType

# https://www.mlb.com/glossary/pitch-types
pitch_type_mapping = {
    'CH': PitchType.CHANGEUP,
    'CU': PitchType.CURVE,
    'FC': PitchType.CUTTER,
    'EP': PitchType.CURVE,  # Eephus is a type of slow curveball
    'FO': PitchType.CHANGEUP,  # Forkball is a rare type of splitter
    'FF': PitchType.FOUR_SEAM,
    'KN': PitchType.CURVE,  # Knuckleball is distinct but closer to a curve in action
    'KC': PitchType.CURVE,  # Knucle-curve
    'SC': PitchType.CHANGEUP,  # Screwball acts like a changeup with reverse break
    'SI': PitchType.TWO_SEAM,
    'SL': PitchType.SLIDER,
    'SV': PitchType.SLIDER,  # Slurve
    'FS': PitchType.CHANGEUP,  # Splitter is often considered a type of changeup
    'FT': PitchType.TWO_SEAM,
    'ST': PitchType.SLIDER,
    'AB': None,  # One instance of this in the dataset
}

# This list was found by looking at unique values in the dataframe
pitch_result_mapping = {
    # Takes
    'ball': PitchResult.CALLED_BALL,
    'called_strike': PitchResult.CALLED_STRIKE,
    'hit_by_pitch': PitchResult.HIT_SINGLE,

    # Swings
    'swinging_strike': PitchResult.SWINGING_STRIKE,
    'hit_into_play': PitchResult.HIT_SINGLE,

    # Fouls
    'foul': PitchResult.SWINGING_FOUL,
    'foul_tip': PitchResult.SWINGING_STRIKE,
    'foul_bunt': PitchResult.SWINGING_STRIKE,
    'bunt_foul_tip': PitchResult.SWINGING_STRIKE,
    'missed_bunt': PitchResult.SWINGING_STRIKE,

    # Intentional balls
    'intent_ball': PitchResult.CALLED_BALL,
    'pitchout': PitchResult.CALLED_BALL,
    'swinging_pitchout': PitchResult.SWINGING_STRIKE,

    # Blocks
    'blocked_ball': PitchResult.CALLED_BALL,
    'swinging_strike_blocked': PitchResult.SWINGING_STRIKE,
}

# A few notes,
# 1. Currently, we only consider the events differentiating between how far the batter goes on HIT_SINGLE.
#    Ultimately, the outcome of a pitch and the outcome of the at-bat event are separate steps
#    and should probably be treated as such. The line does get blurred when the pitch itself ends the
#    at-bat, when a walk or hit by pitch occurs. If we start to consider on field events, this will need to change.
# 2. If we decide to learn a distribution of on-field outcomes, we might be best off ignoring these codes and simply
#    looking at the state of the field after the play.
# 3. Some of these events do not happen at the end of the at-bat
at_bat_event_mapping = {
    # Regular hits
    'single': PitchResult.HIT_SINGLE,
    'double': PitchResult.HIT_DOUBLE,
    'triple': PitchResult.HIT_TRIPLE,
    'home_run': PitchResult.HIT_HOME_RUN,

    # Walks (already covered in pitch_result_mapping)
    'hit_by_pitch': PitchResult.HIT_SINGLE,
    'walk': None,
    'catcher_interf': None,
    'intent_walk': None,

    # Outs
    'field_out': PitchResult.HIT_OUT,
    'force_out': PitchResult.HIT_OUT,
    'other_out': PitchResult.HIT_OUT,

    'strikeout': None,
    'strikeout_double_play': None,

    # Different plays that result in num_outs
    'double_play': PitchResult.HIT_OUT,
    'triple_play': PitchResult.HIT_OUT,

    'grounded_into_double_play': PitchResult.HIT_OUT,  # Typically a force out, maybe we should differentiate by looking at the state after the play
    'sac_fly': PitchResult.HIT_OUT,
    'sac_fly_double_play': PitchResult.HIT_OUT,
    'fielders_choice': PitchResult.HIT_OUT,  # Technically not a normal out
    'fielders_choice_out': PitchResult.HIT_OUT,
    'sac_bunt': PitchResult.HIT_OUT,
    'sac_bunt_double_play': PitchResult.HIT_OUT,

    'field_error': PitchResult.HIT_SINGLE,
    'passed_ball': PitchResult.HIT_SINGLE,
    'wild_pitch': PitchResult.HIT_SINGLE,

    # Events that do not end the at-bat
    'caught_stealing_2b': None,
    'caught_stealing_3b': None,
    'caught_stealing_home': None,
    'ejection': None,
    'pickoff_1b': None,
    'pickoff_2b': None,
    'pickoff_error_2b': None,
    'pickoff_caught_stealing_2b': None,
    'pickoff_caught_stealing_3b': None,
    'pickoff_caught_stealing_home': None,
    'stolen_base_3b': None,
}

# This is a mapping from team names to team codes
team_name_mapping = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago White Sox': 'CWS',
    'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN',
    'Cleveland Indians': 'CLE',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC',
    'Los Angeles Angels': 'ANA',
    'Los Angeles Dodgers': 'LA',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WAS'
}

# This is a mapping from team codes to our MLBTeam enum
team_code_mapping = {
    'ARI': MLBTeam.DIAMONDBACKS,
    'ATL': MLBTeam.BRAVES,
    'BAL': MLBTeam.ORIOLES,
    'BOS': MLBTeam.RED_SOX,
    'CWS': MLBTeam.WHITE_SOX,
    'CHC': MLBTeam.CUBS,
    'CIN': MLBTeam.REDS,
    'CLE': MLBTeam.INDIANS,
    'COL': MLBTeam.ROCKIES,
    'DET': MLBTeam.TIGERS,
    'HOU': MLBTeam.ASTROS,
    'KC': MLBTeam.ROYALS,
    'ANA': MLBTeam.ANGELS,
    'LA': MLBTeam.DODGERS,
    'MIA': MLBTeam.MARLINS,
    'MIL': MLBTeam.BREWERS,
    'MIN': MLBTeam.TWINS,
    'NYM': MLBTeam.METS,
    'NYY': MLBTeam.YANKEES,
    'OAK': MLBTeam.ATHLETICS,
    'PHI': MLBTeam.PHILLIES,
    'PIT': MLBTeam.PIRATES,
    'SD': MLBTeam.PADRES,
    'SF': MLBTeam.GIANTS,
    'SEA': MLBTeam.MARINERS,
    'STL': MLBTeam.CARDINALS,
    'TB': MLBTeam.RAYS,
    'TEX': MLBTeam.RANGERS,
    'TOR': MLBTeam.BLUE_JAYS,
    'WAS': MLBTeam.NATIONALS
}