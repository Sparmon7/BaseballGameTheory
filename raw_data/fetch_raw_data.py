import pickle

import blosc2
from pybaseball import statcast

# Enable the cache at your own peril, on my machine each request also generated tiny little dummy files
# pybaseball.cache.enable()

# pybaseball is glitchy and will sometimes freeze, I apologize in advance. If that happens, update the year range
# and try again. It's still better than scraping it ourselves.
for year in range(2008, 2025):
    raw_data = statcast(start_dt=f'{year}-01-01', end_dt=f'{year + 1}-01-01', verbose=False)
    pickled_data = pickle.dumps(raw_data)
    compressed_data = blosc2.compress(pickled_data)
    with open(f'statcast/{year}.blosc2', 'wb') as f:
        f.write(compressed_data)
