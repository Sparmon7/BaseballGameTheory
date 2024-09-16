### Getting Started
1. Create a folder titled statcast inside the raw data folder.
2. Fetch the raw data with `raw_data/fetch_raw_data.py`
3. Process the data with `src/data/data_loading.py`
4. Try out the zero-sum stochastic game model with `src/policy/optimal_policy.py`
5. Try the batting lineup optimization scripts with `src/policy/batting_order_optimization.py`
6. Feel free to load the data with `bd = BaseballData()` and experiment!

### Project Structure
- `model_weights/` contains pre-trained models for the distributions
- `presentation/` contains the research poster and write-up
- `src/` contains the made codebase
  - `src/data/` contains the data processing scripts and Pytorch datasets
  - `src/distributions/` contains the Pytorch models for learning the distributions
  - `src/model/` contains the object classes for the game model, like players, zones, pitches, etc.
  - `src/policy/` contains the zero-sum stochastic game model and work on batting lineup optimization
