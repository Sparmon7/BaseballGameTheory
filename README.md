## Improving the Scalability and Realism of Game-Theoretic Approaches for Baseball
This project models baseball through a game-theoretic lens using MLB data


### Getting Started
1. Create a virtual environment (myenv will be ignored by git).
2. Install requirements using `pip install -r requirements.txt`
3. In the env file in src/.env, set the folder variable to the absolute path of the BaseballGameTheory folder.
4. Create a folder titled statcast inside the raw data folder.
5. Fetch the raw data with `raw_data/fetch_raw_data.py`
6. Process the data with `src/data/data_loading.py`
7. Try out the zero-sum stochastic game model with `src/policy/optimal_policy.py` (run this from inside the policy folder)
8. Try the batting lineup optimization scripts with `src/policy/batting_order_optimization.py` (run this from inside the policy folder)
9. Feel free to load the data with `bd = BaseballData()` and experiment!

### Project Structure
- `model_weights/` contains pre-trained models for the distributions
- `presentation/` contains the research poster and write-up
- `src/` contains the made codebase
  - `src/data/` contains the data processing scripts and Pytorch datasets
  - `src/distributions/` contains the Pytorch models for learning the distributions
  - `src/model/` contains the object classes for the game model, like players, zones, pitches, etc.
  - `src/policy/` contains the zero-sum stochastic game model and work on batting lineup optimization

### My Contributions
- Implented stochastic runners to accurately model speed
- Sped up ERA computation by dynamically storing inning results
- Modified batter representation to enhance swing outcome neural network calculation
- Enhanced model accuracy by limiting pitchers' arsenals

### Future Ideas
- Batter/pitcher handedness
- Trade deadline targets
- Pinch hitting strategy
- Stolen bases strategy on a runner/catcher basis
- Incorporate on-field events like sacrifices, double plays, stretching, etc.
- Bullpen strategy




Old repository: https://github.com/BOBONA/ZeroSumBaseball
