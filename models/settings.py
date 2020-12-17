"""
Storage for script settings and parameter grids
Authors: Vincent Yu
Date: 12/03/2020
"""

# GENERAL SETTINGS
# Set to true to re-read the latest version of the dataset csv files
update_dset = False

# Set the rescale factor to inflate the dataset
rescale = 1000

# FINE TUNING RELATED
# Set the random grid search status
random = True
random_size_LSTM = 50
random_size_ARIMAX = 1

# MODEL RELATED
# Set ARIMAX fine tuning parameters
params_ARIMAX = {
    'num_extra_states': [i for i in range(6)],
    'p': [i for i in range(5, 10)],
    'd': [i for i in range(2)],
    'q': [i for i in range(2)]
}

# Set LSTM fine tuning parameters
params_LSTM = {
    'order': [i for i in range(15)],
    'num_extra_states': [i for i in range(2, 7)],
    'cases_lag': [i for i in range(11)],
    'deaths_lag': [i for i in range(10)],
    'aug_lag': [i for i in range(10)]
}

# Some LSTM related hyper-parameters
batch_size = 10
epochs = 25

# PREDICTION RELATED
# Testing or Predicted
pred_mode = True

# Rolling prediction iterations
rolling_iters = 10