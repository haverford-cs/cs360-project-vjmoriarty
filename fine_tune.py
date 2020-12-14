"""
Fine tuning functions for ARIMAX and LSTM
Authors: Vincent Yu
Date: 12/13/2020
"""

# Python imports
import os
import pickle
import time

# Package imports
import numpy as np
import tensorflow as tf

# Local imports
from data.dataset import generate_dset_ARIMAX, generate_dset_LSTM
from data.info import population_by_state
from models.arimax import run_arimax
from models.conv_lstm import run_lstm
from models.settings import random, random_size, params_ARIMAX, params_LSTM, \
    epochs, rescale
from utils.utils import get_combos

# Declare all the available states
states = list(population_by_state.keys())

# Suppress the warnings from tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def fine_tune_ARIMAX(random=True, random_size=30):
    """TODO COMPLETE"""
    param_combos = get_combos(params_ARIMAX)

    if random:
        params_combos = np.random.choice(param_combos, size=random_size)

    for combo in param_combos:
        num_extra_states, p, d, q = [val for key, val in combo.items()]

        order = (p, d, q)

    return


def fine_tune_LSTM(random=True, random_size=30):
    """Fine tune ConvLSTM models for case and death predictions

    Since there are 51 "states" to run models on, it is recommended that
        random is set to True, since each hyper-parameter combination can
        take about 17 minutes (with 10 seconds per 25 epochs per state).

    Args:
        random:         A boolean representing whether the function is
                            conducting random grid search or not.
                            Default = True
        random_size:    An integer representing the number of random
                            hyper-parameter combinations to select from.
                            Default = 30

    Returns:
        cases_rmse:     A dictionary representing the model performance on
                            the case numbers for every hyper-parameter
                            combinations. It has the format of:
                            {combination --> weighted average RMSE}
        deaths_rmse:    A dictionary representing the model performance on
                            the deaths numbers for every hyper-parameter
                            combinations. It has the format of:
                            {combination --> weighted average RMSE}
    """

    # Find all combinations from the grid provided in settings.py
    param_combos = get_combos(params_LSTM)

    # Randomly select <random_size> many combinations if requested
    if random:
        param_combos = np.random.choice(param_combos, size=random_size)

    # Keep track of model performances
    cases_rmses = {}
    deaths_rmses = {}

    # Declare common string format to make combination hashable
    template = '{}, {}, {}, {}, {}'

    # Estimate the total time needed for fine tuning
    start_time = time.time()

    random_combo = param_combos[0]
    cases, deaths = generate_dset_LSTM(**random_combo)

    # Train and validate the model on case numbers first
    cases_train, cases_val = cases['Alabama']['train'], \
                             cases['Alabama']['validate']

    _ = run_lstm(cases_train, cases_val, epochs=epochs, verbose=0)

    # Train and validate the model on deaths numbers next
    deaths_train, deaths_val = deaths['Alabama']['train'], \
                               deaths['Alabama']['validate']

    _ = run_lstm(deaths_train, deaths_val, epochs=epochs, verbose=0)

    end_time = time.time()

    # Format the total time needed
    time_diff = (end_time - start_time) * 51 * len(param_combos)

    hrs = int(time_diff / 3600)
    mins = int((time_diff - 3600 * hrs) / 60)
    secs = int((time_diff - 3600 * hrs - 60 * mins))

    print(f'Total amount of time needed: {hrs} hrs {mins} minutes {secs} '
          f'seconds.')

    # Begin fine tuning
    for i, combo in enumerate(param_combos):

        print(f'Trying Combination No.{i + 1}')

        # Generate the corresponding time series dataset with the given combo
        cases, deaths = generate_dset_LSTM(**combo)

        # Keep track of each state's performance on both datasets
        cases_rmse = []
        deaths_rmse = []

        for state in states:

            # Unpack the state population as the weight of RMSE later on
            state_ppl = population_by_state[state]

            # Train and validate the model on case numbers first
            cases_train, cases_val = cases[state]['train'], \
                                     cases[state]['validate']

            cases_history = run_lstm(
                cases_train,
                cases_val,
                epochs=epochs,
                verbose=0)

            # Find the reconstructed RMSE: upscale by population and
            # descale by the previous rescale factor in settings.py
            cases_val_rmse = cases_history['val_rmse'][-1] * state_ppl / rescale

            cases_rmse.append(cases_val_rmse)

            # Train and validate the model on deaths numbers next
            deaths_train, deaths_val = deaths[state]['train'], \
                                       deaths[state]['validate']

            deaths_history = run_lstm(
                deaths_train,
                deaths_val,
                epochs=epochs,
                verbose=0)

            # Find the reconstructed RMSE: upscale by population and
            # descale by the previous rescale factor in settings.py
            deaths_val_rmse = deaths_history['val_rmse'][-1] * state_ppl / \
                              rescale

            deaths_rmse.append(deaths_val_rmse)

        # Convert the hyper-parameter value dictionary into hashable data type
        param_vals = [val for key, val in combo.items()]
        combo_key = template.format(*param_vals)

        # Find the average RMSE for each dataset
        cases_avg_rmse = sum(cases_rmse) / len(cases_rmse)
        cases_rmses[combo_key] = cases_avg_rmse

        deaths_avg_rmse = sum(deaths_rmse) / len(deaths_rmse)
        deaths_rmses[combo_key] = deaths_avg_rmse

    return cases_rmses, deaths_rmses


def main():
    """Main Driver Function to Fine Tune ARIMAX and LSTM"""
    # TODO FINE TUNE ARIMAX

    # Fine tune LSTM
    cases_log, deaths_log = fine_tune_LSTM(
        random=random,
        random_size=random_size
    )

    # Take the five best params for cases and deaths predictions
    ft_params_lstm = {
        'cases': sorted(cases_log.items(), key=lambda x: x[1])[:5],
        'deaths': sorted(deaths_log.items(), key=lambda x: x[1])[:5],
    }

    # Pickle the params for model testing and predictions
    params_dst_LSTM = os.getcwd() + '/models/params/lstm_params.pkl'
    with open(params_dst_LSTM, 'wb') as f:
        pickle.dump(ft_params_lstm, f)


if __name__ == '__main__':
    main()

    '''
    # Uncomment this section for sanity check if needed
    pkl_dst = os.getcwd() + '/models/params/lstm_params.pkl'
    with open(pkl_dst, 'rb') as f:
        dct = pickle.load(f)

    print(dct)
    '''
