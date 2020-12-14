"""
Fine tuning functions for ARIMAX
Authors: Vincent Yu
Date: 12/13/2020
"""

# Python imports
import os
import pickle
import time
import warnings

# Package imports
import numpy as np

# Local imports
from data.dataset import generate_dset_ARIMAX
from data.info import population_by_state
from models.arimax import ARIMAX
from models.settings import random, random_size_ARIMAX, params_ARIMAX, rescale
from utils.utils import get_combos

# Declare all the available states
states = list(population_by_state.keys())

# Suppress ARIMA warnings
warnings.filterwarnings('ignore')


def fine_tune_ARIMAX(random=True, random_size=10, time_est=False,
                     verbose=False):
    """Fine tune ARIMAX models for case and death predictions

    Since there are 51 "states" to run models on, it is recommended that
        random is set to True, since each hyper-parameter combination can
        take about 7 minutes or 1 hour, depending on the convergence conditions.

    Args:
        random:         A boolean representing whether the function is
                            conducting random grid search or not.
                            Default = True
        random_size:    An integer representing the number of random
                            hyper-parameter combinations to select from.
                            Default = 10
        time_est:       A boolean representing whether the function should
                            estimate the total run time or not.
                            Default = False
        verbose:        A boolean representing whether to print performance
                            information to terminal or not.
                            Default = False

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
    param_combos = get_combos(params_ARIMAX)

    # Randomly select <random_size> many combinations if requested
    if random:
        param_combos = np.random.choice(param_combos, size=random_size)

    # Keep track of model performances
    cases_rmses = {}
    deaths_rmses = {}

    # Declare common string format to make combination hashable
    template = '{}, {}, {}, {}'

    if time_est:
        # Estimate the total time needed for fine tuning
        start_time = time.time()

        # Unpack hyper-parameters
        random_combo = param_combos[0]
        num_extra_states, p, d, q = [val for key, val in random_combo.items()]
        order = (p, d, q)

        # Generate corresponding dataset
        cases, deaths = generate_dset_ARIMAX(num_extra_states=num_extra_states)

        # Pick a state for estimation
        state = states[0]

        # Find validation start and end indices
        val_start, val_end = cases[state]['validate']

        # Initialize model
        cases_model = ARIMAX(order, state)

        # Fit and validate the model with RMSE evaluation
        cases_model.fit(cases)
        cases_model.predict(val_start, val_end)
        cases_model.evaluate()

        end_time = time.time()

        # Format the total time needed
        time_diff = (end_time - start_time) * 51 * 2 * len(param_combos)

        hrs = int(time_diff / 3600)
        mins = int((time_diff - 3600 * hrs) / 60)
        secs = int((time_diff - 3600 * hrs - 60 * mins))

        print(f'Total amount of time needed: {hrs} hrs {mins} minutes {secs} '
              f'seconds.')

    # Begin fine tuning
    for i, combo in enumerate(param_combos):

        if len(param_combos) > 1:
            print(f'Trying Combination No.{i + 1}')

        # Keep track of each state's performance on both datasets
        cases_rmse = []
        deaths_rmse = []

        # Unpack hyper-parameters
        param_vals = [val for key, val in combo.items()]
        num_extra_states, p, d, q = param_vals
        order = (p, d, q)

        # Generate corresponding dataset
        cases, deaths = generate_dset_ARIMAX(num_extra_states=num_extra_states)
        
        for state in states:

            # Unpack the state population as the weight of RMSE later on
            state_ppl = population_by_state[state]

            # Initialize cases model
            cases_model = ARIMAX(order, state)

            # Find validation start and end indices
            val_start, val_end = cases[state]['validate']

            # Fit and validate the model with RMSE evaluation
            cases_model.fit(cases)
            cases_model.predict(val_start, val_end)
            cases_model.evaluate()

            # Find the reconstructed RMSE: upscale by population and
            # descale by the previous rescale factor in settings.py
            cases_val_rmse = cases_model.rmse * state_ppl / rescale

            cases_rmse.append(cases_val_rmse)

            # Initialize deaths model
            deaths_model = ARIMAX(order, state)

            # Find validation start and end indices
            val_start, val_end = deaths[state]['validate']

            # Fit and validate the model with RMSE evaluation
            deaths_model.fit(deaths)
            deaths_model.predict(val_start, val_end)
            deaths_model.evaluate()

            # Find the reconstructed RMSE: upscale by population and
            # descale by the previous rescale factor in settings.py
            deaths_val_rmse = deaths_model.rmse * state_ppl / rescale

            deaths_rmse.append(deaths_val_rmse)

        # Convert the hyper-parameter value dictionary into hashable data type
        combo_key = template.format(*param_vals)

        # Find the average RMSE for each dataset
        cases_avg_rmse = sum(cases_rmse) / len(cases_rmse)
        cases_rmses[combo_key] = cases_avg_rmse

        deaths_avg_rmse = sum(deaths_rmse) / len(deaths_rmse)
        deaths_rmses[combo_key] = deaths_avg_rmse

        if verbose:
            print(f'Number of Extra States: {num_extra_states}')
            print(f'Order: {order}')
            print(f'Cases Average RMSE: {cases_avg_rmse}')
            print(f'Deaths Average rmse: {deaths_avg_rmse}')

    return cases_rmses, deaths_rmses


def main():
    """Main Driver Function to Fine Tune ARIMAX and LSTM"""

    # Fine tune ARIMAX
    cases_log_ARIMAX, deaths_log_ARIMAX = fine_tune_ARIMAX(
        random=random,
        random_size=random_size_ARIMAX,
        verbose=True
    )

    # Log the performance to performance dictionary
    params_dst_ARIMAX = os.getcwd() + '/models/params/arimax_params.pkl'
    if os.path.exists(params_dst_ARIMAX):
        with open(params_dst_ARIMAX, 'rb') as f:
            ft_params_arimax = pickle.load(f)

        # Update the existing dictionary
        ft_params_arimax['cases'] += list(cases_log_ARIMAX.items())
        ft_params_arimax['deaths'] += list(deaths_log_ARIMAX.items())

    else:
        # Create params / performance dictionary
        ft_params_arimax = {
            'cases': list(cases_log_ARIMAX.items()),
            'deaths': list(deaths_log_ARIMAX.items())
        }

    # Pickle the params for model testing and predictions
    with open(params_dst_ARIMAX, 'wb') as f:
        pickle.dump(ft_params_arimax, f)


if __name__ == '__main__':
    main()

    '''
    # Uncomment this section for sanity check if needed
    pkl_dst = os.getcwd() + '/models/params/arimax_params.pkl'
    with open(pkl_dst, 'rb') as f:
        dct = pickle.load(f)

    print(dct)
    '''
