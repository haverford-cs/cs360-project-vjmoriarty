"""
Model testing and rolling prediction script.
Authors: Vincent Yu
Date: 12/14/2020
"""

# Python imports
from math import sqrt
import os
import pickle
import warnings

# Package imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Local imports
from data.dataset import generate_dset_ARIMAX, generate_dset_LSTM
from data.info import population_by_state, nearest_states
from models.arimax import ARIMAX
from models.conv_lstm import run_lstm
from models.settings import epochs, rescale, batch_size, rolling_iters, \
    pred_mode

# Suppress ARIMA warnings
warnings.filterwarnings('ignore')

# Suppress the warnings from tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Declare all the available states
states = list(population_by_state.keys())

# Declare fine tuned parameters paths for each model
ft_dst_LSTM = os.getcwd() + '/models/params/lstm_params.pkl'
ft_dst_ARIMAX = os.getcwd() + '/models/params/arimax_params.pkl'


def test_ARIMAX():
    """TODO DOCUMENTATION"""

    # Unpack top 5 hyper-parameter combinations
    with open(ft_dst_ARIMAX, 'rb') as f:
        ft_params = pickle.load(f)

    # Keep track of testing RMSEs
    test_rmses = {'cases': [], 'deaths': []}

    for dset_type in test_rmses:

        params = [i[0] for i in ft_params[dset_type]]

        for combo in params:

            num_extra_states, p, d, q = [int(i) for i in combo.split(', ')]
            order = (p, d, q)

            # Log each state's performance on the cases/deaths datasets
            combo_rmses = []

            # Generate corresponding dataset
            cases, deaths = generate_dset_ARIMAX(
                num_extra_states=num_extra_states
            )

            # Identify the dataset to be used for testing
            dset = cases if dset_type == 'cases' else deaths

            for state in states:

                # Unpack the state population as the weight of RMSE later on
                state_ppl = population_by_state[state]

                # Initialize the ARIMAX model
                model = ARIMAX(order, state)

                # Find validation start and end indices
                test_start, test_end = dset[state]['test']

                # Fit and validate the model with RMSE evaluation
                model.fit(dset)
                model.predict(test_start, test_end)
                model.evaluate()

                # Find the reconstructed RMSE: upscale by population and
                # descale by the previous rescale factor in settings.py
                test_rmse = model.rmse * state_ppl / rescale

                combo_rmses.append(test_rmse)

            # Find the average RMSE of all states
            avg_rmse = sum(combo_rmses) / len(combo_rmses)
            test_rmses[dset_type].append(avg_rmse)

    return test_rmses


def test_LSTM():
    """TODO DOCUMENTATION"""

    # Unpack top 5 hyper-parameter combinations
    with open(ft_dst_LSTM, 'rb') as f:
        ft_params = pickle.load(f)

    # Keep track of testing RMSEs
    test_rmses = {'cases': [], 'deaths': []}

    for dset_type in test_rmses:

        params = [i[0] for i in ft_params[dset_type]]

        for combo in params:
            formatted_params = {
                'lag': 0,
                'num_extra_states': 0,
                'cases_offset': 0,
                'deaths_offset': 0,
                'aug_offset': 0
            }

            param_vals = [int(i) for i in combo.split(', ')]

            for i, key in enumerate(list(formatted_params.keys())):
                formatted_params[key] = param_vals[i]

            # Log each state's performance on the cases/deaths datasets
            combo_rmses = []

            # Generate corresponding dataset
            cases, deaths = generate_dset_LSTM(**formatted_params)

            # Identify the dataset to be used for testing
            dset = cases if dset_type == 'cases' else deaths

            for state in states:

                # Unpack the state population as the weight of RMSE later on
                state_ppl = population_by_state[state]

                # Train and validate the model on case numbers first
                train, val = dset[state]['train'], dset[state]['validate']
                X_test, y_test = dset[state]['test']

                _, model = run_lstm(train, val, epochs=epochs, verbose=0)

                y_pred = model.predict(X_test, batch_size=batch_size)

                # Find and reconstruct RMSE: upscale by population and
                # descale by the previous rescale factor in settings.py
                test_rmse = sqrt(mean_squared_error(y_test, y_pred))
                test_rmse = test_rmse * state_ppl / rescale

                combo_rmses.append(test_rmse)

            # Find the average RMSE of all states
            avg_rmse = sum(combo_rmses) / len(combo_rmses)
            test_rmses[dset_type].append(avg_rmse)

    return test_rmses


def test_models():
    """TODO DOCUMENTATION"""
    rmses_ARIMAX = test_ARIMAX()
    rmses_LSTM = test_LSTM()

    rmses = {
        'LSTM': rmses_LSTM,
        'ARIMAX': rmses_ARIMAX
    }

    test_dst = os.getcwd() + '/results/test_rmses.pkl'

    with open(test_dst, 'wb') as f:
        pickle.dump(rmses, f)

    return rmses_LSTM, rmses_ARIMAX


def predict_ARIMAX():
    """TODO DOCUMENTATION"""

    # Unpack top 5 hyper-parameter combinations
    with open(ft_dst_ARIMAX, 'rb') as f:
        ft_params = pickle.load(f)

    # Keep track of each state's 5 predictions
    preds = {state: {'cases': {}, 'deaths': {}} for state in states}

    # Create each hyper-parameter combination's corresponding dataframe to be
    # rolled over in each iteration
    dsets = {'cases': {}, 'deaths': {}}

    for dset_type in dsets:

        params = [i[0] for i in ft_params[dset_type]]

        for combo in params:

            num_extra_states, p, d, q = [int(i) for i in combo.split(', ')]
            order = (p, d, q)

            # Generate corresponding dataset
            cases, deaths = generate_dset_ARIMAX(
                num_extra_states=num_extra_states
            )

            # Identify the dataset to be used for prediction
            dset = cases if dset_type == 'cases' else deaths

            dsets[dset_type][(num_extra_states, order)] = dset

    # Each iteration, all five models predict cases and deaths number using
    # forecast method. All five predictions are added and reconstructed to
    # <preds> to be pickled later on, and the corresponding result will be
    # added for the next round until all iterations are reached. For data
    # augmentation, take the average of cases for augmentation, and keep
    # track of this separately
    for i in range(rolling_iters):

        print(f'Rolling prediction iteration no.{i+1}')

        # Keep track of avg cases number for death df updates
        avg_cases = {state: [] for state in states}

        for combo in list(dsets['cases'].keys()):

            # Keep track of the case predictions with one hyper-parameter combo
            pred_cases = {state: -1 for state in states}
            
            # Unpack the hyper-parameter and the corresponding dataset
            num_extra_states, order = combo
            dset = dsets['cases'][combo]

            for state in states:

                # Unpack the state population as the weight of RMSE later on
                state_ppl = population_by_state[state]

                # Initialize the ARIMAX model
                model = ARIMAX(order, state)
                
                # Retrain the model and do the next one-step prediction
                model.fit(dset)
                date, y_pred = model.forecast()
                
                # Log the predicted case number to <avg_cases> and <pred_cases>
                avg_cases[state].append(y_pred)
                pred_cases[state] = y_pred

                # Log the reconstructed pred number to the output dictionary
                y_pred_rescaled = int(y_pred * state_ppl / rescale)
                if i not in preds[state]['cases']:
                    preds[state]['cases'][i] = [y_pred_rescaled]
                else:
                    preds[state]['cases'][i].append(y_pred_rescaled)
            
            # Update each states corresponding dataframe for next iteration
            for state in states:
                
                df_to_update = dset[state]['train']
                
                row_vals = {
                    state: pred_cases[state] for state in df_to_update.columns
                }

                new_row = pd.Series(data=row_vals, name=date)
                
                dset[state]['train'] = df_to_update.append(new_row)

        # Average cases number before predicting deaths
        for state in avg_cases:
            avg_cases[state] = sum(avg_cases[state]) / len(avg_cases[state])

        for combo in list(dsets['deaths'].keys()):

            # Keep track of the deaths predictions with one hp combination
            pred_deaths = {state: -1 for state in states}

            # Unpack the hyper-parameter and the corresponding dataset
            num_extra_states, order = combo
            dset = dsets['deaths'][combo]

            for state in states:

                # Unpack the state population as the weight of RMSE later on
                state_ppl = population_by_state[state]

                # Initialize the ARIMAX model
                model = ARIMAX(order, state)

                # Retrain the model and do the next one-step prediction
                model.fit(dset)
                date, y_pred = model.forecast()

                # Log the predicted case number to <pred_deaths>
                pred_deaths[state] = y_pred

                # Log the reconstructed pred number to the output dictionary
                y_pred_rescaled = int(y_pred * state_ppl / rescale)
                if i not in preds[state]['deaths']:
                    preds[state]['deaths'][i] = [y_pred_rescaled]
                else:
                    preds[state]['deaths'][i].append(y_pred_rescaled)

            # Update each states corresponding dataframe for next iteration
            for state in states:
                df_to_update = dset[state]['train']

                row_vals = {}

                for col in df_to_update.columns:
                    if 'cases' in col:
                        extra_state = col.split('_')[0]
                        row_vals[col] = avg_cases[extra_state]
                    else:
                        row_vals[col] = pred_deaths[col]

                new_row = pd.Series(data=row_vals, name=date)

                dset[state]['train'] = df_to_update.append(new_row)

    # Pickle the results for visualization
    preds_dst = os.getcwd() + '/results/arimax_predictions.pkl'

    with open(preds_dst, 'wb') as f:
        pickle.dump(preds, f)

    return preds


def main():
    """Main driver function"""

    if pred_mode:
        _ = predict_ARIMAX()
    else:
        _ = test_models()


if __name__ == '__main__':
    main()
