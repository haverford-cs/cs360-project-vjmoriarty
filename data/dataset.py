"""
Dataset related functions.
Authors: Vincent Yu
Date: 11/28/2020
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from info import population_by_state, nearest_states
from utils import correct_datetime


def read_dataset(dst):
    """Read the John Hopkins data and convert to desired format

    Args:
        dst:        A string, indicating the location of the csv file.


    Returns:
        dset:       A dictionary that contains the dataset.
                        It has the structure of:
                        {state name --> [daily numbers], 'dates': [dates]}
    """

    # Read and transpose the dataset
    df = pd.read_csv(dst, index_col=0, header=None, low_memory=False).T

    # Rename the state column
    df = df.rename(columns={'Province_State': 'state'})

    # Drop counties
    if 'Admin2' in df.columns:
        df = df.drop(columns=['Admin2'])

    # Identify different columns
    state_col = ['state']
    date_cols = [col for col in df.columns if col not in state_col]

    # Convert data types for entries
    df[[col for col in date_cols]] = df[[col for col in date_cols]].astype(
        str).astype('float').astype(int)

    # Aggregate by state and clean up
    df = df.groupby(state_col, as_index=False).agg('sum').reset_index()
    df = df.drop(columns=['index'])
    df = df.set_index('state')

    # Convert to dictionary with states as keys and list of daily cases as value
    dset = df.to_dict('index')
    dates = [date for date in dset['Alabama'].keys()]
    for state in dset.keys():
        daily_cases = [val for key, val in dset[state].items()]
        dset[state] = daily_cases

    # Keep the dates in the dataset
    dset['dates'] = dates

    return dset


# For ARIMAX
def generate_dset_ARIMAX(rescale=1000, num_extra_states=0, agg_cases=True,
                         train_size=0.7, validation_size=0.2):
    """TODO DOCUMENTATION"""
    
    # Indicate csv destinations
    cases_dst = os.getcwd() + '/data/CONVENIENT_us_confirmed_cases.csv'
    deaths_dst = os.getcwd() + '/data/CONVENIENT_us_deaths.csv'

    # Unpack the raw cases and deaths datasets
    cases_dset = read_dataset(cases_dst)
    deaths_dset = read_dataset(deaths_dst)

    # Find the list of dates with format corrected
    dates = [correct_datetime(date) for date in cases_dset['dates']]

    # Initialize temporary and final data dictionaries for ARIMAX
    cases, deaths = {}, {}
    cases_normalized, deaths_normalized = {}, {}

    # Only selecting the 50 states plus D.C. for normalizqtion
    # Sorry Puerto Rico :(
    for state in population_by_state.keys():

        # Divide each daily number by the state total population with rescaling
        state_ppl = population_by_state[state]
        cases_normalized[state] = [num * rescale / state_ppl for num in
                                   cases_dset[state]]
        deaths_normalized[state] = [num * rescale / state_ppl for num in
                                    deaths_dset[state]]

    # After normalization, build the desired per state
    for state in population_by_state.keys():

        # Initialize the dictionaries
        state_cases = {'dates': dates, state: cases_normalized[state]}
        state_deaths = {'dates': dates, state: deaths_normalized[state]}

        # Use cases as explanatory variables for death predictions
        if agg_cases:
            col_name = state + '_cases'
            state_deaths[col_name] = cases_normalized[state]

        # Adding nearest n states
        if num_extra_states != 0:
            if num_extra_states < 51:
                extra_states = nearest_states[state][:num_extra_states]

                for extra_state in extra_states:
                    state_cases[extra_state] = cases_normalized[extra_state]
                    state_deaths[extra_state] = deaths_normalized[extra_state]

                    if agg_cases:
                        col_name = extra_state + '_cases'
                        state_deaths[col_name] = cases_normalized[extra_state]

            else:
                raise(Exception('Exceeding the max number of additional '
                                'states.'))

        # Build the corresponding dataframes before storage
        cases_df = pd.DataFrame.from_dict(state_cases).set_index('dates')
        deaths_df = pd.DataFrame.from_dict(state_deaths).set_index('dates')

        # Split the dataset into train, validate, and test partitions
        num_samples = len(dates)
        num_train = int(train_size * num_samples)
        num_val = int(validation_size * num_samples)
        end_idx_val = num_train + num_val

        cases[state] = {
            'train': cases_df[: num_train],
            'validate': cases_df[num_train: end_idx_val],
            'test': cases_df[end_idx_val:]
        }

        deaths[state] = {
            'train': deaths_df[: num_train],
            'validate': deaths_df[num_train: end_idx_val],
            'test': deaths_df[end_idx_val:]
        }

    return cases, deaths


# For LSTM
def convert_to_tensor(features, labels, batch_size=10):
    """Generate dataset with mini batches

    Args:
        features:       A numpy array representing the features for the dataset.
        labels:         A numpy array representing the labels for the dataset.
        batch_size:     An integer indicating the size of each mini batch.
                            Default = 10

    Returns:
        dset:           The desired dataset with mini batches.
    """

    dset = tf.data.Dataset.from_tensor_slices((features, labels))

    dset = dset.batch(batch_size)

    return dset


def convert_to_time_series(dset, lag, rescale=1000):
    """Convert the number of daily cases/deaths to a time series dataset

    Args:
        dset:           A dictionary representing the original daily number
                            by state dataset. It should have the format of
                            {state: [daily numbers], 'dates': [dates]}
        lag:            An integer representing the time lag used to
                            construct time series features.
        rescale:        An integer (or float) to rescale the normalized daily
                            number.
                            Default = 1000

    Returns:
        time_dset:      A dictionary representing the time series version of
                            the original dataset. It has the format of
                            {state: [features, labels, unknown]}
    """

    # Initialize the time series dataset
    time_dset = {state: [] for state in population_by_state.keys()}

    # Find the number of days in total
    total_days = len(dset['dates'])

    for state in time_dset.keys():

        # Divide each daily number by the state total population
        state_ppl = population_by_state[state]
        daily_num = [num * rescale / state_ppl for num in dset[state]]

        # Capture fragments of length <lag> as feature values
        ft_vals = [daily_num[i: i+lag] for i in range(total_days - lag)]
        labels = [daily_num[i] for i in range(lag, total_days)]

        # Convert the lists into numpy array before storage
        X, y, Z = np.array(ft_vals), np.array(labels), np.array(daily_num)
        time_dset[state] = [X, y, Z]

    return time_dset


def augment_dset(dset, offset=0, num_extra_states=0, extra_cases=None,
                 aug_offset=0):
    """TODO DOCUMENTATION"""
    
    # Make sure cases offset is not larger than death offset to prevent index
    # out of range issue
    offset_diff = offset - aug_offset
    if extra_cases and offset_diff < 0:
        raise(Exception('Case offset exceeding death offset.'))

    # Make sure lag and offset are not too large to not have a proper dataset
    lag = dset['Alabama'][0][0].shape[0]
    num_samples = dset['Alabama'][1].shape[0]
    num_days = dset['Alabama'][2].shape[0]

    if num_samples <= offset or num_days < lag + offset + 3:
        raise (Exception('Dataset too small. Please decrease the lag or the '
                         'offset.'))
    
    # Initialize aggregated dataset
    aug_dset = {state: [] for state in dset.keys()}
    
    for state in dset.keys():

        # Find the nearest n states
        extra_states = nearest_states[state][:num_extra_states]

        # Unpack features, labels, and unlabeled fragments
        X, y, Z = dset[state]

        # Reconstruct features with offset and augmentation with case numbers
        aug_X = []

        end_idx = X.shape[0] - offset

        for i, frag in enumerate(X[: end_idx]):
            aug_frag = []

            # With augmentation
            if extra_cases:
                # Find the case number fragment with the augmentation offset
                case_loc = i + offset_diff
                cases_frag = extra_cases[state][0][case_loc]
                # Rebuild fragment into a 2-channel (3D) fragment
                cubical = [[i, c_i] for i, c_i in list(zip(frag, cases_frag))]
                aug_frag.append(cubical)
            else:
                # Copy and transform the original fragment into 1-channel
                aug_frag.append([[i] for i in frag])

            # Add nearest states' numbers to each fragment as well
            for extra_state in extra_states:
                # Find the corresponding fragment
                extra_X_frag = dset[extra_state][0][i]

                if extra_cases:
                    # Append the death and case fragments to the
                    # corresponding part of the augmented fragment
                    case_loc = i + offset_diff
                    extra_X_frag_cases = extra_cases[extra_state][0][case_loc]
                    add_cubical = [
                        [i, c_i] for i, c_i in list(zip(extra_X_frag,
                                                        extra_X_frag_cases))
                    ]
                    aug_frag.append(add_cubical)
                else:
                    # Copy and stack below the existing data
                    aug_frag.append([[i] for i in extra_X_frag])
            
            aug_X.append(aug_frag)

        # Reshape X to fit the LSTM input dimensions
        aug_X = np.array(aug_X)
        num_samples, num_states, lag, channels = aug_X.shape
        aug_X = aug_X.reshape(num_samples, 1, num_states, lag, channels)

        aug_dset[state].append(aug_X)

        # Store the labels considering prediction offset
        aug_dset[state].append(np.array(y[offset:]))

        # For unlabeled fragments
        aug_Z = []

        # Find the earliest unused data's indices
        start_idx = Z.shape[0] - offset - lag
        cases_start_idx = Z.shape[0] - aug_offset - lag

        # With augmentation
        if extra_cases:
            # Rebuild into a 2-channel (3D) array
            cases_Z = extra_cases[state][2][cases_start_idx:]
            cubical = [
                [i, c_i] for i, c_i in list(zip(Z[start_idx:],
                                                cases_Z))
            ]
            aug_Z.append(cubical)
        else:
            # Copy the unused data with 1-channel configuration
            aug_Z.append([[i] for i in Z[start_idx:]])

        # Augment the data with other states' numbers
        for extra_state in extra_states:
            extra_Z = dset[extra_state][2][start_idx:]
            if extra_cases:
                # Append the death and case numbers to the
                # corresponding part of the augmented array
                extra_Z_cases = extra_cases[extra_state][2][cases_start_idx:]
                add_cubical = [
                    [i, c_i] for i, c_i in list(zip(extra_Z, extra_Z_cases))
                ]
                aug_Z.append(add_cubical)
            else:
                # Stack below the existing unused samples
                aug_Z.append([[i] for i in extra_Z])

        aug_dset[state].append(np.array(aug_Z))

    return aug_dset


def split_dset(dset, train_size=0.7, validation_size=0.2):
    """Split dataset into train/validate/test partitions per state.

    Args:
        dset:               A dictionary representing the original dataset.
                                It should have the format of:
                                {state_name: [X, y]}
        train_size:         A float representing the percentage of samples
                                used for training.
                                Default = 0.7
        validation_size:    A float representing the percentage of samples
                                used for validation.
                                Default = 0.2

    Returns:
        split:              A dictionary representing the sliced dataset.
                                It has the format of:
                                {state: {'train'/'validate'/'test': [X, y]}}
    """

    split = {state: {} for state in dset.keys()}

    for state in dset.keys():
        # Unpack features, corresponding labels, and unobserved samples
        X, y, Z = dset[state]

        # Find the indices to slice the dataset
        num_samples = X.shape[0]
        num_train = int(train_size * num_samples)
        num_val = int(validation_size * num_samples)
        end_idx_val = num_train + num_val

        # Create train, validate, test partitions
        train_X, train_y = X[:num_train], y[:num_train]
        val_X, val_y = X[num_train: end_idx_val], y[num_train: end_idx_val]
        test_X, test_y = X[end_idx_val:], y[end_idx_val:]

        # Update the output dataset dictionary
        split[state] = {
            'train': convert_to_tensor(train_X, train_y),
            'validate': convert_to_tensor(val_X, val_y),
            'test': convert_to_tensor(test_X, test_y),
            'unused': Z
        }
    
    return split


def generate_dset_LSTM(lag, num_extra_states=0, case_offset=0, death_offset=0,
                       aug_offset=0):
    """High level dataset generation function."""

    # Indicate csv destinations
    cases_dst = os.getcwd() + '/data/CONVENIENT_us_confirmed_cases.csv'
    deaths_dst = os.getcwd() + '/data/CONVENIENT_us_deaths.csv'

    # Unpack the raw cases and deaths datasets
    cases_dset = read_dataset(cases_dst)
    deaths_dset = read_dataset(deaths_dst)

    time_cases = convert_to_time_series(cases_dset, lag)
    time_deaths = convert_to_time_series(deaths_dset, lag)

    cases = split_dset(augment_dset(
        time_cases,
        offset=case_offset,
        num_extra_states=num_extra_states
    ))

    deaths = split_dset(augment_dset(
        time_deaths,
        offset=death_offset,
        num_extra_states=num_extra_states,
        extra_cases=time_cases,
        aug_offset=aug_offset
    ))

    return cases, deaths

# TODO
#  1) Pickle the dataset
#  2) PREDICTION USE DATASET FUNCTION: REMEMBER TO RESHAPE EACH Z


if __name__ == '__main__':

    # cases, deaths = generate_dset_ARIMAX(num_extra_states=5)

    cases_lstm, deaths_lstm = generate_dset_LSTM(7, 5, 4, 4)

    for ft, val in cases_lstm['Alabama']['test'].take(1):
        print(tf.reshape(ft, [10, 1, 6, 7, 1]).shape)
        print()

    for ft, val in deaths_lstm['Alabama']['validate'].take(1):
        print(ft.shape)
        print(val.shape)

    # Output should be (with batch_size = 10, extra_states = 5, lag = 7):
    # (10, 1, 6, 7, 1)
    # (10,)
    #
    # (10, 1, 6, 7, 2)
    # (10,)


