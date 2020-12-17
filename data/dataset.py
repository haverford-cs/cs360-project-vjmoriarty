"""
Dataset related functions.
Authors: Vincent Yu
Date: 11/28/2020
"""

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from data.info import population_by_state, nearest_states
from models.settings import update_dset
from utils.utils import correct_datetime

# COVID data csv paths
csv_paths = {
    'cases': os.getcwd() + '/data/CONVENIENT_us_confirmed_cases.csv',
    'deaths': os.getcwd() + '/data/CONVENIENT_us_deaths.csv'
}


def read_dataset(dset_name='cases', cutoff='03/01/20', update=False):
    """Read the John Hopkins data and convert to desired format

    Args:
        dset_name:  A string representing whether this is the cases dset or
                        the deaths number dset.
                        Default = 'cases'
        cutoff:     A string representing the first date of the dataset,
                        cutting off all prior numbers.
                        Default = '03/01/20'
        update:     A boolean indicating whether the raw pkl files should be
                        updated or not.

    Returns:
        dset:       A dictionary that contains the dataset.
                        It has the structure of:
                        {state name --> [daily numbers], 'dates': [dates]}
    """

    # Indicate pickled dataset's path
    dset_dst = os.getcwd() + '/data/{}_raw.pkl'.format(dset_name)

    if os.path.exists(dset_dst) and not update:
        # Unpickle the existing pickle dataset
        with open(dset_dst, 'rb') as f:
            dset = pickle.load(f)
    else:
        # Read and transpose the dataset
        dst = csv_paths[dset_name]
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

        # Convert to dictionary with states as keys and list of daily cases
        # as values without the data prior to the cutoff date
        dset = df.to_dict('index')
        dates = [correct_datetime(date) for date in dset['Alabama'].keys()]
        cutoff_idx = dates.index(cutoff)

        for state in dset.keys():
            daily_num = [val for key, val in dset[state].items()]
            dset[state] = daily_num[cutoff_idx:]

        # Keep the dates in the dataset
        dset['dates'] = dates[cutoff_idx:]

        # Pickle the dataset dictionary for later manipulation
        with open(dset_dst, 'wb') as f:
            pickle.dump(dset, f)

    return dset


# For ARIMAX
def generate_dset_ARIMAX(rescale=1000, num_extra_states=0, aug_cases=True,
                         train_size=0.7, validation_size=0.2):
    """Generate ARIMAX cases and deaths dataframes for each state

    Args:
        rescale:            An integer representing the rescaling factor to
                                inflate the normalized daily numbers.
                                Default = 1000
        num_extra_states:   An integer representing the number of extra
                                states added for data augmentation.
                                Default = 0
        aug_cases:          A boolean representing if the death data is
                                augmented with cases numbers as well.
                                Default = True
        train_size:         A float representing the percentage of data used
                                for training.
                                Default = 0.7
        validation_size:    A float representing the percentage of data used
                                for validation.
                                Default = 0.2

    Returns:
        cases:              A dictionary representing the cases dataset for
                                ARIMAX, with the format of:
                                {state --> 'train'/'validate'/'test' --> df }
        deaths:             A dictionary representing the deaths dataset for
                                ARIMAX, with the format of:
                                {state --> 'train'/'validate'/'test' --> df }
    """

    # Unpack the raw cases and deaths datasets
    cases_dset = read_dataset(dset_name='cases', update=update_dset)
    deaths_dset = read_dataset(dset_name='deaths', update=update_dset)

    # Find the list of dates with format corrected
    dates = cases_dset['dates']

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
        if aug_cases:
            col_name = state + '_cases'
            state_deaths[col_name] = cases_normalized[state]

        # Adding nearest n states
        if num_extra_states != 0:
            if num_extra_states < 51:
                extra_states = nearest_states[state][:num_extra_states]

                for extra_state in extra_states:
                    state_cases[extra_state] = cases_normalized[extra_state]
                    state_deaths[extra_state] = deaths_normalized[extra_state]

                    if aug_cases:
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

        # Given that ARIMAX can't do separate dataframe input, the model will
        # be "trained" on all datapoints, but only validate and test on the
        # corresponding partitions
        cases[state] = {
            'train': cases_df,
            'validate': [num_train, end_idx_val],
            'test': [end_idx_val, num_samples]
        }

        deaths[state] = {
            'train': deaths_df,
            'validate': [num_train, end_idx_val],
            'test': [end_idx_val, num_samples]
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


def convert_to_time_series(dset, order, rescale=1000):
    """Convert the number of daily cases/deaths to a time series dataset

    Args:
        dset:           A dictionary representing the original daily number
                            by state dataset. It should have the format of
                            {state: [daily numbers], 'dates': [dates]}
        order:          An integer representing the time order used to
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

        # Capture fragments of length <order> as feature values
        ft_vals = [daily_num[i: i+order] for i in range(total_days - order)]
        labels = [daily_num[i] for i in range(order, total_days)]

        # Convert the lists into numpy array before storage
        X, y = np.array(ft_vals), np.array(labels)
        time_dset[state] = [X, y]

    return time_dset


def augment_dset(dset, lag=0, num_extra_states=0,
                 aug_cases=True, extra_cases=None, aug_lag=0):
    """Augment the dataset with lag and additional state numbers

    Args:
        dset:               A dictionary representing the original time
                                series dataset. It has the format of:
                                {state --> [X, y]}
        lag:                An integer representing the lag number of days
                                from the output number.
                                Default = 0
        num_extra_states:   An integer representing the number of extra
                                states added for augmentation.
                                Default = 0
        aug_cases:          An boolean representing whether we are adding
                                cases numbers for death augmentation or not.
                                Default = True
        extra_cases:        A dictionary representing the cases dataset with
                                the same order as <dset>. It has the format of:
                                {state --> [X, y]}
        aug_lag:            An integer representing the lag number of days
                                from the output for added case numbers.
                                Default = 0

    Returns:
        aug_dset:           A dictionary representing the augmented dataset.
                                It has the format of:
                                {state_name: [X, y]}
    """

    # Make sure order and lag are not too large to not have a proper dataset
    random_state = list(dset.keys())[0]
    num_samples = dset[random_state][1].shape[0]

    # Find out the maximum number of samples the dataset can have
    max_samples = num_samples - max(lag, aug_lag)

    # Calculate where to truncate the dataset(s)
    end_idx = num_samples - lag
    start_idx = end_idx - max_samples

    if aug_cases:
        aug_end_idx = num_samples - aug_lag
        aug_start_idx = aug_end_idx - max_samples

    # Initialize aggregated dataset
    aug_dset = {state: [] for state in dset.keys()}
    
    for state in dset.keys():

        # Find the nearest n states
        extra_states = nearest_states[state][:num_extra_states]

        # Unpack features and labels
        X, y = dset[state]

        # Reconstruct features with lag and augmentation with case numbers
        aug_X = []

        # Truncate dataset(s)
        X = X[start_idx: end_idx]

        if aug_cases and extra_cases is not None:
            cases_X = extra_cases[state][0][aug_start_idx: aug_end_idx]

        for i, frag in enumerate(X):
            aug_frag = []

            if aug_cases:
                # Find the case number fragment with the augmentation lag
                cases_frag = cases_X[i]

                # Rebuild fragment into a 2-channel (3D) fragment
                cubical = [[i, c_i] for i, c_i in list(zip(frag, cases_frag))]
                aug_frag.append(cubical)
            else:
                # Copy and transform the original fragment into 1-channel
                aug_frag.append([[i] for i in frag])

            aug_X.append(aug_frag)

        # Add nearest states' numbers to each fragment as well
        for extra_state in extra_states:

            # Unpack features of the extra state
            extra_X = dset[state][0]

            # Truncate dataset(s)
            extra_X = extra_X[start_idx: end_idx]

            if aug_cases and extra_cases is not None:
                extra_cases_X = extra_cases[extra_state][0][aug_start_idx:
                                                            aug_end_idx]

            for i, frag in enumerate(extra_X):

                # Find the fragment to augment
                aug_frag = aug_X[i]

                if aug_cases:
                    # Find the case number fragment with the augmentation lag
                    cases_frag = extra_cases_X[i]

                    # Rebuild fragment into a 2-channel (3D) fragment
                    cubical = [[i, c_i] for i, c_i in
                               list(zip(frag, cases_frag))]

                    aug_frag.append(cubical)
                else:
                    # Copy and transform the original fragment into 1-channel
                    aug_frag.append([[i] for i in frag])

        # Reshape X to fit the LSTM input dimensions
        aug_X = np.array(aug_X)
        num_frags, num_states, order, channels = aug_X.shape
        aug_X = aug_X.reshape(num_frags, 1, num_states, order, channels)

        aug_dset[state].append(aug_X)

        # Store the labels considering prediction lag
        start_idx_y = num_samples - max_samples
        aug_dset[state].append(np.array(y[start_idx_y:]))

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
        X, y = dset[state]

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
            'test': [test_X, test_y]
        }
    
    return split


def generate_dset_LSTM(order, num_extra_states=0, cases_lag=0,
                       deaths_lag=0,aug_lag=0):
    """High level dataset generation function.

    Args:
        order:              An integer representing the number of prior days
                                used for prediction with no lag.
        num_extra_states:   An integer representing the number of extra
                                states added for data augmentation.
                                Default = 0
        cases_lag:          An integer representing the number of days
                                between the last day of order and the output
                                date for case numbers.
                                Default = 0
        deaths_lag:         An integer representing the number of days
                                between the last day of order and the output
                                date for deaths numbers.
                                Default = 0
        aug_lag:            An integer representing the number of days
                                between the last day of order and the output
                                date for added case numbers.
                                Default = 0

    Returns:
        cases:              A dictionary representing the sliced cases dataset.
                                It has the format of:
                                {state -> 'train'/'validate'/'test' -> [X, y]}}
        deaths:             A dictionary representing the sliced deaths dataset.
                                It has the format of:
                                {state -> 'train'/'validate'/'test' -> [X, y]}}
    """

    # Unpack the raw cases and deaths datasets
    cases_dset = read_dataset(dset_name='cases', update=update_dset)
    deaths_dset = read_dataset(dset_name='deaths', update=update_dset)

    # Convert both raw dataset to time series dataset with no augmentation
    time_cases = convert_to_time_series(cases_dset, order)
    time_deaths = convert_to_time_series(deaths_dset, order)

    # Augment and split the datasets
    cases = split_dset(augment_dset(
        time_cases,
        lag=cases_lag,
        num_extra_states=num_extra_states,
        aug_cases=False
    ))

    deaths = split_dset(augment_dset(
        time_deaths,
        lag=deaths_lag,
        num_extra_states=num_extra_states,
        extra_cases=time_cases,
        aug_lag=aug_lag
    ))

    return cases, deaths


if __name__ == '__main__':

    # Sanity check
    # cases, deaths = generate_dset_ARIMAX(num_extra_states=5)

    cases_lstm, deaths_lstm = generate_dset_LSTM(7, 5, 4, 4)

    for ft, val in cases_lstm['Alabama']['test'].take(1):
        print(tf.reshape(ft, [10, 1, 6, 7, 1]).shape)
        print()

    for ft, val in deaths_lstm['Alabama']['validate'].take(1):
        print(ft.shape)
        print(val.shape)

    # Output should be (with batch_size = 10, extra_states = 5, order = 7):
    # (10, 1, 6, 7, 1)
    # (10,)
    #
    # (10, 1, 6, 7, 2)
    # (10,)


