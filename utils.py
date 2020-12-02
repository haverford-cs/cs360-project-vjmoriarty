"""
Dataset related functions.
Authors: Vincent Yu
Date: 11/28/2020
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Population by State
# According to 2010 Census Projection
population_by_state = {
    'Alabama': 4903185,
    'Alaska': 731545,
    'Arizona': 7278717,
    'Arkansas': 3017804,
    'California': 39512223,
    'Colorado': 5758736,
    'Connecticut': 3565287,
    'Delaware': 973764,
    'District of Columbia': 705749,
    'Florida': 21477737,
    'Georgia': 10617423,
    'Hawaii': 1415872,
    'Idaho': 1787065,
    'Illinois': 12671821,
    'Indiana': 6732219,
    'Iowa': 3155070,
    'Kansas': 2913314,
    'Kentucky': 4467673,
    'Louisiana': 4648794,
    'Maine': 1344212,
    'Maryland': 6045680,
    'Massachusetts': 6892503,
    'Michigan': 9986857,
    'Minnesota': 5639632,
    'Mississippi': 2976149,
    'Missouri': 6137428,
    'Montana': 1068778,
    'Nebraska': 1934408,
    'Nevada': 3080156,
    'New Hampshire': 1359711,
    'New Jersey': 8882190,
    'New Mexico': 2096829,
    'New York': 19453561,
    'North Carolina': 10488084,
    'North Dakota': 762062,
    'Ohio': 11689100,
    'Oklahoma': 3956971,
    'Oregon': 4217737,
    'Pennsylvania': 12801989,
    'Rhode Island': 1059361,
    'South Carolina': 5148714,
    'South Dakota': 884659,
    'Tennessee': 6829174,
    'Texas': 28995881,
    'Utah': 3205958,
    'Vermont': 623989,
    'Virginia': 8535519,
    'Washington': 7614893,
    'West Virginia': 1792147,
    'Wisconsin': 5822434,
    'Wyoming': 578759,
    'Puerto Rico': 3193694,
}


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


def convert_to_tensor(features, labels=None, batch_size=10):
    """Generate dataset with mini batches

    Args:
        features:       A numpy array representing the features for the dataset.
        labels:         A numpy array representing the labels for the dataset.
                            Default = None
        batch_size:     An integer indicating the size of each mini batch.
                            Default = 10

    Returns:
        dset:           The desired dataset with mini batches. Each batch is in
                            the form of: (features, labels).
    """

    data = (features, labels) if labels is not None else features

    dset = tf.data.Dataset.from_tensor_slices(data)

    dset = dset.batch(batch_size)

    return dset


def convert_to_time_series(dset, interval, offset=0, normalize=True, rescale=1):
    """Convert the number of daily cases/deaths to a time series dataset

    Args:
        dset:           A dictionary representing the original daily number
                            by state dataset. It should have the format of
                            {state: [daily numbers], 'dates': [dates]}
        interval:       An integer representing the time interval used to
                            construct time series features.
        offset:         An integer representing the label time offset from the
                            last point of time in features.
                            Default = 0
        normalize:      A boolean to indicate whether to normalize each state's
                            daily number by its state population or not.
                            Default = True
        rescale:        An integer (or float) to rescale the normalized daily
                            number.
                            Default = 1

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

        if normalize:
            # Divide each daily number by the state total population
            state_ppl = population_by_state[state]
            daily_num = [num * rescale / state_ppl for num in dset[state]]

        else:
            daily_num = [num for num in dset[state]]

        # Capture fragments of length <interval> as feature values, and the
        # offset daily number as its label
        ft_vals = [daily_num[i: i + interval] for i in range(total_days -
                                                             interval - offset)]
        labels = [daily_num[i] for i in range(interval + offset, total_days)]

        # Make the remaining fragments feature values for prediction
        unknown = [daily_num[i: i + interval] for i in range(
            total_days - interval - offset, total_days - interval + 1
        )]

        # Convert the lists into numpy array before storage
        X, y, Z = np.array(ft_vals), np.array(labels), np.array(unknown)
        time_dset[state] = [X, y, Z]

    return time_dset


def split_dset(dset, train_size=0.7, validation_size=0.2, random=True,
               normalize=False, for_tf=False):
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
        random:             A boolean to indicate to randomly shuffle the
                                samples or not.
                                Default = True
        normalize:          A boolean to indicate to normalize the features
                                or not.
                                Default = False
        for_tf:             A boolean to indicate whether the dataset will be
                                used by tensorflow models or not.
                                Default = False

    Returns:
        split:              A dictionary representing the sliced dataset.
                                It has the format of:
                                {state: {'train'/'validate'/'test': [X, y]}}
    """

    split = {state: {} for state in dset.keys()}

    for state in dset.keys():

        # Unpack the number of samples, features, corresponding labels,
        # and unobserved samples
        num_samples = len(dset[state][0])
        X, y, Z = dset[state][0], dset[state][1], dset[state][2]

        if random:
            # Perform a random shuffle of the observed samples
            lst_of_idx = np.random.choice([i for i in range(num_samples)],
                                          size=num_samples, replace=False)

            X, y = dset[state][0][lst_of_idx], dset[state][1][lst_of_idx]

        # Find the indices to slice the dataset
        num_train = int(train_size * num_samples)
        num_val = int(validation_size * num_samples)
        end_idx_val = num_train + num_val

        # Create train, validate, test partitions
        train_X, train_y = X[:num_train], y[:num_train]
        val_X, val_y = X[num_train: end_idx_val], y[num_train: end_idx_val]
        test_X, test_y = X[end_idx_val:], y[end_idx_val:]

        if normalize:
            # Normalize with the mean and standard deviation of the training
            # features if necessary
            train_X_mean = np.mean(train_X)
            train_X_std = np.std(train_X)

            train_X = (train_X - train_X_mean) / train_X_std
            val_X = (val_X - train_X_mean) / train_X_std
            test_X = (test_X - train_X_mean) / train_X_std
            Z = (Z - train_X_mean) / train_X_std

        # Update the output dataset dictionary with model specification
        if for_tf:
            split[state] = {
                'train': convert_to_tensor(train_X, train_y),
                'validate': convert_to_tensor(val_X, val_y),
                'test': convert_to_tensor(test_X, test_y),
                'unused': convert_to_tensor(Z)
            }
        else:
            split[state] = {
                'train': [train_X, train_y],
                'validate': [val_X, val_y],
                'test': [test_X, test_y],
                'unused': [Z]
            }

    return split


def generate_dset(dst, interval, offset=0, tensor=False):
    """High level dataset generation function."""

    dset = read_dataset(dst)

    time_dset = convert_to_time_series(dset, interval, offset, rescale=1000)

    split = split_dset(time_dset, for_tf=tensor)

    return split


if __name__ == '__main__':

    # Load and generate two datasets
    case_dst = os.getcwd() + '/data/CONVENIENT_us_confirmed_cases.csv'
    tf_dset = generate_dset(case_dst, 5, tensor=True)
    dset = generate_dset(case_dst, 5)

    # Sanity checks
    print(dset['Alabama']['train'][0].shape)    # (210, 5)

    for ft, val in tf_dset['Alabama']['train'].take(1):
        print(ft.shape)     # (10, 5)

    # This should be:
    # (TensorSpec(shape=(None, 5), dtype=tf.float64, name=None),
    #  TensorSpec(shape=(None,), dtype=tf.float64, name=None))
    print(tf_dset['Alabama']['train'].element_spec)

