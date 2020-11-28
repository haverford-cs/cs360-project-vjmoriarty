"""
Helper functions.
Authors: Vincent Yu
Date: 11/28/2020
"""

import pandas as pd


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

