"""
Helper functions.
Authors: Vincent Yu
Date: 12/03/2020
"""

import itertools


def correct_datetime(date):
    """Correct the data-time format from the original csv"""
    time_components = date.split('/')
    correct_date = ''
    for comp in time_components:
        if len(comp) != 2:
            correct_date += ('0' + comp + '/')
        else:
            correct_date += (comp + '/')

    return correct_date[:-1]


def get_combos(ft_params):
    """Generate all possible combinations given a grid of parameters"""
    combinations = []
    keys, values = zip(*ft_params.items())
    for bundle in itertools.product(*values):
        combinations.append(dict(zip(keys, bundle)))
    return combinations
