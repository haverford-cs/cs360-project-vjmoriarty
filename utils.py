"""
Helper functions.
Authors: Vincent Yu
Date: 12/03/2020
"""


def correct_datetime(date):
    """TODO DOCUMENTATION"""
    time_components = date.split('/')
    correct_date = ''
    for comp in time_components:
        if len(comp) != 2:
            correct_date += ('0' + comp + '/')
        else:
            correct_date += (comp + '/')

    return correct_date[:-1]
