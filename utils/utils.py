"""
Helper functions.
Authors: Vincent Yu
Date: 12/03/2020
"""

from copy import copy
import itertools
import datetime

import pandas as pd
import plotly.graph_objects as go


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


def add_days(start, num_days=10):
    """Find out the dates of a given number of days in the future """
    current_date = datetime.datetime.strptime(start, "%m/%d/%y")

    future_dates = []

    for i in range(num_days):
        delta = current_date + datetime.timedelta(days=i + 1)
        next_date = delta.strftime("%m/%d/%y")
        future_dates.append(next_date)

    return future_dates


def pred_to_df(preds, prev_dates, prev_nums, dset_type='cases'):
    """Convert prediction results to dataframe for plotting"""

    dfs = {}

    # Find all the dates from the start data point to the end of prediction.
    rolling_iters = len(list(preds['Alabama'][dset_type].keys()))
    last_day = prev_dates[-1]
    future_dates = add_days(last_day, num_days=rolling_iters)
    all_dates = prev_dates + future_dates

    for state in prev_nums:

        # Format the input dictionary for dataframe conversion
        state_data = {
            'dates': all_dates,
            state: copy(prev_nums[state]),
            'min': copy(prev_nums[state]),
            'max': copy(prev_nums[state])
        }

        # Add the appropriate prediction result to the dictionary
        for i in range(rolling_iters):
            state_preds = preds[state][dset_type][i]

            avg_pred = int(sum(state_preds) / len(state_preds))
            state_data[state].append(avg_pred)

            min_pred = min(state_preds)
            state_data['min'].append(min_pred)

            max_pred = max(state_preds)
            state_data['max'].append(max_pred)

        # Create and store the correspoding dataframe
        state_df = pd.DataFrame(data=state_data)
        dfs[state] = state_df

    return dfs


def plot_preds(dfs, dset_type='Case'):
    """Plot all prediction results by state with dropdown option"""

    # Find all states available
    states = list(dfs.keys())

    # Create a plot template for each state
    rand_state = states[0]
    rand_df = dfs[rand_state]

    fig = go.Figure()

    # Add average prediction scatter/line plot
    fig.add_traces(go.Scatter(
        name='Average Prediction',
        x=rand_df['dates'],
        y=rand_df[rand_state],
        mode='lines+markers',
        showlegend=False
    ))

    # Add min and max as moving range
    fig.add_traces(go.Scatter(
        name='Upper Bound',
        x=rand_df['dates'],
        y=rand_df['max'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_traces(
        go.Scatter(
            name='Lower Bound',
            x=rand_df['dates'],
            y=rand_df['min'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    )

    # Construct menus for dropdown
    buttons = []
    for state in states:
        state_df = dfs[state]
        button = {
            'method': 'update',
            'label': state,
            'args': [
                {'y': [state_df[state], state_df['max'], state_df['min']]}
            ]
        }

        buttons.append(button)

    update_menus = [{
        'buttons': buttons,
        'direction': "down",
        'pad': {"r": 10, "t": 10},
        'showactive': True,
        'x': 0.1,
        'xanchor': "left",
        'y': 1.14,
        'yanchor': "top"
    }]

    # update layout with buttons, titles, and axis titles
    fig.update_layout(
        updatemenus=update_menus,
        annotations=[
            dict(text="State",
                 showarrow=False,
                 x=0,
                 y=1.085,
                 yref="paper",
                 align="left")
        ],
        title=f'Predicted {dset_type} Number',
        xaxis_title='Dates',
        yaxis_title=f'Number of {dset_type}'
    )

    fig.show()
