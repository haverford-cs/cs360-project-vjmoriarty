"""
ARIMAX model
Authors: Vincent Yu
Date: 12/05/2020
"""

from statsmodels.tsa.arima_model import ARIMA

from dataset import generate_dset_ARIMAX


# TODO BUILD FIT AND PREDICT IN TWO FUNCTION
def run_arimax(train_eng, train_exog):

    model = ARIMA(train_eng, exog=train_exog, order=(1, 0, 1)).fit(disp=0)

    return model.summary()


if __name__ == '__main__':

    cases, deaths = generate_dset_ARIMAX(num_extra_states=4)

    state = 'California'

    state_dset = deaths[state]['train']

    exog_cols = [col for col in state_dset.columns if col != state]

    eng, exog = state_dset[state], state_dset[exog_cols]

    summary = run_arimax(eng, exog)

    print(summary)

