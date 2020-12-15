"""
ARIMAX model
Authors: Vincent Yu
Date: 12/05/2020
"""

from math import sqrt

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from data.dataset import generate_dset_ARIMAX


class ARIMAX:
    """ARIMA model w/ exog variables for time series prediction

    Attributes:
        order:      A tuple in the form of (p, d, q) as the model's
                        hyper-parameters. See official documentation for
                        more details.
        state:      A string representing the state name.
        data:       A pandas dataframe representing all the available data to
                        train on. Since ARIMA can't do the machine learning
                        way of train, validate, and test, all data are
                        required for training.
        model:      A fitted ARIMAX model.
        y_pred:     A numpy array of predicted values.
        y_true:     A numpy array of true values.
        rmse:       A float representing the root mean squared error between
                        predictions and true values.
    """

    def __init__(self, order, state):
        """Model initialization"""
        self.order = order
        self.state = state

    def fit(self, dset):
        """Fit the model with all available data"""

        # Find the dataframe for the given state
        self.data = dset[self.state]['train']

        # Find the endogenous and exogenous variables
        exog_cols = [col for col in self.data.columns if col != self.state]

        self.eng, self.exog = self.data[self.state], self.data[exog_cols]

        # Fit the model with given data and order
        self.model = ARIMA(self.eng, exog=self.exog, order=self.order).fit()

    def predict(self, start_idx, end_idx):
        """Predict the time series output given a fitted model"""

        # Predict within a given range of time
        # NOTE: the range is inclusive in both ends, so subtract 1 from the
        # end point if needed
        self.y_pred = self.model.predict(start=start_idx, end=end_idx - 1)

        # Unpack the true values
        self.y_true = np.array(self.data[self.state][start_idx: end_idx])

    def forecast(self):
        """One step prediction"""

        # Get the one step forecast with exog variables input
        one_step_pred = self.model.forecast(exog=self.exog.iloc[[-1]])

        # Find the date for index and the predicted value, both as output
        date = str(one_step_pred.index[0]).split(' ')[0]
        y_pred = one_step_pred[0]

        return date, max(y_pred, 0)

    def evaluate(self):
        """Model performance evaluation"""

        # Find the RMSE between the predictions and the true values
        self.rmse = sqrt(mean_squared_error(self.y_true, self.y_pred))


if __name__ == '__main__':

    # Sanity check
    # Must run the following in a script in the main folder

    cases, deaths = generate_dset_ARIMAX(num_extra_states=4)

    state = 'California'

    order = (10, 2, 2)

    val_start, val_end = deaths[state]['validate']

    model = ARIMAX(order, state)

    model.fit(deaths)

    model.predict(val_start, val_end)

    model.evaluate()

    print(model.rmse)

