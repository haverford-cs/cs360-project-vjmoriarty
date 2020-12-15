"""
Conv-LSTM model
Authors: Vincent Yu
Date: 12/01/2020
"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, ConvLSTM2D, Dropout
from tensorflow.keras import losses, metrics


class LSTM(Model):
    """ConvLSTM model with tentative structure"""

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm1 = ConvLSTM2D(
            32, (2, 3), activation='relu', return_sequences=True
        )

        self.drop1 = Dropout(0.2)

        self.lstm2 = ConvLSTM2D(
            64, (1, 3), activation='relu'
        )

        self.f = Flatten()
        self.d1 = Dense(100)
        self.drop2 = Dropout(0.2)
        self.d2 = Dense(1)

    def call(self, inputs):
        x1 = self.lstm1(inputs)
        dp1 = self.drop1(x1)

        x2 = self.lstm2(dp1)

        f = self.f(x2)
        d1 = self.d1(f)
        dp2 = self.drop2(d1)
        output = self.d2(dp2)

        return output


def run_lstm(train_dset, val_dset, epochs=40, verbose=0):

    # Declare model
    model = LSTM()

    # Model configuration
    model.compile(
        optimizer='adam',
        loss=losses.MeanSquaredError(),
        metrics=[
            metrics.RootMeanSquaredError(name='rmse'),
            metrics.MeanSquaredLogarithmicError(name='msle')
                 ]
    )

    # Train and validate the model with designated datasets
    history = model.fit(
        train_dset,
        validation_data=val_dset,
        epochs=epochs,
        verbose=verbose
    )

    # Unpack losses and other metrics
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    train_rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']

    train_msle = history.history['msle']
    val_msle = history.history['val_msle']

    if verbose != 0:
        # Print each epoch's performance to terminal
        template = 'Epoch {}, Loss: {}, RMSE: {}, MSLE: {}, Val Loss: {}, ' \
                   'Val RMSE: {}, Val MSLE: {}'

        for i in range(epochs):
            print(template.format(i + 1,
                                  train_losses[i],
                                  train_rmse[i] * 100,
                                  train_msle[i] * 100,
                                  val_losses[i],
                                  val_rmse[i] * 100,
                                  val_msle[i] * 100,
                                  )
                  )

    return history.history, model


def lstm_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    model = LSTM()

    # Try out both the options below (all zeros and random)
    # shape is: number of examples (mini-batch size), width, height, depth
    x_np = np.random.rand(1, 1, 5, 8, 2)
    y_np = np.random.rand(1)

    # call the model on this input and print the result
    output = model.call(x_np)
    print(output)

    # Check the output of each layer (may not work)
    model.compile(
        optimizer='adam',
        loss=losses.MeanSquaredError(),
        metrics=[
            metrics.RootMeanSquaredError(name='rmse'),
            metrics.MeanSquaredLogarithmicError(name='msle')
        ]
    )
    model.fit(x_np, y_np, epochs=1)
    print()
    print(model.summary())
    print()

    # Look at the model parameter shapes
    for v in model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)


if __name__ == '__main__':
    lstm_test()

