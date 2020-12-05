"""
Prediction machine learning models
Authors: Vincent Yu
Date: 12/01/2020
"""

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, ConvLSTM2D, Dropout, \
    MaxPooling3D
from tensorflow.keras import losses
from tensorflow.keras import metrics

from dataset import generate_dset_LSTM


class LSTM(Model):

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm1 = ConvLSTM2D(
            32, (2, 3), activation='relu', return_sequences=True
        )
        self.lstm2 = ConvLSTM2D(
            32, (2, 3), activation='relu', return_sequences=True
        )

        self.p1 = MaxPooling3D(pool_size=(1, 2, 2))
        self.drop1 = Dropout(0.2)

        self.f = Flatten()
        self.d1 = Dense(10)
        self.drop2 = Dropout(0.2)
        self.d2 = Dense(1)

    def call(self, inputs):
        x1 = self.lstm1(inputs)
        x2 = self.lstm2(x1)
        p1 = self.p1(x2)
        dp1 = self.drop1(p1)

        f = self.f(dp1)
        d1 = self.d1(f)
        dp2 = self.drop2(d1)
        output = self.d2(dp2)

        return output


def run_lstm(train_dset, val_dset, epochs=50, verbose=False):

    # Declare model
    model = LSTM()

    # Model configuration
    model.compile(
        optimizer='adam',
        loss=losses.MeanSquaredLogarithmicError(),
        metrics=[
            metrics.MeanSquaredLogarithmicError(name='msle'),
            metrics.KLDivergence(name='kld')
                 ]
    )

    # Train and validate the model with designated datasets
    history = model.fit(
        train_dset,
        validation_data=val_dset,
        epochs=epochs
    )

    # Unpack losses and other metrics
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    train_msle = history.history['msle']
    val_msle = history.history['msle']

    train_kld = history.history['kld']
    val_kld = history.history['kld']

    # Print each epoch's performance to terminal
    template = 'Epoch {}, Loss: {}, MSLE: {}, KLD: {}, Val Loss: {}, ' \
               'Val MSLE: {}, Val KLD: {}'

    if verbose:
        for i in range(epochs):
            print(template.format(i + 1,
                                  train_losses[i],
                                  train_msle[i] * 100,
                                  train_kld,
                                  val_losses[i],
                                  val_msle[i] * 100,
                                  val_kld
                                  )
                  )

    return history.history


def lstm_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    model = LSTM()

    # Try out both the options below (all zeros and random)
    # shape is: number of examples (mini-batch size), width, height, depth
    x_np = np.zeros((10, 1, 10, 13, 2))
    # x_np = np.random.rand(64, 32, 32, 3)

    # call the model on this input and print the result
    output = model.call(x_np)
    print(output)

    # Look at the model parameter shapes
    for v in model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)


if __name__ == '__main__':

    cases, deaths = generate_dset_LSTM(13, num_extra_states=4)

    train, val = cases['Alabama']['train'], deaths['Alabama']['validate']

    _ = run_lstm(train, val)

