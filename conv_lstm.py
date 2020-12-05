"""
Prediction machine learning models
Authors: Vincent Yu
Date: 12/01/2020
"""

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

        self.lstm3 = ConvLSTM2D(
            64, (2, 3), activation='relu', return_sequences=True
        )
        self.lstm4 = ConvLSTM2D(
            64, (2, 3), activation='relu', return_sequences=True
        )
        self.p2 = MaxPooling3D(pool_size=(1, 2, 2))
        self.drop2 = Dropout(0.2)

        self.lstm5 = ConvLSTM2D(
            128, (1, 3), activation='relu', return_sequences=True
        )
        self.lstm6 = ConvLSTM2D(
            128, (1, 3), activation='relu'
        )
        self.p3 = MaxPooling3D(pool_size=(1, 2, 2))
        self.drop3 = Dropout(0.2)

        self.f = Flatten()
        self.d1 = Dense(10)
        self.drop4 = Dropout(0.2)
        self.d2 = Dense(1)

    def call(self, inputs):
        x1 = self.lstm1(inputs)
        x2 = self.lstm2(x1)
        p1 = self.p1(x2)
        dp1 = self.drop1(p1)
        
        x3 = self.lstm3(dp1)
        x4 = self.lstm4(x3)
        p2 = self.p2(x4)
        dp2 = self.drop2(p2)

        x5 = self.lstm5(dp2)
        x6 = self.lstm6(x5)
        p3 = self.p3(x6)
        dp3 = self.drop3(p3)

        f = self.f(dp3)
        d1 = self.d1(f)
        dp4 = self.drop4(d1)
        output = self.d2(dp4)

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


if __name__ == '__main__':

    cases, deaths = generate_dset_LSTM(13, num_extra_states=4)

    train, val = cases['Alabama']['train'], deaths['Alabama']['validate']

    _ = run_lstm(train, val)
