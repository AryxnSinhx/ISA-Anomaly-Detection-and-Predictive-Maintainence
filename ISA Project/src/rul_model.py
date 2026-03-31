from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):

    model = Sequential()

    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(BatchNormalization())

    
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(0.001),
        loss='mae',
        metrics=['mae']
    )

    return model