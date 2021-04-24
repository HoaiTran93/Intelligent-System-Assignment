import time
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
import keras.backend as K
import shutil
import os



class ResetStatesCallback(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.model.reset_states()



class StatefulMultiStepLSTM(object):
    def __init__(self,batch_size, look_back, look_ahead, layers, dropout, loss, learning_rate):
        self.batch_size = batch_size
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.dropout = dropout

    def build_model(self):
        # first add input to hidden1
        self.model.add(LSTM(
            units=self.layers['hidden1'],
            batch_input_shape=(self.batch_size,self.look_back,self.layers['input']),
            #batch_size=self.batch_size,
            stateful=True,
            # unroll=True,
            return_sequences=True if self.n_hidden > 1 else False))
        self.model.add(Dropout(self.dropout))

        # add dense layer with output dimension to get output for one time_step
        self.model.add(Dense(units=self.layers['output']))

        # Repeat for look_ahead steps to get outputs for look_ahead timesteps.
        self.model.add(RepeatVector(self.look_ahead))

        # add activation
        self.model.add(Activation("linear"))

        # compile model and print summary
        start = time.time()
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate,decay= .99))
        #self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        self.model.summary()
        return self.model


def train_stateful_model(model, x_train, y_train, batch_size, epochs, shuffle, validation, validation_data, patience):
    training_start_time = time.time()

    try:
        shutil.rmtree('checkpoints')
    except:
        pass
    os.mkdir('checkpoints')
    checkpoint = ModelCheckpoint(monitor='val_acc',filepath='checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
                                 save_best_only=True)

    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        print('x size', x_train.shape)

        history_callback = model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, validation_data = validation_data,
                                     shuffle=shuffle, verbose=2, callbacks=[ResetStatesCallback(),early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs= epochs,callbacks=[ResetStatesCallback()],
                                     shuffle=shuffle, verbose=2)
    return history_callback



def train_model(model, x_train, y_train, batch_size, epochs, shuffle, validation, validation_data, patience):
    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        history_callback = model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, validation_data = validation_data,
                                     shuffle=shuffle, verbose=2, callbacks=[early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs= epochs,shuffle=shuffle, verbose=2)
    return history_callback

