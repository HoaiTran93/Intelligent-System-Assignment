import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(123)
rn.seed(123)
#single thread
session_conf = tf.ConfigProto(
intra_op_parallelism_threads=1,
inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import models.lstm as lstm
import configuration.config as cfg
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from keras.utils import plot_model
import utilities.utils as util
import numpy as np

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set_style("whitegrid")


def make_plots(context,predictions_timesteps,true_values,look_ahead,title,path,save_figure):
    step = 1
    if look_ahead > 1:
        step = look_ahead - 1
    for idx, i in enumerate(np.arange(0, look_ahead, step)):
        fig = plt.figure()
        plt.title(" Timestep: %d "%i)
        plt.xlabel("Time Step")
        plt.ylabel("Power Consumption")
        plt.plot(true_values, label="actual", linewidth=1)
        plt.plot(predictions_timesteps[:, i], label="prediction", linewidth=1, linestyle="--")
        error = abs(true_values - predictions_timesteps[:, i])
        plt.plot(error, label="error", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        if save_figure:
            util.save_figure(path,"%s_timestep_%d"%(context,i), fig)
        plt.show()


def get_predictions(context,model,X,y,train_scaler,batch_size,look_ahead,look_back,epochs,experiment_id):
    predictions = model.predict(X, batch_size=batch_size)
    # rescale
    predictions = train_scaler.inverse_transform(predictions)
    y = train_scaler.inverse_transform(y)

    # extract first timestep for true values
    y_true = y[:, 0].flatten()
    # diagonals contains a reading's values calculated at different points in time
    diagonals = util.get_diagonals(predictions)
    # the top left and bottom right predictions do not contain predictions for all timesteps
    # fill the missing prediction values in diagonals. curenttly using the first predicted value for all missing timesteps
    for idx, diagonal in enumerate(diagonals):
        diagonal = diagonal.flatten()
        # missing value filled with the first value
        diagonals[idx] = np.hstack((diagonal, np.full(look_ahead - len(diagonal), diagonal[0])))
    predictions_timesteps = np.asarray(diagonals)

    shifted_1 = util.shift_time_series(y_true, 1)

    title = "Prediction on %s data. %d epochs, look back %d, look_ahead %d & batch_size %d." % (
                 context, epochs, look_back, look_ahead, batch_size)
    path = "%s/%s/"%("imgs",experiment_id)
    make_plots(context,predictions_timesteps,y_true,look_ahead,title,path,cfg.run_config['save_figure'])

    return predictions_timesteps, y_true


def run():
    #load config settings
    experiment_id = cfg.run_config['experiment_id']
    data_folder = cfg.run_config['data_folder']
    look_back = cfg.multi_step_lstm_config['look_back']
    look_ahead = cfg.multi_step_lstm_config['look_ahead']
    batch_size = cfg.multi_step_lstm_config['batch_size'] -(look_back+look_ahead) +1
    epochs = cfg.multi_step_lstm_config['n_epochs']
    dropout = cfg.multi_step_lstm_config['dropout']
    layers = cfg.multi_step_lstm_config['layers']
    loss = cfg.multi_step_lstm_config['loss']
    shuffle = cfg.multi_step_lstm_config['shuffle']
    patience = cfg.multi_step_lstm_config['patience']
    validation = cfg.multi_step_lstm_config['validation']
    learning_rate = cfg.multi_step_lstm_config['learning_rate']

    train_scaler, X_train, y_train, X_validation1, y_validation1, X_validation2, y_validation2, validation2_labels, \
    X_test, y_test, test_labels = util.load_data(data_folder, look_back, look_ahead)

    #For stateful lstm the batch_size needs to be fixed before hand.
    #We also need to ernsure that all batches shud have the same number of samples. So we drop the last batch as it has less elements than batch size
    if batch_size > 1:
        n_train_batches = int(len(X_train)/batch_size)
        len_train = n_train_batches * batch_size
        if len_train < len(X_train):
            X_train = X_train[:len_train]
            y_train = y_train[:len_train]

        n_validation1_batches = int(len(X_validation1)/batch_size)
        len_validation1 = n_validation1_batches * batch_size
        if n_validation1_batches * batch_size < len(X_validation1):
            X_validation1 = X_validation1[:len_validation1]
            y_validation1 = y_validation1[:len_validation1]

        n_validation2_batches = int(len(X_validation2) / batch_size)
        len_validation2 = n_validation2_batches * batch_size
        if n_validation2_batches * batch_size < len(X_validation2):
            X_validation2 = X_validation2[:len_validation2]
            y_validation2 = y_validation2[:len_validation2]

        n_test_batches = int(len(X_test)/batch_size)
        len_test = n_test_batches * batch_size
        if n_test_batches * batch_size < len(X_test):
            X_test = X_test[:len_test]
            y_test = y_test[:len_test]

    stateful_lstm = lstm.StatefulMultiStepLSTM(batch_size=batch_size, look_back=look_back, look_ahead=look_ahead,
                                          layers=layers,
                                          dropout=dropout, loss=loss, learning_rate=learning_rate)
    model = stateful_lstm.build_model()
    history = lstm.train_stateful_model(model, X_train, y_train, batch_size, epochs, shuffle, validation, (X_validation1, y_validation1),
                     patience)
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    if cfg.run_config['save_figure']:
        util.save_figure("%s/%s/"%("imgs",experiment_id), "train_errors" , fig)

    validation2_loss = model.evaluate(X_validation2, y_validation2, batch_size=batch_size, verbose=2)
    test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    predictions_train, y_true_train = get_predictions("Train", model, X_train, y_train, train_scaler,
                                                               batch_size, look_ahead, look_back, epochs, experiment_id,
                                                               )
    np.save(data_folder + "train_predictions", predictions_train)
    np.save(data_folder + "train_true",y_true_train)

    predictions_validation1, y_true_validation1 = get_predictions("Validation1", model, X_validation1, y_validation1,
                                                                  train_scaler, batch_size, look_ahead, look_back,
                                                                  epochs, experiment_id,
                                                                  )

    np.save(data_folder + "validation1_predictions", predictions_validation1)
    np.save(data_folder + "validation1_true", y_true_validation1)


    predictions_validation2, y_true_validation2 = get_predictions("Validation2", model, X_validation2, y_validation2,
                                                                  train_scaler, batch_size, look_ahead, look_back,
                                                                  epochs, experiment_id,
                                                                 )
    
    np.save(data_folder + "validation2_predictions", predictions_validation2)
    np.save(data_folder + "validation2_true", y_true_validation2)
    np.save(data_folder + "validation2_labels", validation2_labels)
    predictions_test, y_true_test = get_predictions("Test", model, X_test, y_test, train_scaler, batch_size, look_ahead,
                                                    look_back, epochs, experiment_id,
                                                   )

    np.save(data_folder + "test_predictions", predictions_test)
    np.save(data_folder + "test_true", y_true_test)
    np.save(data_folder + "test_labels", test_labels)


if __name__ == "__main__":
    run()
