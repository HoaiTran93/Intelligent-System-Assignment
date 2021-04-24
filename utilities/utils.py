import configuration.config as cfg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import seaborn as sns
from keras.callbacks import EarlyStopping
from matplotlib.backends.backend_pdf import PdfPages

def prepare_seq2seq_data(dataset, look_back, look_ahead):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_ahead):
        input_seq = dataset[i:(i + look_back)]
        output_seq = dataset[i + look_back:(i + look_back + look_ahead)]
        dataX.append(input_seq)
        dataY.append(output_seq)
    dataX = np.reshape(np.array(dataX),[-1,look_back,1])
    dataY = np.reshape(np.array(dataY),[-1,look_ahead,1])
    return dataX,dataY

#use only for training data
#use returned scaler object to scale test data
def standardize(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data,scaler


def scale_data(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data,scaler

#get all diagonals from a two d array
def get_diagonals(input):
    #fetch diagonals in a list
    diagonals = [input[::-1, :].diagonal(i) for i in range(-input.shape[0] + 1, 1)]
    return diagonals

def shift_time_series(X,shift_step):
    return np.roll(X,shift_step)
    
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def save_figure(path,filename,fig):
    mkdir_p(path)
    pp = PdfPages("%s/%s.pdf"%(path,filename))
    pp.savefig(fig)
    pp.close()


def load_data(data_folder,look_back,look_ahead):
    train = np.load(data_folder + "train.npy")
    validation1 = np.load(data_folder + "validation1.npy")
    validation2 = np.load(data_folder + "validation2.npy")
    test = np.load(data_folder + "test.npy")

    # standardize data. use the training set mean and variance to transform rewst of the sets
    train, train_scaler = standardize(train[:, 0].reshape(-1,1))
    validation1 = train_scaler.transform(validation1[:, 0].reshape(-1,1))
    validation2_labels = validation2[:, 1].reshape(-1,1)
    validation2 = train_scaler.transform(validation2[:, 0].reshape(-1,1))
    test_labels = test[:, 1].reshape(-1,1)
    test = train_scaler.transform(test[:, 0].reshape(-1,1))

    # prepare sequence data and labels
    X_train, y_train = prepare_seq2seq_data(train, look_back, look_ahead)
    X_validation1, y_validation1 = prepare_seq2seq_data(validation1, look_back, look_ahead)
    X_validation2, y_validation2 = prepare_seq2seq_data(validation2, look_back, look_ahead)
    X_validation2_labels, y_validation2_labels = prepare_seq2seq_data(validation2_labels, look_back, look_ahead)
    X_test, y_test = prepare_seq2seq_data(test, look_back, look_ahead)
    X_test_labels, y_test_labels = prepare_seq2seq_data(test_labels, look_back, look_ahead)
    return train_scaler, X_train, y_train, X_validation1, y_validation1, X_validation2, y_validation2, y_validation2_labels, X_test, y_test, y_test_labels
