import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential #regroupement de layer formant le modele
#dense == tensor ou layer ou ensemble de neurons d'un mm niveau 
from tensorboard.plugins.hparams import api as hp

 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Activation #layer instance

#import la fct elu pour pouvoir gerer HP alpha
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import SGD

from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard

from tensorflow.keras import backend as K

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import numpy as np
import random
import time
from datetime import datetime

from vizu import *


def load_data(dir, seed_ss, seed_pre_shuffle):

    data = pd.read_csv(dir) 

    validation_data = data.groupby('CHF').apply(
        lambda x: x.sample(frac=0.2, random_state=seed_ss)
    ).droplevel(0).sample(frac=1, random_state=seed_pre_shuffle)


    training_data = data.drop(
        validation_data.index
    ).sample(frac=1, random_state=seed_pre_shuffle)


    X_train = training_data.iloc[:, 1:6].values #inputs
    y_train = training_data.iloc[:, 7].values #target outputs

    X_val = validation_data.iloc[:, 1:6].values
    y_val = validation_data.iloc[:, 7].values

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    #then use the normalisation of the first set
    X_val = scaler.transform(X_val)
    return [X_train, X_val, y_train, y_val]


def batch_nrmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))/K.mean(y_true)

def my_std(y_val, predictions):
    """
    return standard deviation from 1 for the 
    ration measured/predict
    """
    
    MP = y_val/predictions
    
    std = 0
    for i in MP:
        std += (1-i)**2
    return np.sqrt(std/len(MP))

def my_nrmse(y_val,predictions):
    sum = 0
    for i in range(len(y_val)):
        sum += (y_val[i]-predictions[i])**2

    nrmse = np.sqrt(sum/len(y_val)) / np.mean(y_val)

    return nrmse


def train_model(hparams, logable_hparams,logdir): #remove data to set a class


    architecture = hparams['architecture']

    dropoutRate = hparams['dropout_rate']
    learningRate = hparams['learning_rate']
    expDecay = hparams['learning_rate_decay']
    rythm = hparams['rythm']
    testPercent = 0.2
    maxEpochs = hparams['max_epochs']
    thresholdActi = 0
    #! >=0  = pente pour tt x en dessous de thresold
    alphaActi = hparams['alpha_acti']
    clipGrad = 1

    
    seed_pre_shuffle = 50 #for numpy
    seed_split = 40
    seed_tf = 60
    seed_python = 30
    seed_numpy = 70
    seed_ss = 90

    seed_nb = hparams['seed']

    seed_pre_shuffle += seed_nb
    seed_split += seed_nb
    seed_tf += seed_nb
    seed_python += seed_nb
    seed_numpy += seed_nb
    seed_ss += seed_nb

    #keep that way or str error , weird
    loss_function = MeanSquaredLogarithmicError()##hparams['loss_function']#MeanSquaredLogarithmicError()#
    
    other_metrics = ['mape', batch_nrmse]

    monitorParam = 'loss'
    loss_delta_stop = hparams['loss_delta_stop']
    batchSize = hparams['batch_size']

    batchNorm = hparams['batch_normalisation']

    opti = hparams['optimizer']
    if opti == 'adam':
        opti = Adam
    else: opti = SGD
    
    actMethod = LeakyReLU(alpha=alphaActi)


    """NAME = "CHF_hp_" + datetime.now().strftime("%Y%m%d-%H%M%S")
 
    #pour avoir un resumé sur tensorboard
    log_dir = "../logs/{}".format(NAME)  # Specify the directory where logs will be stored
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    ---> to implement maybe add to callback"""

    #seed for weights and all
    tf.random.set_seed(seed_tf)

    #seed for data shuffle
    np.random.seed(seed_numpy)

    #python seed to be sure
    random.seed(seed_python)

    X_train, X_val, y_train, y_val = load_data(
        "../sort_data.csv", 
        seed_ss, 
        seed_pre_shuffle
    )

    model = Sequential()
# Add the layers

    for layer, neurons in enumerate(architecture[1:]):
        
        if layer == 0:
            model.add(Dense(neurons, input_shape=(architecture[0],)))
        else:
            model.add(Dense(neurons))
        model.add(Activation(actMethod))
        if  layer != len(architecture[1:]) - 1 :#and layer != len(architecture[1:]) - 2 :
            if batchNorm == True:
                model.add(BatchNormalization())
            
        if layer == 0 or layer == 1:
            model.add(Dropout(dropoutRate))
        
        



    model.compile(optimizer=opti(learning_rate=learningRate), 
                loss=loss_function, metrics=other_metrics)
    




#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#visualize_nn(model, description=True, figsize=(100,100))#prend qq secondes + enregistre



    
    early_stop = EarlyStopping(monitor=monitorParam,          #mieux de monitorer la loss car moins variable 
                           min_delta=loss_delta_stop,             #if nothing precise, it will check loss and stop after no change of >=0.1
                           patience=25, verbose=1,
                           restore_best_weights=True) #when stop ensure we keep best perf

    # tt les 32 epochs lr perd 4%
    #schedule == function to give to LearningScheduler, with epoch and lr as param
    def lr_scheduler(epoch, lr):          #to modify and/or add to HP   
        if epoch % rythm == 0 and epoch > 0:
            return lr * expDecay
        return lr

    learningRateScheduler = LearningRateScheduler(lr_scheduler)
    
    tb_metrics = tf.keras.callbacks.TensorBoard(logdir),  # log metrics log_dir doit etre le mm pr tes les diff modele
    hp_tb = hp.KerasCallback(logdir, logable_hparams)

    history = model.fit(X_train, y_train,
                    shuffle=False,                  #comme ça on le fait ns mm avec une seed
                    validation_data=(X_val,y_val),
                    batch_size=batchSize, epochs=maxEpochs,
                    callbacks=[early_stop, learningRateScheduler, tb_metrics, hp_tb],
                    verbose=1) 


    predictions = np.array([i[0] for i in model.predict(X_val)])

    mean_yv = np.mean(y_val)
    if mean_yv > 0:
        mpe = (np.mean(np.abs(y_val - predictions) / mean_yv)) * 100
        nrmse = my_nrmse(y_val,predictions)
    else: 
        mpe = 1.5
        nrmse = 1.5

    if not np.any(predictions == 0):
        mean_MP = np.mean(y_val/predictions)
        std = my_std(y_val,predictions)
    else: 
        mean_MP = 1.5
        std = 1.5

 
    
    """
    final_metrics = {
        'mean_percent_error': mpe,
        'mean_MP': mean_MP,
        'std' : std,
        'nrmse' : nrmse
    }"""

    final_metrics = model.evaluate(X_val,y_val) #loss , mape , nrmse
    
    return final_metrics
    
