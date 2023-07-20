import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential #=regroupment of layers

# dense == tensor or layer or set of neurons of a same level
# dropout = layer for regularisation
# bachtnormalization =  layer to normalize batch
# batch == matrix of sample for training
# LeakyRelu = layer for activation
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Activation #layer instance

#other activation function
from tensorflow.keras.activations import relu, sigmoid

#different type of loss function
from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanAbsolutePercentageError

#optimizers 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import SGD

#action take at each epoch (epoch == 1 training over the all training data)
#LearningRateScheduler = programmed decrease of lr
# #earlystop when no big change
# tensorboard to vizualize evolution of metrics and hyperparameters 
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard

#for the computation of custom metrics during training
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model, save_model

#to normalize input
from sklearn.preprocessing import StandardScaler

import numpy as np
import random
import time
from datetime import datetime
from typing import List

import CHF_model_api as CHF


tf.random.set_seed(CHF.config.SEED_TF)

    #seed for data shuffle
np.random.seed(CHF.config.SEED_NP)

    #python seed to be sure
random.seed(CHF.config.SEED_RAND)




class My_model:


    #= {seed : {validation_targets':[...]}} 
    DATA = {}
    DATA[1] = CHF.tools.load_data(1)

    def __init__(
            self, 
            hparams:    dict = None,
            model_name:   str = None
    ) ->None:
        """
        either create a CHF prediction model from scratch
        or load one and his hp from a .h5 file
        """
        self.hparams = hparams
        self.name = None
        self.X_val = None
        self.y_val = None
        self.X_train = None
        self.y_train = None
        self.normalization_mean = None
        self.normalization_std = None
        #to now if model already exist
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.loss_function = MeanSquaredLogarithmicError()
        self.other_metrics = ['mape', self.batch_nrmse]
        self.model = None
        self.callbacks = None
        self.load_my_model(model_name)
        
        #ici hparams enfin fini (sauf pour metrics)
        
    #a tester prc pas sur que tf kiffe
    def batch_nrmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))/K.mean(y_true)


    def _load_data(self, seed: int) -> None:
        """check if corresponding valid/train set already computed,
        if no compute and add the new data in DATA"""
        if seed not in My_model.DATA.keys():
            My_model.DATA[seed] = CHF.tools.load_data(seed)
            
        self.X_val = My_model.DATA[seed]['validation_features']
        self.y_val = My_model.DATA[seed]['validation_targets']
        self.X_train = My_model.DATA[seed]['train_features']
        self.y_train = My_model.DATA[seed]['train_targets']
        self.normalization_mean = My_model.DATA[seed]['mean']
        self.normalization_std = My_model.DATA[seed]['std']

        return None

    def slr_scheduler(self, epoch: int, lr: float) ->float:   
        if epoch % self.hparams['rythm'] == 0 and epoch > 0:
            return lr * self.hparams['learning_rate_decay']
        return lr

    def init_callbacks(self) -> list:
        
        early_stop = EarlyStopping(
            monitor='loss',          #mieux de monitorer la loss car moins variable 
            min_delta=self.hparams['loss_delta_stop'],   #if nothing precise, it will check loss and stop after no change of >=0.1
            patience=self.hparams['patience'], 
            verbose=1,
            restore_best_weights=True
        ) 

        # lr loose (1-expdecay)*100 % evry rythm epoch
        #schedule == function to give to LearningScheduler, with epoch and lr as param
        learningRateScheduler = LearningRateScheduler(self.lr_scheduler)
        logdir = "./logs/hparam_tuning/" + self.name
        tb_metrics = TensorBoard(logdir)
        return [early_stop, learningRateScheduler, tb_metrics]

    def create(self)->None:

        self.callbacks = self.init_callbacks()

        #que pr les nouvaux modeles
        if self.name == self.now:
            architecture = self.hparams['architecture']
            for layer, neurons in enumerate(architecture[1:]):
        
                if layer == 0:
                    self.model.add(Dense(
                        neurons, 
                        input_shape=(architecture[0],)
                    ))
                else:
                    self.model.add(Dense(neurons))

                self.model.add(Activation(LeakyReLU))

                if  layer != len(architecture[1:]) - 1 :
                    if self.hparams['batch_normalisation'] == True:
                        self.model.add(BatchNormalization())
                    
                if layer == 0 or layer == 1:
                    self.model.add(Dropout(self.hparams['dropout_rate']))
            
            opti = self.hparams['optimizer']
            if opti == 'adam':
                opti = Adam
            else: opti = SGD

            self.model.compile(
                optimizer=opti(learning_rate=self.hparams['learning_rate']), 
                loss=self.loss_function, 
                metrics=self.other_metrics
            )
            
        return None

    def train(self) -> dict:
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            shuffle=False,      #because already shuffled
            validation_data=(self.X_val,self.y_val),
            batch_size=self.hparams['batch_size'], 
            epochs=self.hparams['max_epochs'],
            callbacks=self.callbacks, #mpa_callback
            verbose=1
        ) 
        return history 

    def save(self, overwrite=False)->None:
        """save model in .h5 format, to make predictions
        save also the hparams of the model"""

        if self.name != self.now: #already existing model
            if overwrite:
                #no need to change name, save the model on top on the old
                #version but change hparams because performance might be diff
                
                CHF.tools.erase_hparams(self.name)
            else: #create a new hparam json and model.h5
                self.name = self.now
                self.hparams['name'] = self.name
        
        path = "./saved_models/" + self.name + ".h5"
        self.model.save_model(path)
        CHF.tools.save_hparams(self.name, self.hparams)
        return None
    
    def load_my_model(self, model_name: str)->None:
        """Create a model based on hparams or
        use one based on path of previous model"""
        #model already existing
        if model_name != None:
            path = "./models/" + model_name + ".h5"

            self.model = load_model(path)
            self.hparams = CHF.tools.get_hparams_saved_model(model_name)
            self.name = self.hparams['name']
            
            
        #new
        else:
            self.name = self.now
            self.model = Sequential()
            
        self.create()
        #check if the corresponding vali/train set already computed
        self._load_data(self.hparams['data_seed'])
        return None
    
    
    def vizualize(self) -> None:
        """
        save a png file containing vizual rpz of the network
        """
        dimension = (100,100)
        CHF.tools.visualize_nn(
            self.model, 
            description=True, 
            figsize=dimension)
        return None


    

    


    
