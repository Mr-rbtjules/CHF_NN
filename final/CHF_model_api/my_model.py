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

from tensorflow.keras.models import load_model
#important to link to the custom metric when loading
from tensorflow.keras.utils import custom_object_scope

#to normalize input
from sklearn.preprocessing import StandardScaler

import numpy as np
import random
import time
from datetime import datetime
from typing import List

import CHF_model_api as CHF



    #seed for data shuffle
np.random.seed(CHF.config.SEED_NP)

    #python seed to be sure
random.seed(CHF.config.SEED_RAND)


def batch_nrmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))/K.mean(y_true)


class My_model:


    #= {seed : {validation_targets':[...]}} 
    DATA = {}
    DATA[1] = CHF.tools.load_data(1)

    def __init__(
            self, 
            hparams:    dict = None,
            model_name:   str = None,
            auto_save: bool = False
    ) ->None:
        """
        either create a CHF prediction model from scratch using hparams
        or load a saved one from model_name from a .h5 file and his hp 
        from json
        """
        self.auto_save = auto_save
        self.hparams = hparams
        self.name = None
        self.X_val = None
        self.y_val = None
        self.X_train = None
        self.y_train = None
        self.normalization_mean = None
        self.normalization_std = None

        self.model = None
        self.callbacks = None
        #to compare name and now if model already exist
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.loss_function = MeanSquaredLogarithmicError()
        self.actMethod = LeakyReLU
        self.monitor_param = 'loss'
        self.other_metrics = ['mape', batch_nrmse]
        

        #load hp and attributes
        self.load_my_model(model_name)
        #ici hparams enfin fini (sauf pour metrics)

    def load_my_model(self, model_name: str)->None:
        """Create a model based on hparams or
        use a saved one based on his name, set also all the attributes"""
        #new
        if model_name == None:
            if self.hparams == None:
                print("Error provide hparams or name")
            self.model = Sequential()
            self.name = self.now
            print("Load new model: ", self.name)
            self.create_new_model()   
        #model already existing
        else:
            path = "./saved_models/models/" + model_name + ".h5"
            #load model from files
            with custom_object_scope({'batch_nrmse': batch_nrmse}):
                self.model = load_model(path)
            #load hparams
            self.hparams = CHF.tools.get_hparams_saved_model(model_name)
            self.name = self.hparams['name']
            print("Load saved model: ", self.name)
            print(self.hparams)

        #make ready for train
        print("Model Loaded, init callbacks")
        self.callbacks = self.init_callbacks()
        #check if the corresponding vali/train set already computed
        #set it and the other attributes
        self._load_data(self.hparams['data_seed'])
        print("Model ready to train")
        return None
    
    def create_new_model(self)->None:
        """create new model based on hparams"""
        #set a defined seed for the weights to be controlled
        #if we want to reproduce results
        seed_tf = self.hparams['seed_tf']
        tf.random.set_seed(seed_tf)

        architecture = self.hparams['architecture']
        for layer, neurons in enumerate(architecture[1:]):
            if layer == 0:
                self.model.add(Dense(
                    neurons, 
                    input_shape=(architecture[0],)
                ))
            else:
                self.model.add(Dense(neurons))
            self.model.add(Activation(
                self.actMethod(self.hparams['alpha_acti'])
            ))
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

    def init_callbacks(self) -> list:
        """action taken at evry epoch for monitoring"""
        early_stop = EarlyStopping(
            monitor='loss',          #loss easier to monitor because less variable
            min_delta=self.hparams['loss_delta_stop'],   #if nothing precise, it will check loss and stop after no change of >=0.1
            patience=self.hparams['patience'], 
            verbose=1,
            restore_best_weights=True
        )
        #schedule == function to give to LearningScheduler, with epoch and lr as param
        learningRateScheduler = LearningRateScheduler(self.lr_scheduler)
        logdir = "./logs/hparam_tuning/" + self.name
        tb_metrics = TensorBoard(logdir)
        return [early_stop, learningRateScheduler, tb_metrics]

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
        self.save_results()
        #record result in hparams
        if self.auto_save:
            self.save(overwrite=True)
        return history 
    
    def save_results(self) -> None:
        """save metrics in hparams"""
        predictions = np.array([i[0] for i in self.model.predict(self.X_val)])

        mpe = (np.mean(
            np.abs(self.y_val - predictions) / np.mean(self.y_val)
            )) * 100
        print("pourcentage moyen d'erreur relative final  : ", mpe)
        mean_MP = np.mean(self.y_val/predictions)
   
        std_MP = CHF.tools.std_MP(self.y_val,predictions)
        print("mean mp :", mean_MP,"std mp :", std_MP)

        nrmse = CHF.tools.nrmse(self.y_val, predictions)
        print("NRMSE:", nrmse)

        self.hparams['mpe'] = mpe
        self.hparams['mean_MP'] = mean_MP
        self.hparams['std_MP'] = std_MP
        self.hparams['nrmse'] = nrmse
        return None

    
    def save(self, overwrite=False)->None: 
        """save model in .h5 format, to make predictions
        save also the hparams of the model"""
        
        if self.name != self.now: #already existing model 
            if overwrite:
                print("overwrite previous model")
                #no need to change name, save the model on top on the old
                #version but change hparams because performance might be diff
                
                CHF.tools.erase_hparams(self.name)
            else: #create a new hparam json and model.h5
                #set a new name
                self.name = self.now
                #new now to show the fact model already saved
                self.now = datetime.now().strftime("%Y%m%d-%H%M%S")
                
        else:
            #to mark the fact that we will have a model already saved
            
            self.now = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.hparams['name'] = self.name
        print("Save model ", self.name)
        path = "./saved_models/models/" + self.name + ".h5"
        #save model.h5

        with custom_object_scope({'batch_nrmse': batch_nrmse}):
            self.model.save(path)
        #save hparams in json
        CHF.tools.save_hparams(self.name, self.hparams)
        return None
    
    def _load_data(self, seed: int) -> None:
        """check if corresponding valid/train set already computed,
        if no compute and add the new data in DATA"""
        if seed not in My_model.DATA.keys():
            My_model.DATA[seed] = CHF.tools.load_data(seed)
            
        self.X_val = My_model.DATA[seed]['validation_features']
        self.y_val = My_model.DATA[seed]['validation_targets']
        self.X_train = My_model.DATA[seed]['train_features']
        self.y_train = My_model.DATA[seed]['train_targets']
        self.normalization_mean = My_model.DATA[seed]['mean'].tolist()
        self.normalization_std = My_model.DATA[seed]['std'].tolist()

        self.hparams['normalization_mean'] = self.normalization_mean
        self.hparams['normalization_std'] = self.normalization_std
        return None
    # lr loose (1-expdecay)*100 % evry rythm epoch
    def lr_scheduler(self, epoch: int, lr: float) ->float: 

        if epoch % self.hparams['rythm'] == 0 and epoch > 0:
            return lr * self.hparams['learning_rate_decay']
        return lr
    
    def vizualize(self) -> None:
        """
        save a png file containing vizual rpz of the network
        """
        dimension = (100,100)
        print("Model visualisation ~15s")
        CHF.tools.visualize_nn(
            self.model, 
            self.name,
            description=True, 
            figsize=dimension
        )
        return None
