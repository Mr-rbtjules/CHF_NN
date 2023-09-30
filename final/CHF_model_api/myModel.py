import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential #=regroupment of layers         
# dense == tensor or layer or set of neurons of a same level
# dropout = layer for regularisation
# bachtnormalization =  layer to normalize batch
# batch == matrix of sample for training
# LeakyRelu = layer for activation
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LeakyReLU, Activation
) #layer instance
#other activation function
from tensorflow.keras.activations import relu, sigmoid
#different type of loss function
from tensorflow.keras.losses import (
    MeanSquaredLogarithmicError, MeanAbsolutePercentageError
)
#optimizers 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import SGD
#action take at each epoch (epoch == 1 training over the all training data)
#LearningRateScheduler = programmed decrease of lr
# #earlystop when no big change
# tensorboard to vizualize evolution of metrics and hyperparameters 
from tensorflow.keras.callbacks import (
    LearningRateScheduler, EarlyStopping, TensorBoard
)
from tensorflow.keras import backend as K

#for the computation of custom metrics during training
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


#only way to work is in this file not in the class and with this name
def batch_nrmse(y_true, y_pred) -> float:
        return K.sqrt(K.mean(K.square(y_pred - y_true)))/K.mean(y_true)



class MyModel:
    #= {seed : {validation_targets':[...]}} 
    DATA = {}

    def __init__(
            self, 
            hparams:        dict = None,
            model_name:     str = None,
            auto_save:      bool = False,
            process_number: int = None
    ) ->None:
        """either create a CHF prediction model from scratch using hparams
        or load a saved one from model_name from a .h5 file and his hp 
        from json"""
        #to auto save after training
        self.auto_save = auto_save
        self.hparams = hparams
        self.name = None
        self.input_number = None
        #validation data input (or features)
        self.X_val = None
        #validation data output (or targets)
        self.y_val = None
        self.X_train = None
        self.y_train = None
        #data of normalization
        self.normalization_mean = None
        self.normalization_std = None
        self.model = None
        # callback = action took after each epoch
        self.callbacks = None
        #to compare name and now if model already exist
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.loss_function = MeanSquaredLogarithmicError()
        self.actMethod = LeakyReLU
        self.monitor_param = 'loss'
        self.other_metrics = ['mape', batch_nrmse]
        self.process_number = process_number        
        #load hp and attributes
        self._loadMyModel(model_name)

    ###PUBLIC METHOD###
    def train(
            self, 
            logs=True, 
            callbacks=True, 
            train_epochs=None
    ) -> dict:
        """train the model over train_epochs epochs and save
        the results in the attribute hparams""" 
        call = self.callbacks
        #add some callback option for special training as optimization
        if not logs:
            call = call[:-1]
        if not callbacks:
            call = None
        if not train_epochs:
            train_epochs = self.hparams['max_epochs']
        print(f"Start training model {self.name}")
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            shuffle=False,      #because already shuffled
            validation_data=(self.X_val,self.y_val),
            batch_size=self.hparams['batch_size'], 
            epochs=train_epochs,
            callbacks=call, #mpa_callback
            verbose=self.hparams['verbose']
        )
        self.hparams['trained_epochs'] += train_epochs
        self._saveResults()
        #record result in hparams
        if self.auto_save:
            self.save(overwrite=True)
        return history 

    def save(self, overwrite=False)->None: 
        """save model in .h5 format, to make predictions
        save also the hparams of the model in json files"""
        #already existing model 
        if self.name != self.now: 
            if overwrite:
                print("overwrite previous model")
                #no need to change name, save the model on top on the old
                #version but change hparams because performance might be diff
                CHF.tools.eraseHparams(self.name)
            else: #create a new hparam json and model.h5
                #set a new name
                self.name = self.now
                #new now to show the fact model already saved
                self.now = datetime.now().strftime("%Y%m%d-%H%M%S")  
        else:
            #to mark the fact that we will have a model already saved
    
            self.now = datetime.now().strftime("%Y%m%d-%H%M%S")
        #if during optimisation to distinguish
        name = self.name
        if self.process_number:
            name = 'opti' + self.name
        self.hparams['name'] = self.name
        print("Save model ", self.name)
        path = f"./saved_models/models/{name}.h5"
        #save model.h5 (always with special metric or bug)
        with custom_object_scope({'batch_nrmse': batch_nrmse}):
            self.model.save(path)
        #save hparams in json
        CHF.tools.saveHparams(name, self.hparams)
        return None

    def makeRealPredictions(self, features_data: list) -> list:
        
        scaler = StandardScaler()
        scaler.mean_ = self.normalization_mean
        scaler.scale_ = self.normalization_std
        normalized_data = scaler.transform(features_data)
        predictions = np.array([i[0] for i in self.model.predict(normalized_data)])
        print("Prediction of the CHF: ",predictions)
        return predictions
        
    def plotResult(self) -> None:
        predictions = np.array([i[0] for i in self.model.predict(self.X_val)])
        CHF.tools.plotResults(predictions, self.y_val, save_fig=True)

    def vizualize(self) -> None:
        """save a png file containing vizual rpz of the network"""
        dimension = (100,100)
        print("Model visualisation ~15s")
        CHF.tools.visualizeNn(
            self.model, 
            self.name,
            description=True, 
            figsize=dimension
        )
        return None

    ###PRIVATE METHOD###
    def _loadMyModel(self, model_name: str)->None:
        """Create a model based on hparams given in parameterss
        or use a saved one based on his name and load it from saved_models
        directory set also all the attributes"""
        #no name = new model from scratch
        if model_name == None:
            if self.hparams == None:
                Exception("Error provide hparams or model_name")
            self.input_number = self.hparams['input_number']
            self.model = Sequential()
            #name-now for new/saved models ambiguity
            self.name = self.now
            print("Load new model: ", self.name)
            self._createNewModel()   
        else:#model already existing
            path = f"./saved_models/models/{model_name}.h5"
            #load model from files alway with custom metric or bug
            with custom_object_scope({'batch_nrmse': batch_nrmse}):
                self.model = load_model(path)
            #load hparams
            self.hparams = CHF.tools.getHparamsSavedModel(model_name)
            self.name = self.hparams['name']
            self.input_number = self.hparams['input_number']
            print("Load saved model: ", self.name)

        
        #make ready for train
        print("Model Loaded, init callbacks")
        self.callbacks = self._init_callbacks()
        #check if the corresponding vali/train set already computed
        #set it and the other attributes
        self._loadData()
        print("Model ready to train")
        return None
    
    def _createNewModel(self) -> None:
        """create new model based on hparams"""
        #set a defined seed for the weights to be controlled
        #if we want to reproduce results
        seed_tf = self.hparams['seed_tf']
        tf.random.set_seed(seed_tf)
        architecture = self.hparams['architecture']
        #ex:[5,40,34,1] 2 hidden layer  5 and 1 not really layer
        for layer, neurons in enumerate(architecture[1:]):
            if layer == 0:
                self.model.add(Dense(
                    neurons, #neuron number of outputs
                    #precision of the input for the first hidden layer
                    input_shape=(architecture[0],) 
                ))
            else:
                self.model.add(Dense(neurons))
            #add activation layer
            self.model.add(Activation(
                self.actMethod(self.hparams['alpha_acti'])
            ))
            if layer != len(architecture[1:]) - 1 :
                #layer of normalisation of the batch to keep correct values, help generalize
                if self.hparams['batch_normalisation'] == True:
                    self.model.add(BatchNormalization())  
            if layer == 0 or layer == 1:
                #normalisation layer of the 2 first hidden layer
                self.model.add(Dropout(self.hparams['dropout_rate']))
        #optimization function
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

    def _init_callbacks(self) -> list:
        """action taken at evry epoch for monitoring"""
        early_stop = EarlyStopping(
            monitor='loss',          #loss easier to monitor because less variable
            min_delta=self.hparams['loss_delta_stop'],   #if nothing precise, it will check loss and stop after no change of >=0.1
            patience=self.hparams['patience'], 
            verbose=1,
            restore_best_weights=True
        )
        #schedule == function to give to LearningScheduler, with epoch and lr as param
        learningRateScheduler = LearningRateScheduler(self._lrScheduler)
        logdir = f"./logs/{self.name}"
        tb_metrics = TensorBoard(logdir)
        return [early_stop, learningRateScheduler, tb_metrics]#tb tjrs en dernier

    def displayPerf(self):
        print("mean absolute percent error : ",self.hparams['mape'])
        print("mean square logarithmic error", self.hparams['msle'])
        print("mean percent error", self.hparams['mpe'])
        print("mean ratio measured/predicted", self.hparams['mean_MP'])
        print("standart deviation of 1 ratio measured/predicted",self.hparams['std_MP'])
        print("normalized root mean squared error",self.hparams['nrmse'])

    def _saveResults(self) -> None:
        """compute and save metrics in hparams"""
        #model.predict is a weird object
        predictions = np.array([i[0] for i in self.model.predict(self.X_val)])
        mean_val = np.mean(self.y_val)
        #if mean_val != 0:
         #   mape = (np.mean(
          #      np.abs(self.y_val - predictions) / np.mean(self.y_val)
           #     )) * 100
        #else:
         #   mpe = 100
    
        mape = ((np.sum(np.abs(self.y_val - predictions) / self.y_val)) /len(self.y_val)) * 100
        print("mean absolute percent error  : ", mape)

        mpe = ((np.sum((predictions-self.y_val) / self.y_val)) /len(self.y_val)) * 100

        

        print("mean percent error  : ", mpe)

        #mean of ration measured/predict
        if predictions.all() != 0:
            mean_MP = np.mean(self.y_val/predictions)
        else: 
            mean_MP = 'inf'
        std_MP = CHF.tools.stdMP(self.y_val,predictions)
        print("mean mp :", mean_MP,"std mp :", std_MP)
        #normalised root mean squared
        nrmse = CHF.tools.nrmse(self.y_val, predictions)
        print("NRMSE:", nrmse)
        #loss 
        msle = CHF.tools.myMsle(self.y_val, predictions)
        print('msle : ', msle)

        self.hparams['mape'] = mape
        self.hparams['msle'] = msle
        self.hparams['mpe'] = mpe
        self.hparams['mean_MP'] = mean_MP
        self.hparams['std_MP'] = std_MP
        self.hparams['nrmse'] = nrmse
        return None

    def _loadData(self) -> None:
        """check if corresponding valid/train set already computed,
        if no compute and add the new data in DATA"""

        seed = self.hparams['data_seed']
        if self.process_number == None:
            key = f"input_number {self.input_number} seed {seed}"
            if key not in MyModel.DATA.keys():
                MyModel.DATA[key] = '' #te set a value to show that this set is taken
                MyModel.DATA[key] = CHF.tools.loadData(seed, self.input_number)

        ##optimize several process part
        else:
            key = f'copy {self.process_number} input_number ' \
                    f'{self.input_number} seed {seed}'
            if key not in MyModel.DATA.keys():
                Exception("error no copy _load_data prblm")

        self.X_val = MyModel.DATA[key]['validation_features']
        self.y_val = MyModel.DATA[key]['validation_targets']
        self.X_train = MyModel.DATA[key]['train_features']
        self.y_train = MyModel.DATA[key]['train_targets']
        self.normalization_mean = MyModel.DATA[key]['mean'].tolist()
        self.normalization_std = MyModel.DATA[key]['std'].tolist()

        self.hparams['normalization_mean'] = self.normalization_mean
        self.hparams['normalization_std'] = self.normalization_std
        return None

     # lr loose (1-expdecay)*100 % evry rythm epoch
    def _lrScheduler(self, epoch: int, lr: float) ->float: 
        """program a decrease of the learning rate"""
        if epoch % self.hparams['rythm'] == 0 and epoch > 0:
            return lr * self.hparams['learning_rate_decay']
        return lr
    


    ###CLASS METHOD###
    #2 things set up in the class for process, here and in _load_data
    @classmethod
    def makeDataCopies(cls, jobs, seed, input_number):#must not be called in this class
        #check first if origin exist   
        key1 = f"input_number {input_number} seed {seed}"
        if key1 not in cls.DATA.keys():
            cls.DATA[key1] = '' #te set a value to show that this set is taken
            cls.DATA[key1] = CHF.tools.loadData(seed, input_number)
        #we ensure we have 5 copies
        for i in range(jobs):
            key2 = f"copy {i} input_number {input_number} seed {seed}"
            if key2 not in cls.DATA.keys(): #add new key for new copy
                #copy from the origin
                cls.DATA[key2] = {}
                cls.DATA[key2]['validation_features'] = np.copy(
                    cls.DATA[key1]['validation_features']
                )
                cls.DATA[key2]['validation_targets'] = np.copy(
                    cls.DATA[key1]['validation_targets']
                )
                cls.DATA[key2]['train_features'] = np.copy(
                    cls.DATA[key1]['train_features']
                )
                cls.DATA[key2]['train_targets'] = np.copy(
                    cls.DATA[key1]['train_targets']
                )
                cls.DATA[key2]['mean'] = np.copy(
                    cls.DATA[key1]['mean']
                )
                cls.DATA[key2]['std'] = np.copy(
                    cls.DATA[key1]['std']
                )
            print(f"add {key2} to DATA")
            
