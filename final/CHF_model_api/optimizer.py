import optuna
import tensorflow as tf
import multiprocessing
import numpy as np
import CHF_model_api as CHF
import copy
import sqlite3
import datetime
import os
"""pruning = early stop of optuna
pruninghook = tensorflow integration for pruning

diff type of pruner = 
MedianPruner when metric worst than metrics median it pruned
SuccessiveHalvingPruner:save resource make competition for longer and longer epoch
and less and less trials (diff type of archi)

HyperbandPruner: previous one + computer power allocate more efficiently for 
promising trials -> the one
param:
min_resource=1, max_resource=n_train_iter, reduction_factor=3

okay set the prun like under and in the objective function make several fit
spaced with  several epochs then raise

step !=epoch step 1 ->n_train_iter ins max_resource arg
intermediate_value = clf.score(X_valid, y_valid)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

usage = study = optuna.create_study(direction='minimize', pruner=optuna.pruners.HyperbandPruner())



Pruning hook concept : pruning : stop the train or
the search of optuna to see if good guess or not
hook ~ callback at a custom point -> maybe utilise pas car pas assez doc

with tf.compat.v1.Session() as sess:
        # Create the TensorFlowPruningHook and pass the session and trial object
        pruning_hook = TensorFlowPruningHook(trial, session=sess)


report ?
"""

"""sql principle we use a database to store the logs of optuna
and also the best guess informations and the db is automatically setup 
and easely shared between each process nothing to do
we also have a non reusable db not stored with sqlite:///:memory:"""

#pr l'instant garde 1 mm seed
class My_optimizer:

    def __init__(
            self,
            generic_hparams:    dict,
            opti_architecture:  bool = True,
            opti_dropout:       bool = False,
            opti_learning_rate: bool = False,
            opti_optimizer:     bool=False,
            type:               int = 1,
            jobs:               int = 2,
            trials:             int = 5,
            db_name:            str = None,
            pruner:             bool = False

    ) -> None:
        self.opti_archirecture = opti_architecture
        self.opti_dropout = opti_dropout
        self.opti_learning_rate = opti_learning_rate
        self.opti_optimizer = opti_optimizer
        self.pruner = pruner
        self.steps = 5
        self.epochs = 10
        self.db_name = db_name
        self.db_path = f'./optuna_sql_databases/{db_name}.db'
        self.conn = None
        self.trials = trials
        self.jobs = jobs
        self.seed = 1
        self.dropout_range = [0.1,0.9]
        self.learning_range = [0.001, 0.01]
        self.MIN_LAYER = 1
        self.MAX_LAYER = 10
        self.MIN_NEURONS = 32 #not absolute minimum
        self.MAX_NEURONS = 70
        self.OBJECTIVE_TYPE = type
        #locks for multiprocessing data access of optuna objective function
        self.locks = [multiprocessing.Lock() for _ in range(jobs)]
        self.lock_locks = multiprocessing.Lock()
        self.lock_attributes = multiprocessing.Lock()

        self.input_nb = 5
        self.output_nb = 1
        self.generic_hparams = generic_hparams
        #initialise the different copies needed (one for each job/process)
        CHF.My_model.make_data_copies(self.jobs,self.seed)


    def optimize_my_models(self):
        pruner = None
        """if self.pruner:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1, 
                max_resource=self.steps, 
                reduction_factor=3
            )"""
        #set up data base to regroup optimisation informations
        
        db_path = os.path.abspath(self.db_path)
        db_path = 'sqlite:///' + db_path
        if self.db_name:
            study = optuna.load_study(
                study_name=self.db_name, 
                storage=db_path
            )#minimize mean percent error
        else:
            self.db_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.db_path = f'./optuna_sql_databases/{self.db_name}.db'
            study = optuna.create_study(
                direction="minimize", 
                storage=db_path,
                pruner=pruner
            )#minimize mean percent error
        study.optimize(self.objective, n_trials=self.trials, n_jobs=self.jobs)
        best_params = study.best_params
        print("Best Architecture:", best_params)
        return best_params
    


    def get_archi_guessed(self, trial):
        archi = []
        archi.append(self.input_nb)
        #make a guess for the nb og layers in the nn
        num_layers = trial.suggest_int("num_layers", self.MIN_LAYER, self.MAX_LAYER)
        #start with a lot of neurons
        first_layer_neurons = trial.suggest_int(
            "first_layer_neurons", 
            self.MIN_NEURONS, 
            self.MAX_NEURONS, 
            log=False #to sample smaller value
        ) 
        # Decrease 
        for i in range(num_layers - 1):
            # Decrease the number of neurons
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+2), 
                self.MIN_NEURONS, 
                first_layer_neurons, 
                log=False
            )
            archi.append(num_neurons)
            #keep the previous one to make smaller
            first_layer_neurons = num_neurons
            
        archi.append(self.output_nb)
        return archi
    
    def opti_hparams_safe(self, trial)-> dict:
        """Return safely the guess optimized of ,
        the hparameters and the epochs number
        """
        learning_rate = None
        dropout_rate = None
        opti_hparams = None

        with self.lock_attributes: #gain access to the attributes of the class optimizer (all the configs)
            print("get attribute access")
            opti_hparams = copy.copy(self.generic_hparams)
            opti_hparams['max_epochs'] = self.epochs
                ##archi guess###
            if self.opti_archirecture:
                
                opti_hparams['architecture'] = self.get_archi_guessed(trial)
            
            if self.opti_dropout:
                trial.suggest_uniform("dropout_rate", self.dropout_range[0], self.dropout_range[1])
                opti_hparams['dropout_rate'] = dropout_rate
            if self.opti_learning_rate:
                trial.suggest_loguniform("learning_rate", self.learning_range[0], self.learning_range[1])
                opti_hparams['learning_rate'] = learning_rate
            if self.opti_optimizer:
                opti_hparams['optimizer'] = trial.suggest_categorical('optimizer', ['SGD', 'adam'])

        #set optimized hparams    
        #relase the lock of attribute 
        return opti_hparams

    def access_data_index_safe(self) -> list:
        our_lock = None
        our_lock_number = None
        #first we wait the list of locks to be available
        with self.lock_locks:#quick release 
            i = 0
            while i <= len(self.locks):
                if self.locks[i].acquire(block=False):#manual mode to pass over the diff locks
                    print(f"trial access to lock {i}")
                    our_lock = self.locks[i] #we acquire and reserve a lock  
                    our_lock_number = i            
                    i = len(self.locks) + 1 #break
                else:
                    i += 1
        return [our_lock_number, our_lock]
                
        #free the list of locks but still have one of it for us

    def objective(self,trial): 
        
        opti_hparams = self.opti_hparams_safe(trial)


        # Find an available dataset copy and lock to use
        model = None
        our_lock = None
        our_lock_number = None
        metric = None
        #passe sur tt les locks pour voir lequel est dispo
        
        try:
            our_lock_number, our_lock = self.access_data_index_safe()

        finally:
            model = CHF.My_model(
                    hparams=opti_hparams, 
                    process_number=our_lock_number, #will load the available list while init my_model object
                    auto_save=False
            )
            
            if our_lock != None:
                if model != None:
                    self.train_and_report(model, trial, opti_hparams)
                    metric = model.hparams['mpe']#*0.5 +  model.hparams['nrmse']*0.5 
                #drop instance and release lok 
                model = None#need to be sure that data released
                our_lock.release() #no need the data anymore
                print(f'release lock {our_lock_number}')
                our_lock = None
                our_lock_number = None
            else:
                Exception("lock error")
        return metric 
    

    def train_and_report(self, model, trial, opti_hparams) -> None:
        steps = opti_hparams['steps']
        for epoch in range(0,opti_hparams['max_epochs'],steps):

            model.train(logs=False, callbacks=False, train_epochs=steps)
            metric = model.hparams['mpe']#*0.5 +  model.hparams['nrmse']*0.5 
            
            trial.report(metric, step=epoch)

            # Optional: Pruning based on intermediate results (early stopping)
            if trial.should_prune():
                raise optuna.TrialPruned()        

    
        
    



