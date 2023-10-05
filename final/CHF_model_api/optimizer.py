import optuna
import tensorflow as tf
import multiprocessing
import numpy as np
import CHF_model_api as CHF
import copy
import datetime
import os
from pathlib import Path



class MyOptimizer:
    """This class create an optimizer for hyperparameter
        of deep neural network to predict CHF using 
        mainly optuna library"""
    def __init__(
            self,
            generic_hparams:        dict,
            opti_architecture:      bool = True,
            opti_dropout:           bool = False,
            opti_learning_rate:     bool = False,
            opti_lr_decrease:       bool = False,
            opti_optimizer:         bool = False,
            opti_activation_method: bool = False,
            type:                   int = 1,
            jobs:                   int = 2,
            trials:                 int = 5,
            db_name:                str = None,
            metric:                 str = 'mpe',
            verbose:                int = 0
    ) -> None:
        #value to optimize
        self.metric_name = metric
        #set to true if want to optimize those
        self.opti_archirecture = opti_architecture
        self.opti_dropout = opti_dropout
        self.opti_learning_rate = opti_learning_rate
        self.opti_optimizer = opti_optimizer
        self.opt_lr_decrease = opti_lr_decrease
        self.opti_activation_method = opti_activation_method
        #if no database create one
        if not db_name: 
            self.db_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        else: self.db_name = db_name
        
        self.db_path = Path(CHF.config.SQLDB_DIR) / f"{self.db_name}.db"
        #nb of process
        self.jobs = jobs
        #nb of diff guess for 1 process
        self.trials = trials
        self.seed = generic_hparams['data_seed']
        self.input_nb = generic_hparams['input_number']
        #range of guess
        self.dropout_range = [0.32,0.35]
        self.learning_range = [0.003, 0.004]
        self.rythm_range = [15,30]
        self.lr_decrease_range = [0.94, 0.999]
        self.alpha_activation_range = [0, 0.4]
        self.MIN_LAYER = 4
        self.MAX_LAYER = 8
        self.MIN_NEURONS = 15 #not absolute minimum
        self.MAX_NEURONS = 50
        #type of architecture
        self.type = type
        #locks for multiprocessing data access of optuna objective function
        #1 lock for each set of data shared between processes
        self.locks = [multiprocessing.Lock() for _ in range(jobs)]
        self.databases = [ CHF.MyDB(
            seed=self.seed, 
            input_number=self.input_nb
        ) for _ in range(jobs)]
        #lock for the list of locks
        self.lock_locks = multiprocessing.Lock()
        #to have access to the attributes
        self.lock_attributes = multiprocessing.Lock()

        self.output_nb = 1
        self.generic_hparams = generic_hparams
        self.verbose = verbose
        #initialise the different copies needed (one for each job/process)

    ###PUBLIC METHOD###
    def optimize_my_models(self):
                                                         
        db_path = os.path.abspath(self.db_path)
        db_path = 'sqlite:///' + db_path
        processes = []
        studies = []
        #division in process , sharing 1 same database
        for _ in range(self.jobs):
            study = optuna.create_study(
                direction="minimize", 
                storage=db_path
            )#minimize mean percent error
            studies.append(study)
            process = multiprocessing.Process(
                target=study.optimize,
                args=(self.objective,),
                kwargs={'n_trials':self.trials}
            )
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        
        best_params_list = [study.best_params for study in studies]
        best_values_list = [study.best_value for study in studies]
        best_index = min(
            range(len(best_values_list)), 
            key=best_values_list.__getitem__
        )
        best_params = best_params_list[best_index]
        best_value = best_values_list[best_index]
        print('Best Parameters:', best_params)
        print('Best value:', best_value)
        return best_params

    ###PRIVATE METHOD###

    def objective(self,trial): 
        opti_hparams = self.opti_hparams_safe(trial)
        # Find an available dataset copy and lock to use
        model = None
        our_lock = None
        our_lock_number = None
        metric = None
        #go aver all the locks to get the free fata
        try:
            our_lock_number, our_lock = self.access_data_index_safe()
            model = CHF.MyModel(
                    hparams=opti_hparams, 
                    process_number=our_lock_number, #will load the available list while init my_model object
                    auto_save=False
            )
            if our_lock != None:
                if model != None:
                    self.train_and_report(model, trial)
                    metric = model.hparams[model.hparams['metric_name']]
                    #*0.5 +  model.hparams['nrmse']*0.5 
                #drop instance and release lok 
        finally:
            if our_lock != None:
                model = None#need to be sure that data released
                our_lock.release() #no need the data anymore
                print(f'release lock {our_lock_number}')
                our_lock = None
                our_lock_number = None
            else:
                Exception("lock error")
        return metric 
    
    def access_data_index_safe(self) -> list:
        our_lock = None
        our_lock_number = None
        #first we wait the list of locks to be available
        with self.lock_locks:#quick release 
            i = 0
            loop = True
            while loop:#i <= len(self.locks):
                index = i%self.jobs
                if self.locks[index].acquire(block=False):#manual mode to pass over the diff locks
                    print(f"trial access to lock {index}")
                    our_lock = self.locks[index] #we acquire and reserve a lock  
                    our_lock_number = index            
                    loop = False
                else:
                    i += 1
                    if i >1000000:
                        loop = False
                        raise Exception("\n\n\n Error no lock\n\n\n")
                        
        return [our_lock_number, our_lock]
                
        #free the list of locks but still have one of it for us


    def train_and_report(self, model, trial) -> None:
        steps = model.hparams['steps']
        epochs = model.hparams['max_epochs']

        epoch = 0
        while epoch < epochs:
            model.train(logs=False, callbacks=False, train_epochs=steps)
            metric = model.hparams[model.hparams['metric_name']]
            #*0.5 +  model.hparams['nrmse']*0.5  if we want to weights metrics
            
            if model.hparams['mape'] < 10 or model.hparams['msle'] < 0.02:
                if model.name == model.now:#encore jamais save
                    print("\n\n\n save model good results\n\n\n")
                    model.save()
                    #mais on continue prc donne pt etre des bonnes indics

            trial.report(metric, step=epoch)
            print(f"{model.name} finished training step "\
                  f"{(epoch//model.hparams['steps'])+1}/"\
                  f"{(epochs//model.hparams['steps'])}")
            # Optional: Pruning based on intermediate results (early stopping)
            epoch += steps
            if trial.should_prune():
                print("\n\nPRUNED\n\n")
                epoch = epochs

    def make_archi_type(self, trial, type, opti_hparams) -> list:
        archi = None
        if type == 1:
            archi = self.make_decreased_archi(trial)
        elif type == 2:
            archi = self.make_increased_archi(trial)
        elif type == 3:
            archi = self.make_random_archi(trial)

        elif type == 4:
            archi = self.make_around_archi(trial, opti_hparams)
        return archi
    
    def make_around_archi(self, trial, opti_hparams):
        archi = opti_hparams['architecture']
        #make a guess for the nb og layers in the nn
        num_layers = len(archi)
        for i in range(1,num_layers-1):
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+1), 
                (archi[i] -5), 
                (archi[i] + 5), 
                log=False
            )
            archi[i] = num_neurons            
        return archi
        

    def opti_hparams_safe(self, trial)-> dict:
        """Return safely the guess optimized of ,                              
        the hparameters and the epochs number"""
        learning_rate = None
        dropout_rate = None
        opti_hparams = None                                                    
        #gain access to the attributes of the class optimizer (all the configs)
        with self.lock_attributes: 
            print("get attribute access")
            opti_hparams = copy.deepcopy(self.generic_hparams)
            
            ##archi guess###
            if self.opti_archirecture:
                opti_hparams['architecture'] = self.make_archi_type(
                    trial, 
                    self.type,
                    opti_hparams
                )
            if self.opti_dropout:
                dropout_rate = trial.suggest_float(
                    "dropout_rate", 
                    self.dropout_range[0], 
                    self.dropout_range[1]
                )
                opti_hparams['dropout_rate'] = dropout_rate
            if self.opti_learning_rate:
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    self.learning_range[0], 
                    self.learning_range[1],
                    log=False
                )
                opti_hparams['learning_rate'] = learning_rate
            if self.opti_optimizer:
                opti_hparams['optimizer'] = trial.suggest_categorical(
                    'optimizer', 
                    ['SGD', 'adam']
                )
            if self.opt_lr_decrease:
                opti_hparams['learning_rate_decay'] = trial.suggest_float(
                    'learning_rate_decay',
                    self.lr_decrease_range[0], 
                    self.lr_decrease_range[1],
                    log=False
                )
                """opti_hparams['rythm'] = trial.suggest_int(
                    'rythm',
                    self.rythm_range[0], 
                    self.rythm_range[1],
                    log=False
                )"""
            if self.opti_activation_method:
                alpha_activation = trial.suggest_float(
                    "alpha_activation",
                    self.alpha_activation_range[0], 
                    self.alpha_activation_range[1],
                    log=False
                )
                opti_hparams['alpha_acti'] = alpha_activation

        if self.jobs > 1:
            opti_hparams['verbose'] = self.verbose
        else: opti_hparams['verbose'] = 1
        #set optimized hparams    
        #relase the lock of attribute 
        return opti_hparams

    

    def make_decreased_archi(self,trial) -> list:
        archi = []
        archi.append(self.input_nb)
        #make a guess for the nb og layers in the nn
        num_layers = trial.suggest_int(
            "num_layers", 
            self.MIN_LAYER, 
            self.MAX_LAYER
        )
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
    
    def make_increased_archi(self,trial) -> list:
        archi = []
        archi.append(self.input_nb)
        #make a guess for the nb og layers in the nn
        num_layers = trial.suggest_int(
            "num_layers", 
            self.MIN_LAYER, 
            self.MAX_LAYER
        )
        first_layer_neurons = trial.suggest_int(
            "first_layer_neurons", 
            self.input_nb, 
            (self.input_nb + 5), 
            log=True #to sample smaller value
        ) 
        for i in range(num_layers - 1):
            # Decrease the number of neurons
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+2), 
                first_layer_neurons, 
                self.MAX_NEURONS, 
                log=True
            )
            archi.append(num_neurons)
            first_layer_neurons = num_neurons
            
        archi.append(self.output_nb)

        return archi
    
    def make_random_archi(self, trial) -> list:
        archi = []
        archi.append(self.input_nb)
        #make a guess for the nb og layers in the nn
        num_layers = trial.suggest_int(
            "num_layers", 
            self.MIN_LAYER, 
            self.MAX_LAYER
        )
        for i in range(num_layers):
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+1), 
                self.MIN_NEURONS, 
                self.MAX_NEURONS, 
                log=False
            )
            archi.append(num_neurons)            
        archi.append(1)
        return archi

    
    
    

    

                        

    
        
    



