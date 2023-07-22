import optuna
import tensorflow as tf
import model as m


#cree 2 opti differentes 
# 1 la sienne avec direct bcp de neurons
# 2 la mienne avec une augmentation progressive
#3 libre (plus couteux)


MIN_LAYER = 1
MAX_LAYER = 10
MIN_NEURONS = 32 #not absolute minimum
MAX_NEURONS = 70
OBJECTIVE_TYPE = 1

#define hparam
hp_architecture = [[5, 39, 40, 31, 42, 1], 
                   [5, 61, 51, 28, 39, 26, 21,20, 14, 1]]
hp_learning_rate = [i/1000 for i in range(1,10,1)]

hp_dropout_rate = [i/10 for i in range(1,10,1)]

hp_lrDecay_rate = [i/100 for i in range(60,100,10)]

hp_rythm = [i for i in range(10,30,10)]

hp_alpha_acti = [i/50 for i in range(0,10,1)]

hp_acti = ['relu','linear']

hp_batch_size = [16,32,64]

hp_batch_norm = [0, 1] #False/True

hp_seed = [0,1,2]

hp_optimizer = ['adam', 'sgd']

hp_max_epochs = [1]

hp_loss_function = ['msle']

hp_loss_delta_stop = [0.001]

#all none are to tune
my_hparams = {
    'architecture':None,
    'learning_rate': 0.001,
    'dropout_rate' : 0.1,
    'learning_rate_decay' : 0.96,
    'rythm' : 32,
    'alpha_acti' : 0,
    'batch_size' : 32,
    'seed' : 1,
    'optimizer' : 'adam',
    'max_epochs' : 100,
    'loss_function' : 'msle',
    'loss_delta_stop' : 0.01,
    'patience' : 15,
    'batch_normalisation' : True
}
#load data externally





#doit se baser sur le msle
def objective(trial):

    archi = [5]
    num_layers = trial.suggest_int("num_layers", MIN_LAYER, MAX_LAYER)

    if OBJECTIVE_TYPE == 1:

        #commence avec bcp
        first_layer_neurons = trial.suggest_int(
            "first_layer_neurons", 
            MIN_NEURONS, 
            MAX_NEURONS, 
            log=False #to sample smaller value
        ) 
        # Decrease 
        for i in range(num_layers - 1):
            # Decrease the number of neurons
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+2), 
                MIN_NEURONS, 
                first_layer_neurons, 
                log=False
            )
            archi.append(num_neurons)
            first_layer_neurons = num_neurons
            
        archi.append(1)
    
    elif OBJECTIVE_TYPE == 2: #increase 
        first_layer_neurons = trial.suggest_int(
            "first_layer_neurons", 
            5, 
            10, 
            log=True #to sample smaller value
        ) 
        # Decrease or increase the number of neurons in subsequent layers
        for i in range(num_layers - 1):
            # Decrease the number of neurons
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+2), 
                first_layer_neurons, 
                MAX_NEURONS, 
                log=True
            )
            archi.append(num_neurons)
            first_layer_neurons = num_neurons
            
        archi.append(1)

    elif OBJECTIVE_TYPE == 3: #random
        
        for i in range(num_layers):
            num_neurons = trial.suggest_int(
                "layer{}_neurons".format(i+1), 
                MIN_NEURONS, 
                MAX_NEURONS, 
                log=False
            )
            archi.append(num_neurons)
            first_layer_neurons = num_neurons
            
        archi.append(1)


    #trial.suggest_uniform(“dropout_rate”, 0, 1)
    #trial.suggest_loguniform('learning_rate', 1e-5, 1000)
    #trial.suggest_categorical(‘optimizer’, [‘SGD’, ‘Adam’])

    #trial.report(accuracy) ?
    #can minimise several metrics by combili
    #study.optimize(objective, n_trials=100, n_jobs=5) for parralel ?
    #maybe train prune train prune 
    #persistent storage backend like SQLite,
    #multiprocessing.array

    model = create_model(archi)
    performance_metric = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    return performance_metric

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

best_params = study.best_params
print("Best Architecture:", best_params)
