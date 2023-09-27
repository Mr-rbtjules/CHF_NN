import CHF_model_api as CHF

if __name__ == '__main__':

    ###TEST 1###
    hparams = {
    'input_number':4,
    'name': None,
    'architecture':[4, 61, 51, 28, 39, 26, 21,20, 14, 1],
    'learning_rate': 0.003420540282034936,
    'dropout_rate' : 0.3348589818400747,
    'learning_rate_decay' : 0.96,
    'rythm' : 20,
    'alpha_acti' : 0,
    'batch_size' : 32,
    'optimizer' : 'adam',
    'max_epochs' : 400,
    'loss_function' : 'msle',
    'loss_delta_stop' : 0.001,                                   
    'batch_normalisation' : True,
    'patience': 30,
    'data_seed' : 1,
    'seed_tf': 1,
    'normalization_mean': None,
    'normalization_std': None,
    'metric_name': 'msle',
    'mpe': None,
    'mean_MP': None,
    'std_MP': None,
    'nrmse': None,
    'trained_epochs': 0,
    'verbose': 1
    }
    #ajouter nb epoch entrainement totale aux hp
    
    
    """my_model = CHF.MyModel(hparams=hparams,model_name = None,auto_save=True)
    my_model.train()
    print("archi 1 finished")"""
    
    
    hparams2 = {
    'input_number':4,
    'name': None,
    'architecture':[4,60,56,25,42,23,22,16,19,1],
    'learning_rate': 0.003420540282034936,
    'dropout_rate' : 0.3348589818400747,
    'learning_rate_decay' : 0.96,
    'rythm' : 20,
    'alpha_acti' : 0,
    'batch_size' : 32,
    'optimizer' : 'adam',
    'max_epochs' : 1000,
    'loss_function' : 'msle',
    'loss_delta_stop' : 0.001,                                   
    'batch_normalisation' : True,
    'patience': 50,
    'data_seed' : 1,
    'seed_tf': 1,
    'normalization_mean': None,
    'normalization_std': None,
    'metric_name': 'msle',
    'mpe': None,
    'mean_MP': None,
    'std_MP': None,
    'nrmse': None,
    'trained_epochs': 0,
    'verbose': 1
    }

    my_model2 = CHF.MyModel(hparams=hparams2,model_name = None,auto_save=True)
    my_model2.train()
    print("finished archi 2")

    
    """names = ['20230802-143516', '20230802-143244']
    tb = CHF.MyTensorboard(names)"""
    
    """my_model = CHF.MyModel(hparams=hparams,model_name = None,auto_save=True)
    my_model.hparams['max_epochs'] = 700

    my_model.train()"""

    ###TEST2###
    
    #without the param we going to tune
    #for now only architecture
    generic_hparams = {
            'input_number': 4,
            'name': None,
            'architecture':[4,61,51,28,39,26,21,20,14,1],#[4,60,56,25,42,23,22,16,19,1],
            'learning_rate': 0.003708,
            'dropout_rate' : 0.325,
            'learning_rate_decay' : 0.96,
            'rythm' : 20,
            'alpha_acti' : 0,#0.2584341076378446,
            'batch_size' : 32,
            'optimizer' : 'adam',
            'max_epochs' : 300,
            'steps': 25,
            'loss_function' : 'msle',
            'loss_delta_stop' : 0.001,
            'batch_normalisation' : True,
            'patience': 50,
            'data_seed' : 1,
            'seed_tf': 1,
            'normalization_mean': None,
            'normalization_std': None,
            'mpe': None,
            'msle': None,
            'mean_MP': None,
            'std_MP': None,
            'nrmse': None,
            'trained_epochs': 0,
            'verbose': 1,
            'metric_name': 'mape'
    }


    
    """my_opti = CHF.MyOptimizer(
        generic_hparams=generic_hparams,
        jobs=4,
        trials=20,
        db_name='opti_mape_all',
        type=3,
        opti_architecture=True,
        opti_dropout=True,
        opti_learning_rate=True,
        opti_lr_decrease=True,
        opti_activation_method=False,
        verbose=0
    )
    my_opti.optimize_my_models()

"""
    """

    CHF.tools.reset_directories()
    my_model = CHF.MyModel(hparams=hparams,auto_save=True)
    my_model.train()

    my_model.train()"""

    ###TEST3###

    hparams = {
    'name': None,
    'input_number': 5,
    'architecture':[5, 43, 45, 48, 49, 46, 41,40,39, 36,44, 1],
    'learning_rate': 0.001633,
    'dropout_rate' : 0.243,
    'learning_rate_decay' : 0.96,
    'rythm' : 15,
    'alpha_acti' : 0,
    'batch_size' : 32,
    'optimizer' : 'adam',
    'max_epochs' : 347,
    'loss_function' : 'msle',
    'loss_delta_stop' : 0.01,                                   
    'batch_normalisation' : True,
    'patience': 15,
    'data_seed' : 1,
    'seed_tf': 1,
    'normalization_mean': None,
    'normalization_std': None,
    'mpe': 20,
    'mean_MP': None,
    'std_MP': None,
    'nrmse': None,
    'trained_epochs': 0,
    'verbose': 0
    }

    

    hparams2 = {
    'name': None,
    'architecture':[5, 61, 51, 28, 39, 26, 21,20, 14, 1],
    'learning_rate': 0.00342,
    'dropout_rate' : 0.33485,
    'learning_rate_decay' : 0.96,
    'rythm' : 15,
    'alpha_acti' : 0,
    'batch_size' : 32,
    'optimizer' : 'adam',
    'max_epochs' : 2,
    'loss_function' : 'msle',
    'loss_delta_stop' : 0.01,                                   
    'batch_normalisation' : True,
    'patience': 15,
    'data_seed' : 1,
    'seed_tf': 1,
    'normalization_mean': None,
    'normalization_std': None,
    'mpe': None,
    'mean_MP': None,
    'std_MP': None,
    'nrmse': None,
    'trained_epochs': 0,
    'verbose': 1
    }

    """my_model = CHF.MyModel(hparams=hparams,model_name = None,auto_save=False)
    my_model.hparams['max_epochs'] = 150

    my_model.train()
    my_model.save(overwrite=False)"""

    """my_model = CHF.MyModel(hparams=hparams,auto_save=True)
    my_model.train()
    my_model2 = CHF.MyModel(hparams=hparams2,auto_save=True)
    my_model2.train()"""


    """names = ['20230722-134840','20230722-134852']
    tb = CHF.MyTensorboard(names)
"""

    

    

