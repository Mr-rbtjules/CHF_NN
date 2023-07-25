import CHF_model_api as CHF

if __name__ == '__main__':

    ###TEST 1###
    hparams = {
    'input_number': 5,
    'name': None,
    'architecture':[5,61,51,28,39,26,21,20,14,1],
    'learning_rate': 0.01,
    'dropout_rate' :  0.25401106288895464,
    'learning_rate_decay' : 0.96,
    'rythm' : 32,
    'alpha_acti' : 0,
    'batch_size' : 32,
    'optimizer' : 'adam',
    'max_epochs' : 2,
    'loss_function' : 'msle',
    'loss_delta_stop' : 0.001,
    'batch_normalisation' : True,
    'patience': 20,
    'data_seed' : 1,
    'seed_tf': 1,
    'normalization_mean': None,
    'normalization_std': None,
    'mpe': None,
    'msle': None,
    'mean_MP': None,
    'std_MP': None,
    'nrmse': None,
    'verbose': 1,
    'metric_name': None
    }

    
    my_model = CHF.MyModel(hparams=hparams,model_name = '20230725-154946',auto_save=True)
    my_model.hparams['max_epochs'] = 1
    my_model._saveResults()


    ###TEST2###
    

    #without the param we going to tune
    #for now only architecture
    generic_hparams = {
            'name': None,
            'architecture':[5,61,51,28,39,26,21,20,14,1],
            'learning_rate': 0.001,
            'dropout_rate' : 0.2,
            'learning_rate_decay' : 0.96,
            'rythm' : 15,
            'alpha_acti' : 0,
            'batch_size' : 32,
            'optimizer' : 'adam',
            'max_epochs' : 105,
            'steps': 15,
            'loss_function' : 'msle',
            'loss_delta_stop' : 0.01,
            'batch_normalisation' : True,
            'patience': 15,
            'data_seed' : 1,
            'seed_tf': 1,
            'normalization_mean': None,
            'normalization_std': None,
            'mpe': None,
            'msle': None,
            'mean_MP': None,
            'std_MP': None,
            'nrmse': None,
            'verbose': 1,
            'metric_name': None
    }


    #
    """my_opti = CHF.MyOptimizer(
        generic_hparams=generic_hparams,
        jobs=4,
        trials=10,
        db_name='opti_dp_lr',
        type=2,
        opti_architecture=False,
        opti_dropout=True,
        opti_learning_rate=True,
        metric='msle'
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
    'architecture':[5, 61, 51, 28, 39, 26, 21,20, 14, 1],
    'learning_rate': 0.01,
    'dropout_rate' : 0.1,
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
    'verbose': 1
    }
    hparams2 = {
    'name': None,
    'architecture':[5, 61, 51, 28, 39, 26, 21,20, 14, 1],
    'learning_rate': 0.001,
    'dropout_rate' : 0.1,
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
    'verbose': 1
    }

    """my_model = CHF.MyModel(hparams=hparams,auto_save=True)
    my_model.train()
    my_model2 = CHF.MyModel(hparams=hparams2,auto_save=True)
    my_model2.train()"""


    """names = ['20230722-134840','20230722-134852']
    tb = CHF.MyTensorboard(names)
"""

    

    

