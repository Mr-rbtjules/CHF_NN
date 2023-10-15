import CHF_model_api as CHF

if __name__ == '__main__':
    
    
    generic_hparams = {
            'input_number': 4,
            'name': None,
            'architecture':[4,61,51,28,39,26,21,20,14,1],#[4,60,56,25,42,23,22,16,19,1],
            'learning_rate': 0.001,
            'dropout_rate' : 0.125,
            'learning_rate_decay' : 0.96,
            'rythm' : 32,
            'alpha_acti' : 0,#0.2584341076378446,
            'batch_size' : 16,
            'optimizer' : 'adam',
            'max_epochs' : 700,
            'steps': 25,
            'loss_function' : 'msle',
            'loss_delta_stop' : 0.005,
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
    
    model = CHF.MyModel(hparams=generic_hparams, auto_save=True)
    model.train()

    
    """my_opti = CHF.MyOptimizer(
        generic_hparams=generic_hparams,
        jobs=4,
        trials=5,
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
