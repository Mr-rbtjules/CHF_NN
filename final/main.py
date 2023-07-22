import CHF_model_api as CHF

if __name__ == '__main__':


    hparams = {
    'name': None,
    'architecture':[5, 39, 40, 31, 42, 1],
    'learning_rate': 0.001,
    'dropout_rate' : 0.2,
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
    'nrmse': None
    }
    hparams2 = {
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
    'nrmse': None
    }


    """
    my_model = CHF.My_model(hparams=hparams,auto_save=True)
    my_model.train()"""

    #CHF.tools.reset_directories()

    """my_model = CHF.My_model(hparams=hparams,auto_save=True)
    my_model.train()
    my_model2 = CHF.My_model(hparams=hparams2,auto_save=True)
    my_model2.train()"""


    """names = ['20230722-134840','20230722-134852']
    tb = CHF.My_tensorboard(names)
"""

    

    

