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
    'normalization_std': None
}
    CHF.tools.remove_directory_content()
    #my_model = CHF.My_model(hparams=hparams, auto_save=True)
    #my_model.train()

    

    

