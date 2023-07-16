import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanAbsolutePercentageError


import model as m

hp_architecture = [[5, 39, 40, 31, 42, 1], 
                   [5, 61, 51, 28, 39, 26, 21,20, 14, 1]]
hp_learning_rate = [i/1000 for i in range(1,10,1)]

hp_dropout_rate = [i/10 for i in range(1,10,1)]

hp_lrDecay_rate = [i/100 for i in range(60,100,10)]

hp_rythm = [i for i in range(10,30,10)]

hp_alpha_acti = [i/50 for i in range(0,10,1)]

hp_batch_size = [16,32,64]

hp_batch_norm = [True, False]

hp_seed = [0,1,2]

hp_optimizer = [Adam, SGD]

hp_max_epochs = [100]

hp_loss_function = [MeanSquaredLogarithmicError]

hp_loss_delta_stop = [0.001]





HP_ARCHITECTURE = hp.HParam('architecture',
                              hp.Discrete([str(i) for i in hp_architecture]))
HP_LEARNING_RATE = hp.HParam('learning_rate', 
                             hp.Discrete(hp_learning_rate))
HP_DROPOUT_RATE = hp.HParam('drouptout_rate',
                            hp.Discrete(hp_dropout_rate))
HP_LRDECAY_RATE = hp.HParam('lrDecay_rate', 
                            hp.Discrete(hp_lrDecay_rate))
HP_RYTHM_RATE = hp.HParam('rythm', 
                          hp.Discrete(hp_rythm))
HP_ALPHA_ACTI = hp.HParam('alpha_acti', 
                          hp.Discrete(hp_alpha_acti))
HP_BATCH_SIZE = hp.HParam('batch_size', 
                          hp.Discrete(hp_batch_size))
HP_BATCHNORM = hp.HParam('batch_normalisation', 
                         hp.Discrete(hp_batch_norm))


HP_SEED = hp.HParam('seed', 
                    hp.Discrete(hp_seed))

HP_OPTIMIZER = hp.HParam('optimizer', 
                         hp.Discrete(['adam','sgd']))

myMetrics = [hp.Metric('msle', display_name='msle'), 
             hp.Metric('mape', display_name='mape')]#,
             #hp.Metric(nrmse, display_name='nrmse')]


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[
            HP_ARCHITECTURE, 
            HP_LEARNING_RATE,
            HP_DROPOUT_RATE,
            HP_LRDECAY_RATE,
            HP_RYTHM_RATE,
            HP_ALPHA_ACTI,
            HP_BATCH_SIZE,
            HP_BATCHNORM,
            HP_SEED,
            HP_OPTIMIZER
            ],
        metrics=myMetrics,
    )


test_hparams = {
    'architecture':hp_architecture[0],
    'learning_rate': hp_learning_rate[0],
    'dropout_rate' : hp_dropout_rate[0],
    'lrDdecay_rate' : hp_lrDecay_rate[0],
    'rythm' : hp_rythm[0],
    'alpha_acti' : hp_alpha_acti[0],
    'batch_size' : hp_batch_size[0],
    'batch_norm' : hp_batch_norm[0],
    'seed' :hp_seed[0],
    'optimizer' : hp_optimizer[0]
}


data = m.load_data('../sort_data.csv')

m.train_model(test_hparams, data)



