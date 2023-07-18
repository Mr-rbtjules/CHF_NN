import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tensorflow.keras.losses import MeanSquaredLogarithmicError, MeanAbsolutePercentageError


import model as m

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




"""HP_ARCHITECTURE = hp.HParam('architecture',
                              hp.Discrete([str(i) for i in hp_architecture]))
"""
HP_ARCHITECTURE = hp.HParam('architecture',
                              hp.Discrete([0,1]))


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


myMetrics = [hp.Metric('accuracy', display_name='Accuracy')]
"""[hp.Metric('msle', display_name='msle'), #voir diff avec et sans
             hp.Metric('mape', display_name='mape')]#,
             #hp.Metric(nrmse, display_name='nrmse')]"""

prime_log_dir = 'logs/hparam_tuning'
with tf.summary.create_file_writer(prime_log_dir).as_default():
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
        metrics=myMetrics
    )

#ajouter une fonction qui selectionne random 
#chaque hp
my_hparams = {
    'architecture':hp_architecture[0],
    'learning_rate': hp_learning_rate[0],
    'dropout_rate' : hp_dropout_rate[0],
    'learning_rate_decay' : hp_lrDecay_rate[0],
    'rythm' : hp_rythm[0],
    'alpha_acti' : hp_alpha_acti[0],
    'batch_size' : hp_batch_size[0],
    'seed' : hp_seed[0],
    'optimizer' : hp_optimizer[0],
    'max_epochs' : hp_max_epochs[0],
    'loss_function' : hp_loss_function[0],
    'loss_delta_stop' : hp_loss_delta_stop[0],
    'batch_normalisation' : hp_batch_norm[0]
}


#1run
#besoin de 2 hparam 
#1 pour les configs
logable_hparams = {
        HP_ARCHITECTURE : 0,
        HP_LEARNING_RATE : my_hparams['learning_rate'],
        HP_DROPOUT_RATE : my_hparams['dropout_rate'],
        HP_LRDECAY_RATE : my_hparams['learning_rate_decay'],
        HP_RYTHM_RATE : my_hparams['rythm'],
        HP_ALPHA_ACTI : my_hparams['alpha_acti'],
        HP_BATCH_SIZE : my_hparams['batch_size'],
        HP_BATCHNORM : my_hparams['batch_normalisation'],
        HP_SEED : my_hparams['seed'],
        HP_OPTIMIZER : my_hparams['optimizer']
      }

hparams2 = {
        HP_ARCHITECTURE : HP_ARCHITECTURE.domain.values[0],
        HP_LEARNING_RATE : HP_LEARNING_RATE.domain.values[0],
        HP_DROPOUT_RATE : HP_DROPOUT_RATE.domain.values[0],
        HP_LRDECAY_RATE : HP_LRDECAY_RATE.domain.values[0],
        HP_RYTHM_RATE : HP_RYTHM_RATE.domain.values[0],
        HP_ALPHA_ACTI : HP_ALPHA_ACTI.domain.values[0],
        HP_BATCH_SIZE : HP_BATCH_SIZE.domain.values[0],
        HP_BATCHNORM : HP_BATCHNORM.domain.values[0],
        HP_SEED : HP_SEED.domain.values[0],
        HP_OPTIMIZER : HP_OPTIMIZER.domain.values[0]
      }



#et un autre pour train model

run_dir = 'logs/hparam_tuning/' + "archi1"
with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(logable_hparams)  # record the values used in this trial
    final_metrics = m.train_model(my_hparams, logable_hparams, prime_log_dir)
    tf.summary.scalar(
            'accuracy', 
            final_metrics[1],
            step=1 #ou  jsp
            )
    """for measure in final_metrics:
        tf.summary.scalar(
            measure, 
            final_metrics[measure],
            step=1 #ou  jsp
            )"""



run_dir = 'logs/hparam_tuning/' + "archi2"
logable_hparams[HP_ARCHITECTURE] = 1
my_hparams['architecture'] = hp_architecture[1]

logable_hparams[HP_LEARNING_RATE] = hp_learning_rate[1]
my_hparams['learning_rate'] = hp_learning_rate[1]

with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(logable_hparams)  # record the values used in this trial
    final_metrics = m.train_model(my_hparams,logable_hparams, prime_log_dir)
    tf.summary.scalar(
            'accuracy', 
            final_metrics[1],
            step=1 #ou  jsp
            )
    """for measure in final_metrics:
        tf.summary.scalar(
            measure, 
            final_metrics[measure],
            step=1 #ou  jsp
            )"""
        






