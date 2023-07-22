"""On va faire en sorte que on puisse donner
une liste de modele sauvegarder et faire leur
tb 
et aussi creer un decorator qui s'applique
que dans la classe Optimizer
"""

#check metrics and how to record it after
#or just make a decorator link with the tbclass


"""
def tensorBoard_log(train_function):
    def wrapper(*args, **kwargs):
        
        result = train_function(*args, **kwargs)
        return result
    return wrapper
"""

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


from datetime import datetime
import os
import shutil
from typing import List

import CHF_model_api as CHF


class My_tensorboard:

    def __init__(
            self,
            saved_models_name:    List[str]
    ) -> None:
        
        self.models_name = saved_models_name
        self.models = self.init_models()
        self.learning_rates = []
        self.architectures = []
        self.dropout_rates = []
        self.session_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_path = f'./hparams_tuning_tb/{self.session_name}'

        self.HP_architecture = None
        self.HP_learing_rate = None
        self.HP_dropout_rate = None
        self.metrics = None
        
        self.load_tb()

    def load_tb(self) -> None:
        #do all the work
        self.set_up_directories()
        self.copy_logs()
        self.make_Hparams_range()
        self.log_Hparams_range()
        self.log_Hparams()
        print(f"Type this in terminal in final directory to run tensorboard: tensorboard --logdir {self.session_path}")
        #just write tensorboard --logdir hparams_tuning_tb/session_name
        #in terminal
        return None



    def make_Hparams_range(self) -> None:
        archi = []
        lr = []
        dp = []
        for model in self.models:
            #str(list) because doesn't accept list
            archi.append(str(model.hparams['architecture']))
            lr.append(model.hparams['learning_rate'])
            dp.append(model.hparams['dropout_rate'])
        
        self.HP_architecture = hp.HParam('architecture',
                              hp.Discrete(archi))
        self.HP_learing_rate = hp.HParam('learning_rate', 
                             hp.Discrete(lr))
        self.HP_dropout_rate = hp.HParam('drouptout_rate',
                            hp.Discrete(dp))
        

        self.metrics = [hp.Metric('accuracy', display_name='Accuracy')]
        return None
    
    def log_Hparams_range(self) -> None:
        #log Hparam_range
        with tf.summary.create_file_writer(self.session_path).as_default():
            hp.hparams_config(
                hparams=[
                    self.HP_architecture,
                    self.HP_dropout_rate,
                    self.HP_learing_rate
                    ],
                metrics=self.metrics
            )
        return None
    
    

    def log_Hparams(self) -> None:
        for model in self.models:
            logable_hparams = {
                self.HP_architecture: str(model.hparams['architecture']),
                self.HP_dropout_rate: model.hparams['dropout_rate'],
                self.HP_learing_rate: model.hparams['learning_rate']
            }
            metric = model.hparams['mpe']

            path = self.session_path + f'/{model.name}'
            with tf.summary.create_file_writer(path).as_default():
                hp.hparams(logable_hparams)  # record the values used in this trial
                tf.summary.scalar(
                        'accuracy', 
                        metric,
                        step=1 #ou  jsp
                        )

        return None

    def set_up_directories(self) -> None:
        """create validation and training directory and also
         a directory for each different model (with diff hp) """
        os.makedirs(self.session_path)
        for name in self.models_name:
            os.makedirs(self.session_path + f'/{name}')
        
        os.makedirs(self.session_path + '/train')
        os.makedirs(self.session_path + '/validation')
        return None

    def init_models(self)-> List[CHF.My_model]:
        """get the saved models by creating object my_models
        based on the name of the saved models"""
        models = []
        for name in self.models_name:
            models.append(CHF.My_model(model_name=name))
        return models
    
    def add_saved_model(self, model_name: str) -> None:
        return None
    
    def add_model(self, model)-> None:
        #move_log
        return None
    
    def add_hparams(self):
        return None
    
    def copy_logs(self) -> None:
        """get the logs of all the models to compare"""
        for name in self.models_name:
            self.copy_log(name)
        return None
    

    def copy_log(self, name) -> None:
        """copy logs in ./logs/modelname/train et /validation
        to the same directories in ./hparams_tuning/sessionname"""
        source_val_path = f'./logs/{name}/validation'
        source_train_path = f'./logs/{name}/train'
        destination_val_path = self.session_path + '/validation' 
        destination_train_path = self.session_path + '/train'

        val_logs = os.listdir(source_val_path)
        # Filter the list to get only regular files (not directories)
        for log in val_logs:
            if os.path.isfile(os.path.join(source_val_path, log)):
            # Construct the full paths for the source and destination files
                source_file_path = os.path.join(source_val_path, log)
                destination_file_path = os.path.join(destination_val_path, log)
                # Copy the file to the destination directory
                shutil.copy(source_file_path, destination_file_path)
        
        train_logs = os.listdir(source_train_path)
        for log in train_logs:
            if os.path.isfile(os.path.join(source_train_path, log)):
            # Construct the full paths for the source and destination files
                source_file_path = os.path.join(source_train_path, log)
                destination_file_path = os.path.join(destination_train_path, log)
                # Copy the file to the destination directory
                shutil.copy(source_file_path, destination_file_path)

        return None
    
    
    
    #question, est ce que j'ai juste a lancer une session puis recharge 
    #qd nouvaux modeles ajoutes ?
