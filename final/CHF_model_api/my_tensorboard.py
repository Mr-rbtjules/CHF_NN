"""On va faire en sorte que on puisse donner
une liste de modele sauvegarder et faire leur
tb 
et aussi creer un decorator qui s'applique
que dans la classe Optimizer
"""

#check metrics and how to record it after
#or just make a decorator link with the tbclass

from typing import My_model, List

class My_tensorboard:

    def __init__(
            self,
            saved_models_name:    str
    ) -> None:
        
        self.models_name = saved_models_name
        self.models = self.init_models()
        self.learning_rates = []
        self.architectures = []
        self.dropout_rates = []
        self.load_tb()

    def load_tb(self):
        #do all the work
        #set up directories(session name ?)
        #move logs
        #make_Hparam_range
        #log_Hparam_range
        #log_Hparam

        return None

    def make_Hparams_range(self):
        return None
    
    def log_Hparams_range(self):
        return None
    
    def log_Hparams(self):
        return None

    def set_up_directories(self):
        #all models names train and validation
        return None

    def init_models(self)-> List[My_model]:

        models = []
        for name in self.models_name:
            pass
        return models
    
    def add_saved_model(self, model_name: str) -> None:
        return None
    
    def add_model(self, model: My_model)-> None:
        #move_log
        return None
    
    def add_hparams(self):
        return None
    
    def move_logs(self, names):
        return None
    def move_log(self, name):
        return None
    
    
    
    #question, est ce que j'ai juste a lancer une session puis recharge 
    #qd nouvaux modeles ajoutes ?
