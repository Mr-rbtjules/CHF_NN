import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential #regroupement de layer formant le modele
#dense == tensor ou layer ou ensemble de neurons d'un mm niveau 
 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization #layer instance

#import la fct elu pour pouvoir gerer HP alpha
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import StandardScaler


###HYPERPARAM###
# type d'archi qu'on veut 
architecture = [5, 39, 40, 31, 42, 1]#small
#big = "[5,61,51,28,39,26,2120,14,1]"

dropoutRate = 0.2
learningRate = 0.001
testPercent = 0.2
maxEpochs = 500 #pas trop important car peut de chance d'overfitt
thresholdActi = 0
alphaActi = 0 #! >=0  = pente pour tt x en dessous de thresold
clipGrad = 1

### ###

# met la LUT
train_data = pd.read_csv('./sort.csv') 

x_train = train_data.iloc[:, 1:6].values #inputs
y_train = train_data.iloc[:, 7].values #target outputs

# normalisation std
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

#create the model
model = Sequential()

# Add the layers
for layer, neurons in enumerate(architecture):
    model.add(Dense(neurons, activation=relu(alpha=alphaActi, threshold=thresholdActi)))       #relu ?= 'relu' pt comme ça on peut ajouter des param a relu ?
    # BatchNormalization + dropout regularization
    if layer == 1 or layer == 2:
        #add droupout to the previous layer(last one on the top)
        model.add(Dropout(dropoutRate))  
    if layer != 0 and layer != len(architecture) - 1: #pr tt sauf premier dernier
        model.add(BatchNormalization())

# tt les 32 epochs lr perd 4%
#schedule == function to give to LearningScheduler, with epoch and lr as param
def lr_scheduler(epoch, lr):          #to modify and/or add to HP   
    if epoch % 32 == 0 and epoch > 0:
        return lr * 0.96
    return lr

# choix de l'optimizer, de son rate et de la loss function
model.compile(optimizer=Adam(learning_rate=learningRate), 
              loss=MeanSquaredLogarithmicError())


#Callback == modification durant l'entrainement
#on stop qd plus de chgmt apres 50 epoch
early_stop = EarlyStopping(min_delta=0,             #maybe change
                           patience=50, verbose=1,
                           restore_best_weights=True) #when stop ensure we keep best perf
learningRateScheduler = LearningRateScheduler(lr_scheduler)


# on entraine en prenant que 20% pour les test
#a partir d'ici que les weigghts sont crée 
history = model.fit(x_train, y_train, 
                    validation_split=testPercent, 
                    batch_size=32, epochs=maxEpochs,
                    callbacks=[early_stop, learningRateScheduler],
                    verbose=1)                                    #progress bar

# resumé
model.summary()
