import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential #regroupement de layer formant le modele
#dense == tensor ou layer ou ensemble de neurons d'un mm niveau 
 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization #layer instance

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

### ###

# met la LUT
train_data = pd.read_csv('train_data.csv')  

# split inputs et target output
X_train = train_data.iloc[:, :5].values #inputs
y_train = train_data.iloc[:, 5].values #target outputs

# normalisation std
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#create the model
model = Sequential()

# Add the layers
for layer, neurons in enumerate(architecture):
    model.add(Dense(neurons, activation=relu))       #relu ?= 'relu' 
    # BatchNormalization + dropout regularization
    if layer == 1 or layer == 2:
        #add droupout to the previous layer(last one on the top)
        model.add(Dropout(dropoutRate))  
    if layer != 0 and layer != len(architecture) - 1: #pr tt sauf premier dernier
        model.add(BatchNormalization())

# tt les 32 epochs lr perd 4%
def lr_scheduler(epoch, lr):
    if epoch % 32 == 0 and epoch > 0:
        return lr * 0.96
    return lr

# choix de l'optimizer, de son rate et de la loss function
model.compile(optimizer=Adam(learning_rate=learningRate), 
              loss=MeanSquaredLogarithmicError())

#on stop qd plus de chgmt apres 50 epoch
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=50, verbose=1)

# on entraine en prenant que 20% pour les test
history = model.fit(X_train, y_train, 
                    validation_split=testPercent, 
                    batch_size=32, epochs=maxEpochs,
                    callbacks=[early_stop, LearningRateScheduler(lr_scheduler)])

# resumé
model.summary()
