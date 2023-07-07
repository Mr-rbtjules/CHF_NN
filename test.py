###INIT MODEL###


model = Sequential()
# Add the layers

for layer, neurons in enumerate(architecture):
    
    if layer != len(architecture)-1:
        model.add(
        Dense(
            neurons, 
            activation=actMethod,#(alpha=alphaActi, threshold=thresholdActi), 
            input_shape=(neurons,)
            )
        )
    if layer == len(architecture) -1:
        Dense(
            neurons, 
            activation='linear', 
            input_shape=(neurons,)
            )
        
    if layer == 1 or layer == 2:
        model.add(Dropout(dropoutRate))
    if layer != 0 and layer != len(architecture) - 1:
        model.add(BatchNormalization())



model.compile(optimizer=Adam(learning_rate=learningRate), 
              loss=MeanSquaredLogarithmicError())


print(maxEpochs)