from tensorflow.keras.models import load_model

# Load the saved model
#!!! si custom metrics we need to save and load in 
#a custom contexte 
loaded_model = load_model('my_model.h5')