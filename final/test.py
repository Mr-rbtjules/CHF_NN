import os
import shutil

path = './hparams_tuning_tb/123'
#os.makedirs(path)
name = '20230720-193121'

source_val_path = f'./logs/hparam_tuning/{name}/validation'
source_train_path = f'./logs/hparam_tuning/{name}/train'
destination_val_path = path + '/validation' 
destination_train_path = path + '/train'



val_logs = os.listdir(source_val_path)
# Filter the list to get only regular files (not directories)

for log in val_logs:
    if os.path.isfile(os.path.join(source_val_path, log)):
    # Construct the full paths for the source and destination files
        source_file_path = os.path.join(source_val_path, log)
        destination_file_path = os.path.join(destination_val_path, log)
        # Copy the file to the destination directory
        shutil.copy(source_file_path, destination_file_path)


"""def my_decorator(target_function):
    def wrapper_function(self, *args, **kwargs):
        print("Decorator: Accessing attribute 'attribute_name':", self.attribute_name)
        result = target_function(self, *args, **kwargs)
        return result
    return wrapper_function

class MyClass:
    def __init__(self, value):
        self.attribute_name = value

    @my_decorator
    def my_method(self, jsp):
        jsp+=1
        print("Inside my_method")

obj = MyClass(42)
obj.my_method(3)


import threading

class TB:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
"""
#within the model class
"""with self.shared_attributeTBobject.lock:
            # Perform the critical section of code with shared_attribute
            self.shared_attribute.value = new_value"""
