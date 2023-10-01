import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy import interpolate
import CHF_model_api as CHF
from pathlib import Path

###JSON hparams###
def getHparamsSavedModel(model_name: str) -> dict:                          
    """return a dict with all the hyperparameters saved
    in the directory hparams in saved_models"""
    path = (Path(CHF.config.MODELS_DIR) / 
            f"hparams/{model_name}.json")
    
    with open(path, "r") as file:
        return json.load(file)


def saveHparams(model_name: str, hparams: dict) -> None:
    """save the dict containing results and hyperparameters 
    in hparams idrectory"""
    path =(Path(CHF.config.MODELS_DIR) / 
            f"hparams/{model_name}.json")
    with open(path, "w") as file:
        json.dump(hparams, file)
    return None

def eraseHparams(model_name: str) -> None:
    """Erase stored hyperparameters"""
    path = (Path(CHF.config.MODELS_DIR) / 
            f"hparams/{model_name}.json")
    try:
        os.remove(path)
    except:
        print("File not found to supress")
        
    return None


###VIZUALIZATION TOOLS###

def plotResults(
        predictions: list, 
        y_val: list, 
        save_fig: bool=False
) -> None:
    """Plot the the graph of the predicted value in functino
    of the measured values"""
    plt.figure()

    plt.plot(y_val, predictions, '.r', label='')
    plt.plot(y_val, y_val, 'b-', label='y=x')
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Measured')
    if save_fig:
        path = (Path(CHF.config.VISU_DIR) / 'test_plot_results.png')
        plt.savefig(path)
    plt.show()
    return None

def utilsNnConfig(model) :
    lst_layers = []
    if "Sequential" in str(model): #-> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({
            "name":"input", 
            "in":int(layer.input.shape[-1]), 
            "neurons":0, 
            "out":int(layer.input.shape[-1]),
            "activation":None,
            "params":0, "bias":0
        })
    for layer in model.layers:
        try:
            dic_layer = {
                "name":layer.name, 
                "in":int(layer.input.shape[-1]), 
                "neurons":layer.units, 
                "out":int(layer.output.shape[-1]), 
                "activation":layer.get_config()["activation"],
                "params":layer.get_weights()[0], "bias":layer.get_weights()[1]
            }
        except:
            dic_layer = {
                "name":layer.name, 
                "in":int(layer.input.shape[-1]), 
                "neurons":0, 
                "out":int(layer.output.shape[-1]), 
                "activation":None,
                "params":0, "bias":0
            }
        lst_layers.append(dic_layer)
    return lst_layers





def visualizeNn(model, name, description=False, figsize=(10,8)):
    """Plot the structure of a keras neural network."""
    ## get layers info
    lst_layers = utilsNnConfig(model)
    layer_sizes = [layer["out"] for layer in lst_layers]
    ## fig setup
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    ## nodes
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        layer = lst_layers[i]
        color = "green" if i in [0, len(layer_sizes)-1] else "blue"
        color = "red" if (layer['neurons'] == 0) and (i > 0) else color
        
        ### add description
        if (description is True):
            d = i if i == 0 else i-0.5
            if layer['activation'] is None:
                plt.text(
                    x=left+d*x_space, 
                    y=top, 
                    fontsize=10, 
                    color=color, 
                    s=layer["name"].upper()
                )
            else:
                plt.text(
                    x=left+d*x_space, 
                    y=top, fontsize=10, 
                    color=color, 
                    s=layer["name"].upper()
                )
                plt.text(
                    x=left+d*x_space,
                    y=top-p, 
                    fontsize=10, 
                    color=color, 
                    s=layer['activation']+" ("
                )
                plt.text(
                    x=left+d*x_space, 
                    y=top-2*p, 
                    fontsize=10, 
                    color=color, 
                    s="Σ"+str(layer['in'])+"[X*w]+b"
                    )
                out = " Y"  if i == len(layer_sizes)-1 else " out"
                plt.text(
                    x=left+d*x_space, 
                    y=top-3*p, 
                    fontsize=10, 
                    color=color, 
                    s=") = "+str(layer['neurons'])+out
                )
        ### circles
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(
                xy=(left+i*x_space, top_on_layer-m*y_space-4*p), 
                radius=y_space/4.0, 
                color=color, 
                ec='k', 
                zorder=4
            )
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                plt.text(
                    x=left-4*p,
                    y=top_on_layer-m*y_space-4*p, 
                    fontsize=10, 
                    s=r'$X_{'+str(m+1)+'}$'
                )
            elif i == len(layer_sizes)-1:
                plt.text(
                    x=right+4*p, 
                    y=top_on_layer-m*y_space-4*p, 
                    fontsize=10, 
                    s=r'$y_{'+str(m+1)+'}$'
                    )
            else:
                plt.text(
                    x=left+i*x_space+p, 
                    y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, 
                    fontsize=10, s=r'$H_{'+str(m+1)+'}$'
                )
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i+1]
        color = "green" if i == len(layer_sizes)-2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D(
                    [i*x_space+left, (i+1)*x_space+left], 
                    [layer_top_a-m*y_space, layer_top_b-o*y_space], 
                    c=color, 
                    alpha=0.5
                )
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    path = Path(CHF.config.VISU_DIR) / f"{name}.png"
    plt.savefig(path)
    print("Vizualisation saved in visuals directory")
    #plt.show()
#visualize_nn(model, description=True, figsize=(100,100))
    
def RemoveDirectoryContent(directory_path) -> None:
    """just a function to clean when we want to reset"""
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.rmdir(dir_path)

def remove_backups(name) -> None:

    model_path = Path(CHF.config.MODELS_DIR) / f"models/{name}.h5"
    hparams_path = Path(CHF.config.MODELS_DIR) / f"hparams/{name}.json"
    os.remove(model_path)
    os.remove(hparams_path)
    return None

def reset_directories() -> None:

    RemoveDirectoryContent(CHF.config.LOGS_DIR)
    path = Path(CHF.config.MODELS_DIR) / "models"
    RemoveDirectoryContent(path)
    path = Path(CHF.config.MODELS_DIR) / "hparams"
    RemoveDirectoryContent(path)
    RemoveDirectoryContent(CHF.config.HPTB_DIR)
    return None


###METRICS###

def nrmse(y_true,y_pred) -> float:
    """Compute normalised root mean suqared error"""
    res = None
    if np.mean(y_true) != 0:
        res = np.sqrt(np.mean(np.square(y_pred - y_true)))/np.mean(y_true)
    else:
        res = None
    return res

def myMsle(y_true, y_pred) -> float:
    """Compute mean squared logaritmic error"""
    return np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)


def stdMP(y_val, predictions) -> float:
    """compute the standart deviation of the ration 
    Measured / predict from 1"""
    res = None
    if predictions.all() != 0:
        MP = y_val/predictions
        std = 0
        for i in MP:
            std += (1-i)**2
        res = np.sqrt(std/len(MP))                                             
    else:
        res = None
    return res


