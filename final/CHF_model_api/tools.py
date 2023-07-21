import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tabula as tb
from sklearn.preprocessing import StandardScaler

from CHF_model_api.config import TEST_DATA_PROPORTION


import json
import os




###JSON hparams###
def get_hparams_saved_model(model_name: str) -> dict:

    path = "./saved_models/hparams/" + model_name + ".json"
    with open(path, 'r') as file:
        return json.load(file)


def save_hparams(model_name: str, hparams: dict) -> None:

    path = "./saved_models/hparams/" + model_name + ".json"
    with open(path, 'w') as file:
        json.dump(hparams, file)
    return None

def erase_hparams(model_name: str) -> None:

    path = "./saved_models/hparams/" + model_name + ".json"
    try:
        os.remove(path)
    except:
        print("File not found to supress")
        
    return None

###Load DATA###

def extract_from_pdf(path):
    """
    create a csv file based on Groeneveld 2006 LUT pdf
    take 2 min to run
    """
    all = tb.read_pdf(path, pages='all')

    
    ##start with the first page containing the header
    #remove units
    df = all[0].drop(0)
    #remove unuseful units
    del df['Number']
    del df['Data']
    del df['Reference']


    #columns of interest to float
    columns = ['D', 'L', 'P', 'G','Xchf', 'DHin', 'CHF', 'Tin']
    for col in columns:
        df[col] = df[col].astype(float)


    #other pages
    for page in range(1,len(all)):
        #if 1 header missing we don't take the page
        if len(all[page].keys()) < 11:
            pass
        else:
            #first line interpreted as header by tabula but 
            # it is actual data, we transform it
            header = all[page].keys()
            first_row = [[]]
            for i in header[2:10]:
                if i[-2:] == ' 0':
                    i = i[:-2]
                first_row[0].append(float(i))
            rest = all[page].iloc[:,2:10].values
            for i in range(len(rest)):
                for j in range(len(rest[i])):
                    if type(rest[i][j]) == str:
                        if rest[i][j][-2:] == ' 0':
                            rest[i][j] = rest[i][j][:-2]
                        rest[i][j] = float(rest[i][j])
            np_data_page = np.concatenate((first_row, rest))
            small_df = pd.DataFrame(np_data_page, columns =columns)
            df = pd.concat([df,small_df], ignore_index=True)

    #save
    df.to_csv('./csv_files/original_data.csv', index=False)

    sort = pd.read_csv('./csv_files/original_data.csv')

    #SORT
    #to SI units
    sort['P'] = sort['P']*1000
    sort['DHin'] = sort['DHin']*1000
    sort['CHF'] = sort['CHF']*1000
    sort['Tin'] = sort['Tin'] + 273.15


    #keep physical values
    sort = sort.loc[sort['Xchf'] < 1 ]
    sort = sort.loc[(sort['P'] <= 21000000) &  (sort['P'] >= 100000)]
    sort = sort.loc[(sort['D'] < 0.025) &  (sort['D'] > 0.003) ]
    sort = sort.loc[(sort['G'] < 8000) &  (sort['G'] >= 0) ]

    #add a knew colomn
    sort['L/D'] = sort['L']/sort['D'] 
    sort = sort.loc[ ((sort['Xchf'] > 0) & ( sort['L/D']> 50))  
                    |  ((sort['Xchf'] < 0) & ( sort['L/D'] > 25))]

    #save
    sort.to_csv('./csv_files/sort_data.csv')
    
    data = pd.read_csv('./csv_files/sort_data.csv') 
    return data


def load_data(data_seed: int = 1) -> dict:
    """
    take the data from a csv containing data SI units
    or create it from the Groeneveld 2006 LUT pdf

    return a dict containing the keys:
    'validation_targets', 'validation_features','training_features'
    'training_targets', 'mean' 'std'(mean and std of the training_features before
     normalization we need to keep when predicting)
     and is meant to be add to DATA[seed] = {'validation':...}
    """

    try:
        print("Load data from csv")
        data = pd.read_csv('../csv_files/sort_data.csv') 
    except:
        print("No csv found, extraction from LUT.pdf")
        data = extract_from_pdf('../pdf_files/LUT.pdf')

    # Stratified Sampling 
    validation_data = data.groupby('CHF').apply(
        lambda x: x.sample(frac=TEST_DATA_PROPORTION, random_state=data_seed)
    ).droplevel(0).sample(frac=1, random_state=(data_seed+10)) #+10 jut to be different than seed_ss


    training_data = data.drop(
        validation_data.index
    ).sample(frac=1, random_state=(data_seed+20))


    #inputs = X,L/D,P, G, DHin
    LD = training_data.iloc[:, 9].values
    Xchf = training_data.iloc[:, 5].values
    DH = training_data.iloc[:, 6].values
    P = training_data.iloc[:, 3].values
    G = training_data.iloc[:, 4].values
    
    X_train = np.column_stack((LD,P,G,Xchf,DH))
    y_train = training_data.iloc[:, 7].values

    LD_val = validation_data.iloc[:, 9].values
    Xchf_val = validation_data.iloc[:, 5].values
    DH_val = validation_data.iloc[:, 6].values
    P_val = validation_data.iloc[:, 3].values
    G_val = validation_data.iloc[:, 4].values
    
    X_val = np.column_stack((LD_val,P_val,G_val,Xchf_val,DH_val))
    y_val = validation_data.iloc[:, 7].values


    # normalisation std  only training features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #then use the normalisation of the first set
    X_val = scaler.transform(X_val)
    ##import to save bc use it when want to predict
    #array each value correspond mean of 1 features
    mean_value = scaler.mean_
    std_deviation = scaler.scale_
    data = {
        'validation_targets': y_val,
        'validation_features': X_val,
        'train_features': X_train,
        'train_targets': y_train,
        'mean': mean_value,
        'std': std_deviation
    }
    print("Data loaded")
    return data


###VIZUALIZATION TOOLS###

def utils_nn_config(model):
    lst_layers = []
    if "Sequential" in str(model): #-> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({"name":"input", "in":int(layer.input.shape[-1]), "neurons":0, 
                           "out":int(layer.input.shape[-1]), "activation":None,
                           "params":0, "bias":0})
    for layer in model.layers:
        try:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
                         "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
                         "params":layer.get_weights()[0], "bias":layer.get_weights()[1]}
        except:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
                         "out":int(layer.output.shape[-1]), "activation":None,
                         "params":0, "bias":0}
        lst_layers.append(dic_layer)
    return lst_layers


'''
Plot the structure of a keras neural network.
'''
def visualize_nn(model, name, description=False, figsize=(10,8)):
    ## get layers info
    lst_layers = utils_nn_config(model)
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
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
            else:
                plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
                plt.text(x=left+d*x_space, y=top-p, fontsize=10, color=color, s=layer['activation']+" (")
                plt.text(x=left+d*x_space, y=top-2*p, fontsize=10, color=color, s="Σ"+str(layer['in'])+"[X*w]+b")
                out = " Y"  if i == len(layer_sizes)-1 else " out"
                plt.text(x=left+d*x_space, y=top-3*p, fontsize=10, color=color, s=") = "+str(layer['neurons'])+out)
        
        ### circles
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
            elif i == len(layer_sizes)-1:
                plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
            else:
                plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
    
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i+1]
        color = "green" if i == len(layer_sizes)-2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], 
                                  [layer_top_a-m*y_space, layer_top_b-o*y_space], 
                                  c=color, alpha=0.5)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    path = './visuals/' + name +'.png'
    plt.savefig(path)
    print("Vizualisation saved in visuals directory")
    #plt.show()
#visualize_nn(model, description=True, figsize=(100,100))
    
def remove_directory_content(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            os.rmdir(dir_path)

def reset_directories():
    remove_directory_content("./logs")
    remove_directory_content("./saved_models/models")
    remove_directory_content("./saved_models/hparams")
    

def nrmse(y_true,y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))/np.mean(y_true)


def std_MP(y_val, predictions):
    MP = y_val/predictions
    std = 0
    for i in MP:
        std += (1-i)**2
    return np.sqrt(std/len(MP))

