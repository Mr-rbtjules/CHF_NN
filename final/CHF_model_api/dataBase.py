import CHF_model_api as CHF
import pandas
import numpy as np
import tabula as tb
from sklearn.preprocessing import StandardScaler
from CHF_model_api.config import TEST_DATA_PROPORTION, REMOVE_NEG_DHIN
import os
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
from typing import List



class MyDB:
    """This class purpose is to make object containing the data needed 
    for training and validation and making data quickly accessible.
    It can be vizualise as an object representation of the LUT table, 
    it will contain linear interpolation function of the LUT table """

    AVAILABLE_DB = []

    def __init__(
            self,
            seed:            int = 1,
            input_number:    int = 4,
            interpolation:   bool = False
    ) -> None:
        self.seed = seed
        self.input_number = input_number
        self.associeted_models = []#useful?
        self.data = self.loadData(seed, input_number)
        if interpolation:
            self.interp_func = self.getInterpFunction()
        MyDB.AVAILABLE_DB.append(self)

    def interpolate(self, X_list) -> float:
        """allow to use the LUT interpolated function
        input : [LD,P,G,Xchf]"""
        
        scaler = StandardScaler()
        scaler.mean_ = self.data['mean']
        scaler.scale_ = self.data['std']
        normalized_data = scaler.transform(X_list)
        return self.interp_func(normalized_data)

    def getInterpFunction(self) -> LinearNDInterpolator:
        """return the linear interpolator object/function"""
        print("Creation of the interpolation function of LUT")
        X_train = self.data['train_features']
        y_train = self.data['train_targets']
        fun = LinearNDInterpolator(
            X_train,
            y_train
        )
        return fun

    def extractSortFromPdf(self,path) -> pandas.DataFrame:
        """create a csv file based on Groeneveld 2006 LUT pdf
        take 2 min to run"""
        self.extraction(path)
        IS_data = self.ISUnitsTransformation()
        sort = self.filtration(IS_data)
        return sort

    def isCompatible(self, model: CHF.MyModel) -> bool:
        """return True if the database can be used for training a model
        based"""
        seed = model.hparams['data_seed']
        input_number = model.hparams['input_number']
        if self.seed == seed and self.input_number == input_number:
            return True
        return False

    def makeDictDatabase(
        self, 
        input_number:    int, 
        training_data:   pandas.DataFrame,
        validation_data: pandas.DataFrame
    ) -> dict:
        """select the desired data features tu put it in a dict """
        #inputs = X,L/D,P, G, DHin
        LD = training_data.iloc[:, 9].values
        Xchf = training_data.iloc[:, 5].values
        DH = training_data.iloc[:, 6].values
        P = training_data.iloc[:, 3].values
        G = training_data.iloc[:, 4].values

        LD_val = validation_data.iloc[:, 9].values
        Xchf_val = validation_data.iloc[:, 5].values
        DH_val = validation_data.iloc[:, 6].values
        P_val = validation_data.iloc[:, 3].values
        G_val = validation_data.iloc[:, 4].values

        if input_number == 5:
            #add up all the necessary colomns toegether
            X_train = np.column_stack((LD,P,G,Xchf,DH))
            X_val = np.column_stack((LD_val,P_val,G_val,Xchf_val,DH_val))
            
        elif input_number == 4:
            X_train = np.column_stack((LD,P,G,Xchf))
            X_val = np.column_stack((LD_val,P_val,G_val,Xchf_val))
            
        y_val = validation_data.iloc[:, 7].values
        y_train = training_data.iloc[:, 7].values
        data = {
            'validation_targets': y_val,
            'validation_features': X_val,
            'train_features': X_train,
            'train_targets': y_train,
            'mean': 0,
            'std': 0
        }
        return data


    def loadData(self, data_seed: int = 1, input_number: int = 5) -> dict:
        """take the data from a csv containing data in IS units
        or create it from the Groeneveld 2006 LUT pdf
        return a dict containing the keys:
        'validation_targets', 'validation_features','training_features'
        'training_targets', 'mean' 'std'(mean and std of the 
        training_features before normalization we need to keep when
        predicting) and is meant to be add to DATA[seed] = {'validation':...}"""
        try:
            print("Load data from csv")
            path = Path(CHF.config.CSV_DIR) / "sort_data.csv"
            data = pandas.read_csv(path) 
        except:
            print("No csv found, extraction from LUT.pdf")
            path = Path(CHF.config.PDF_DIR) / "LUT.pdf"
            data = self.extractSortFromPdf(path)                         
        # Stratified Sampling 

        if REMOVE_NEG_DHIN:
            print("Remove negative DHIN")
            data = data.loc[(data['DHin'] > 0)]

        validation_data = data.groupby('CHF').apply(
            lambda x: x.sample(frac=TEST_DATA_PROPORTION, random_state=data_seed)
        ).droplevel(0).sample(frac=1, random_state=(data_seed+10)) #+10 jut to be different than seed_ss
        training_data = data.drop(
            validation_data.index
        ).sample(frac=1, random_state=(data_seed+20))

        data = self.makeDictDatabase(
            input_number,
            training_data,
            validation_data
        )
        data = self.normalizeData(data)
        print(f"Data loaded seed: {data_seed}")
        return data

    def normalizeData(self, data: dict) -> dict:
        """take the dict with the data and return it
        normalized"""
        # normalisation std  only training features
        scaler = StandardScaler()
        data['train_features'] = scaler.fit_transform(
            data['train_features']
        )
        #then use the normalisation of the first set
        data['validation_features'] = scaler.transform(
            data['validation_features']
        )
        ##import to save bc use it when want to predict
        #array each value correspond mean of 1 features
        data['mean'] = scaler.mean_
        data['std'] = scaler.scale_
        return data


    def ISUnitsTransformation(self) -> pandas.DataFrame:
        """Convert the raw data in SI units and return """
        path = Path(CHF.config.CSV_DIR) / "original_data.csv"
        sort = pandas.read_csv(path)
        #SORT
        #to SI units
        sort['P'] = sort['P']*1000
        sort['DHin'] = sort['DHin']*1000
        sort['CHF'] = sort['CHF']*1000
        sort['Tin'] = sort['Tin'] + 273.15
        return sort
    
    def extraction(self, path) -> None:
        """extract the data from pdf and put it in original_data.csv
        take 2 min to run"""
        all = tb.read_pdf(path, pages='all')
        ##start with the first page containing the header
        #remove units on the firs line
        df = all[0].drop(0)
        #remove unuseful colomns
        del df['Number']
        del df['Data']
        del df['Reference']

        #columns of interest to float
        columns = ['D', 'L', 'P', 'G','Xchf', 'DHin', 'CHF', 'Tin']
        for col in columns:
            df[col] = df[col].astype(float)
        ##other pages
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
                            #some error in the table with space and 0
                            #need to be transformed as .00
                            if rest[i][j][-2:] == ' 0':
                                rest[i][j] = rest[i][j][:-2]
                            rest[i][j] = float(rest[i][j])
                np_data_page = np.concatenate((first_row, rest))
                small_df = pandas.DataFrame(np_data_page, columns =columns)
                df = pandas.concat([df,small_df], ignore_index=True)
        #save
        output_path = Path(CHF.config.CSV_DIR) / "original_data.csv"
        df.to_csv(output_path, index=False)
        return None

    def filtration(self, sort: pandas.DataFrame) -> pandas.DataFrame:
        """The dataframe is filtered in order to kick outliers and
        nonsense values"""
        sort = sort.loc[sort['CHF'] > 0]
        sort = sort.loc[sort['Xchf'] < 1 ]
        sort = sort.loc[(sort['P'] <= 21000000) &  (sort['P'] >= 100000)]
        sort = sort.loc[(sort['D'] < 0.025) &  (sort['D'] > 0.003) ]
        sort = sort.loc[(sort['G'] < 8000) &  (sort['G'] >= 0) ]

        #add a knew colomn
        sort['L/D'] = sort['L']/sort['D'] 
        sort = sort.loc[ ((sort['Xchf'] > 0) & ( sort['L/D']> 50))  
                        |  ((sort['Xchf'] < 0) & ( sort['L/D'] > 25))]
        #save
        path = Path(CHF.config.CSV_DIR) / "sort_data.csv"
        sort.to_csv(path)
        data = pandas.read_csv(path) 
        return data

    
    def getLUTPerformances(self) -> None:
        """print lut performances"""
        y_val = self.data['validation_targets']
        X_val = self.data['validation_features']
        X_train = self.data['train_features']
        y_train = self.data['train_targets']

        predicted_y_val = interpolate.griddata(X_train, y_train, X_val, method='linear')
        #other = LinearNDInterpolator(X_train,y_train)
        
        
        # Identify non-nan indices (i.e., points inside the convex hull)
        inside_hull_indices = ~np.isnan(predicted_y_val)

        # Filter out the points outside the convex hull
        filtered_predicted_y_val = predicted_y_val[inside_hull_indices]
        filtered_y_val = y_val[inside_hull_indices]

        percentage_errors = ((filtered_y_val - filtered_predicted_y_val) / filtered_y_val) * 100

        # Compute the Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs(percentage_errors))

        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

        if filtered_predicted_y_val.all() != 0:
            mean_MP = np.mean(filtered_y_val/filtered_predicted_y_val) 
            print(mean_MP)      
        print('stdMp : ',CHF.tools.stdMP(filtered_y_val, filtered_predicted_y_val))
        print('nrmse ',CHF.tools.nrmse(filtered_y_val, filtered_predicted_y_val))
        print('msle ',CHF.tools.myMsle(filtered_y_val, filtered_predicted_y_val)) 
        CHF.tools.plotResults(filtered_predicted_y_val, filtered_y_val)
            
    @classmethod
    def getAvailableDataBases(cls) -> List['MyDB']:
        return cls.AVAILABLE_DB


