import pandas as pd



pd.options.display.max_rows = 10 #print le debut et la fin pour avoir un visu
col = ['D','L','P','G','X','DH', 'CHF','T']

df = pd.read_csv('../netLUT.csv')




#unités SI
df['P'] = df['P']*1000
df['DH'] = df['DH']*1000
df['CHF'] = df['CHF']*1000
df['T'] = df['T'] + 273.15





###FILTRE###
df = df.loc[df['X'] < 1 ]
df = df.loc[(df['P'] <= 21000000) &  (df['P'] >= 100000) ]
df = df.loc[(df['D'] < 0.025) &  (df['D'] > 0.003) ]
df = df.loc[(df['G'] < 8000) &  (df['G'] >= 0) ]



#à check
df['L/D'] = df['L']/df['D'] 
df = df.loc[ ((df['X'] > 0) & ( df['L/D']> 50))  |  ((df['X'] < 0) & ( df['L/D'] > 25))]



###SPLIT###
"""
dfTrain = df.sample(frac=0.8)
dfTest = df.drop(dfTrain.index)

dfTrain.to_csv("train.csv")
dfTest.to_csv("test.csv")

"""


print("sort SI LUT\n", df)


df.to_csv('test.csv')
