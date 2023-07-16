import pandas as pd
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('./sort.csv') 
x_train = train_data.iloc[:3, 1:6].values #inputs
y_train = train_data.iloc[:3, 7].values #target outputs


print(x_train)

scaler = StandardScaler()
xstd= scaler.fit_transform(x_train)

mean_value = scaler.mean_
std_deviation = scaler.scale_



new_data = x_train
new_data = scaler.transform(new_data)


print(xstd)
print(new_data)





def mean(l):
    m = 0
    for i in l:
        m += i
    return m/len(l)

def var(l):
    m = mean(l)
    sd = 0
    for i in l:
        sd += ((i-m)**2)
    return (sd/len(l))**0.5

def normStd(l):
    m = mean(l)
    sd = var(l)
    return [(i-m)/sd for i in l]

l3 = [i[3] for i in x_train]
print(l3)
print(normStd(l3))
print("mean fct", mean(l3))
print("mean", mean_value)

print("st fct", var(l3))

print("st", std_deviation)
