from pandas import read_csv
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

scaler = MinMaxScaler(feature_range = (0, 1))
variables = 5
timestep = 10
leadtime = 3
groups = ["Speed","LaneDistanceRight","Driving style","Along_s","Alat_s"]

dataset = read_csv('dataset.csv',usecols=groups)
scaler.fit(dataset)
dataset = read_csv('dataset.csv',usecols=["event"])
events = dataset.event.unique()

RMSE = []
MAE = []
R2 = []
TS = []

for iteration in range (0,1):
    train_e, test_e = train_test_split(events,test_size=0.3)
    features_set = []
    labels = []
    test_f = []
    test_l = []
    len_test = [0]
    cnt = 0
    length = []
    for event in train_e:
        # load dataset
        init_groups = groups+["event"]
        dataset = read_csv('dataset.csv' ,usecols=init_groups)
        dataset = dataset[dataset['event'] == event]
        # specify columns to plot
        # plot each column
        dataset = dataset.drop(columns=["event"])
        train = scaler.transform(dataset)
        for i in range(timestep,len(train)-leadtime):
            t =[]
            for j in range(i-timestep,i):
                temp = []
                for k in range(0,variables):
                    temp.append(train[j][k])
                t.append(temp)
            temp = []
            for k in range(0,variables):
                temp.append(train[i+leadtime][k])
            features_set.append(t)
            labels.append(temp)
    for event in test_e:
        # load dataset
        init_groups = groups+["event"]
        dataset = read_csv('dataset.csv' ,usecols=init_groups)
        dataset = dataset[dataset['event'] == event]
        # specify columns to plot
        # plot each column
        dataset = dataset.drop(columns=["event"])
        test = scaler.transform(dataset)
        length.append(len(test)-leadtime-timestep)
        cnt += len(test)-leadtime-timestep
        for i in range(timestep,len(test)-leadtime):
            t =[]
            for j in range(i-timestep,i):
                temp = []
                for k in range(0,variables):
                    temp.append(test[j][k])
                t.append(temp)
            temp = []
            for k in range(0,variables):
                temp.append(test[i+leadtime][k])
            test_f.append(t)
            test_l.append(temp)
        len_test.append(cnt)


    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], features_set.shape[2]))
    test_f, test_l = np.array(test_f), np.array(test_l)
    test_f = np.reshape(test_f, (test_f.shape[0], test_f.shape[1], test_f.shape[2]))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timestep, variables)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units = variables))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(features_set, labels, epochs = 100, batch_size = 32)

    predictions = model.predict(test_f)
    predictions = scaler.inverse_transform(predictions)
    test_l = scaler.inverse_transform(test_l)

    r2 = []
    mae = []
    rmse = []
    for i in range(0,variables):
        r2.append(r2_score(test_l[:,i], predictions[:,i]))
        mae.append(mean_absolute_error(test_l[:,i], predictions[:,i]))
        rmse.append(mean_squared_error(test_l[:,i], predictions[:,i], squared = False))
    R2.append(r2)
    MAE.append(mae)
    RMSE.append(rmse)


    shift = []
    for i in range(0,variables):
        temp_shift = []
        t = []
        for j in range(0,len(test_e)-1): 
            temp = []       
            temp.append(mean_absolute_error(test_l[len_test[j]:len_test[j+1],i], predictions[len_test[j]:len_test[j+1],i]))
            for k in range(1,leadtime+1):
                temp.append(mean_absolute_error(test_l[len_test[j]:len_test[j+1]-k,i], predictions[len_test[j]+k:len_test[j+1],i]))
            temp_shift.append(np.argmin(temp))
        shift.append(np.average(temp_shift))

    TS.append(shift)

    e = []
    for i in range(0,len(test_e)):
        for j in range(0,length[i]):
            e.append(test_e[i])

    name_dict = {
                'event': e,
                "T_Speed": test_l[:,0],
                "T_LaneDistanceRight": test_l[:,1],
                "T_Along_s": test_l[:,3],
                "T_Alat_s": test_l[:,4],
                "P_Speed": predictions[:,0],
                "P_LaneDistanceRight": predictions[:,1],
                "P_Along_s": predictions[:,3],
                "P_Alat_s": predictions[:,4]
            }
    df = pd.DataFrame(name_dict)
    df.to_csv('svm.csv')

    for j in range(0,5):
        pyplot.figure()
        for i in range(1,6):
            pyplot.subplot(5, 1, i)
            pyplot.plot(predictions[100*i-100:100*i,j])
            pyplot.plot(test_l[100*i-100:100*i,j])
        pyplot.show()

print("R2:\n",R2)
print("MAE:\n",MAE)
print("RMSE:\n",RMSE)
print("TS:\n",TS)

    
