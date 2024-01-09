from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import metrics, initializers

#normalize data within range 0 to 1
scaler = MinMaxScaler(feature_range = (0, 1))

#parameters
variables = 7
timestep = 10
leadtime = 5

groups = ["Range","RangeRate","Speed","Driving style","LaneDistanceLeft","Along_s","Alat_s"]
Accuracy = []
Sensitivity = []
Specificity = []

dataset = read_csv('phase1.csv',usecols=groups)
scaler.fit(dataset)
dataset = read_csv('phase1.csv',usecols=["event"])
events = dataset.event.unique()

for iteration in range(0,10):
    #cross validation 10 times random train/test split
    train_e, test_e = train_test_split(events,test_size=0.3)
    features_set = []
    labels = []
    test_f = []
    test_l = []
    for event in train_e:
        # load dataset
        init_groups = groups+["event"]
        dataset = read_csv('phase1.csv' ,usecols=init_groups)
        dataset = dataset[dataset['event'] == event]
        dataset = dataset.drop(columns=["event"])
        labelset = read_csv('phase1.csv' ,usecols=["event","Crossing"])
        labelset = labelset[labelset['event'] == event]
        labelset = labelset.drop(columns=["event"])
        labelset = labelset.reset_index()
        train = dataset
        train = scaler.transform(train)
        #compose each train input with given timestep and lead time
        for i in range(timestep,len(train)-leadtime):
            t =[]
            for j in range(i-timestep,i):
                temp = []
                for k in range(0,variables):
                    temp.append(train[j][k])
                t.append(temp)
            temp = []
            temp.append(labelset["Crossing"][i+leadtime])
            features_set.append(t)
            labels.append(temp)

    #repeat for test events
    for event in test_e:
        # load dataset
        init_groups = groups+["event"]
        dataset = read_csv('phase1.csv' ,usecols=init_groups)
        dataset = dataset[dataset['event'] == event]
        dataset = dataset.drop(columns=["event"])
        labelset = read_csv('phase1.csv' ,usecols=["event","Crossing"])
        labelset = labelset[labelset['event'] == event]
        labelset = labelset.drop(columns=["event"])
        labelset = labelset.reset_index()
        test = dataset
        test = scaler.transform(test)
        for i in range(timestep,len(test)-leadtime):
            t =[]
            for j in range(i-timestep,i):
                temp = []
                for k in range(0,variables):
                    temp.append(test[j][k])
                t.append(temp)
            temp = []
            temp.append(labelset["Crossing"][i+leadtime])
            test_f.append(t)
            test_l.append(temp)

    #standardize test and train data structures into array
    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], features_set.shape[2]))
    test_f, test_l = np.array(test_f), np.array(test_l)
    test_f = np.reshape(test_f, (test_f.shape[0], test_f.shape[1], test_f.shape[2]))

    #features_set, test_f,labels, test_l = train_test_split(features_set,labels,test_size=0.3)
    l = np.reshape(labels,(labels.shape[0]))
    neg, pos = np.bincount(l)
    total = neg+pos
    print(neg, pos)

    #calculate class weight for positive and negative labels
    initial_bias = np.log([pos/neg])
    output_bias = initializers.Constant(initial_bias)
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    #model training and test label prediction
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timestep, variables)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1, activation='sigmoid', bias_initializer=output_bias))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    model.fit(features_set, labels, epochs = 200, batch_size = 64, class_weight=class_weight)

    predictions = model.predict(test_f)
    predictions = (predictions > 0.5)

    #metrics calculations
    tn, fp, fn, tp = confusion_matrix(test_l, predictions).ravel()
    q= (tn+tp)/(tn+fp+fn+tp)
    se= tp/(tp+fn)
    sp= tn/(tn+fp)
    Accuracy.append(q)
    Sensitivity.append(se)
    Specificity.append(sp)
    #print (q,se,sp)
print (Accuracy,Sensitivity,Specificity)

#average metrics by 10 times cross validation
q = np.average(Accuracy)
se = np.average(Sensitivity)
sp = np.average(Specificity)
print (q,se,sp)
