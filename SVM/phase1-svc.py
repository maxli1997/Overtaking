from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

#parameter and normalization
scaler = MinMaxScaler(feature_range = (0, 1))
variables = 7
timestep = 10
leadtime = 15
groups = ["Range","RangeRate","Speed","Driving style","LaneDistanceLeft","Along_s","Alat_s"]

dataset = read_csv('phase1.csv',usecols=["event"])
events = dataset.event.unique()

Accuracy = []
Sensitivity = []
Specificity = []

dataset = read_csv('phase1.csv',usecols=groups)
scaler.fit(dataset)

for iteration in range(0,10):
    #cross validation
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
        for i in range(0,len(train)-leadtime):
            t =[]
            for j in range(i-timestep,i):
                for k in range(0,variables):
                    t.append(train[j][k])
            temp = []
            temp.append(labelset["Crossing"][i+leadtime])
            features_set.append(t)
            labels.append(temp)
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
        for i in range(0,len(test)-leadtime):
            t =[]
            for j in range(i-timestep,i):
                for k in range(0,variables):
                    t.append(test[j][k])
            temp = []
            temp.append(labelset["Crossing"][i+leadtime])
            test_f.append(t)
            test_l.append(temp)

    features_set, labels = np.array(features_set), np.array(labels)
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1]))
    l = np.reshape(labels,(labels.shape[0]))
    neg, pos = np.bincount(l)
    total = neg+pos
    test_f, test_l = np.array(test_f), np.array(test_l)
    test_f = np.reshape(test_f, (test_f.shape[0], test_f.shape[1]))

    #output_bias = initializers.Constant(initial_bias)
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    clf = svm.SVC(class_weight=class_weight)

    clf.fit(features_set,l)

    predictions = clf.predict(test_f)

    predictions = (predictions > 0.5)

    tn, fp, fn, tp = confusion_matrix(test_l, predictions).ravel()
    q= (tn+tp)/(tn+fp+fn+tp)
    se= tp/(tp+fn)
    sp= tn/(tn+fp)
    Accuracy.append(q)
    Sensitivity.append(se)
    Specificity.append(sp)
print (Accuracy,Sensitivity,Specificity)
q = np.average(Accuracy)
se = np.average(Sensitivity)
sp = np.average(Specificity)
print (q,se,sp)
