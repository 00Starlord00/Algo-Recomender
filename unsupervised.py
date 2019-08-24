import sklearn
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def data_split(dataSet, spliSize = 0.33, yCol = -1):                            #Splitting the dataset for the model.
    data = pd.read_csv(dataSet, header = 0)
    data.replace('?', -9999, inplace = True)                                    #Replacing the unknown values.
    y = data.iloc[:, yCol]
    x = data.iloc[:, :yCol]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = spliSize)
    return xTrain, xTest, yTrain, yTest

def cluster(dataSet, yCol = -1):
    spliSize = float(input("Enter the size of the testing dataset : "))         #User input for the testing size.
    xTrain, xTest, yTrain, yTest = data_split(dataSet, spliSize, yCol)
    algoDict = {0: "k_means Clustering", 1: "Agglomerative Clustering"}
    accuracyList = []
    results = []
    clusters = int(input("Enter the number of clusters : "))
    results.append(k_means(xTrain, xTest, yTest, clusters))
    results.append(agglomerative_cluster(xTrain, xTest, yTest, clusters))
    for i in range(0,2):
        accuracyList.append(results[i][0])
    highAccuracy = max(accuracyList)                                            #Calculating the highest accuracy.
    algorithmName = algoDict[accuracyList.index(highAccuracy)]
    savedModel = results[accuracyList.index(highAccuracy)][1]
    return algorithmName, highAccuracy, savedModel                              #Returns the algorithm name which gave the maximum accuracy,
                                                                                # highest accuracy and the saved model.
'''
    Main Computation
'''

def k_means(xTrain, xTest, yTest, clusters = 4):
    km = KMeans(n_clusters = clusters)
    km.fit(xTrain)
    xKmean = km.fit_predict(xTest)
    acc = accuracy_score(yTest, xKmean)
    accuracy = acc*100
    save_kmeans_model = pickle.dumps(km)
    return [accuracy, save_kmeans_model]

def agglomerative_cluster(xTrain, xTest, yTest, clusters):
    aggc = AgglomerativeClustering(n_clusters = clusters)
    aggc.fit(xTrain)
    xAggC = aggc.fit_predict(xTest)
    acc = accuracy_score(yTest, xAggC)
    accuracy = acc * 100
    save_agg_model = pickle.dumps(aggc)
    return [accuracy, save_agg_model]
