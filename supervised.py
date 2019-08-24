import sklearn as sk
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def data_split(dataSet, yCol = -1, spliSize = 0.33):                            #Splitting the dataset for the model.
    data = pd.read_csv(dataSet, header = 0)
    data.replace('?', -9999, inplace = True)                                    #Replacing the unknown values.
    y = data.iloc[:, yCol]
    x = data.iloc[:, :yCol]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = spliSize)
    return xTrain, xTest, yTrain, yTest

def classifier(dataSet, yCol = -1):
    algoDict = {0:"Logistic Regression", 1:"Naive Bayes", 2:"Stochastic Gradient Descent", 3:"K Nearest Neighbors", 4:"Decision Tree", 5:"Random Forest", 6:"SVM"}  #Dictionary of the algorithms used.
    spliSize = float(input("Enter the size of the testing dataset : "))         #User input for the testing size.
    xTrain, xTest, yTrain, yTest = data_split(dataSet, yCol, spliSize)
    accuracyList = []
    results = []
    results.append(logistic_regression(xTrain, yTrain, xTest, yTest))
    results.append(naive_bayes(xTrain, yTrain, xTest, yTest))
    results.append(stochastic_gradient_descent(xTrain, yTrain, xTest, yTest))
    results.append(k_nearest_neighbors(xTrain, yTrain, xTest, yTest))
    results.append(decision_tree(xTrain, yTrain, xTest, yTest))
    results.append(random_forest(xTrain, yTrain, xTest, yTest))
    results.append(svm(xTrain, yTrain, xTest, yTest))
    for i in range(0,7):
        accuracyList.append(results[i][0])
    highAccuracy = max(accuracyList)                                            #Calculating the highest accuracy.
    algorithmName = algoDict[accuracyList.index(highAccuracy)]
    savedModel = results[accuracyList.index(highAccuracy)][1]
    return algorithmName, highAccuracy, savedModel                              #Returns the algorithm name which gave the maximum accuracy,
                                                                                # highest accuracy and the saved model.
'''
    Main Computation.
'''

def logistic_regression(xTrain, yTrain, xTest, yTest):
    lr = LogisticRegression()
    lr.fit(xTrain, yTrain)
    yPredict = lr.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_lr_model = pickle.dumps(lr)
    return [accuracy, save_lr_model]

def naive_bayes(xTrain, yTrain, xTest, yTest):
    nb = GaussianNB()
    nb.fit(xTrain, yTrain)
    yPredict = nb.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_lr_model = pickle.dumps(nb)
    return [accuracy, save_lr_model]

def stochastic_gradient_descent(xTrain, yTrain, xTest, yTest):
    sgd = SGDClassifier(loss = 'modified_huber', shuffle = True, random_state = 101)
    sgd.fit(xTrain, yTrain)
    yPredict = sgd.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_lr_model = pickle.dumps(sgd)
    return [accuracy, save_lr_model]

def k_nearest_neighbors(xTrain, yTrain, xTest, yTest):
    k = int(input("Enter the numbers of classes for k-neighbors classifier.:"))
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(xTrain, yTrain)
    yPredict = knn.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_lr_model = pickle.dumps(knn)
    return [accuracy, save_lr_model]

def decision_tree(xTrain, yTrain, xTest, yTest):
    leaf = int(input("Enter the number of classes for deccision tree algorithm. :"))
    dt = DecisionTreeClassifier(min_samples_leaf = leaf)
    dt.fit(xTrain, yTrain)
    yPredict = dt.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_dt_model = pickle.dumps(dt)
    return [accuracy, save_dt_model]

def random_forest(xTrain, yTrain, xTest, yTest):
    estimators = int(input("Enter the number of estimators for random forest algorithm. :"))
    rfc = RandomForestClassifier(n_estimators = estimators)
    rfc.fit(xTrain, yTrain)
    yPredict = rfc.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_rfc_model = pickle.dumps(rfc)
    return [accuracy, save_rfc_model]

def svm(xTrain, yTrain, xTest, yTest):
    kernel_fn = input("Enter the kernal function SVM algorithm. :")
    s_v_m = SVC(kernel = kernel_fn)
    s_v_m.fit(xTrain, yTrain)
    yPredict = s_v_m.predict(xTest)
    acc = accuracy_score(yTest, yPredict)
    accuracy = acc*100
    save_svm_model = pickle.dumps(s_v_m)
    return [accuracy, save_svm_model]
