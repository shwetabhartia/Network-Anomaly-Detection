import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cluster
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(path):
    """
    Loads the data from CSV file
    
    Args:
    path: Location of the csv file
    
    Returns:
    A pandas Dataframe that contains the dataset loaded from the file
    """
    data = pd.read_csv(path)
    return data

def delete_outliers(data):
    """
    Deletes outliers from a column and returns the modified Dataframe

    Args:
        dataframe: The input Dataframe on which the outliers have to be removed

    Returns:
        A pandas Dataframe with that doesnt contain outliers
    """
    
    data = data[data["src_bytes"]!= 693375640]
    return data

def transform_data(data, numerical_features):
    """
    Transforms numerical data from existing scale to specified scale,Data transformation is required since numerical features can take various ranges and a common range is required among all the numerical features 

    Args:
        dataframe: The input Dataframe on which transformations have to be applied
        numerical_features: List of all numerical feature names

    Returns:
        A pandas Dataframe containing transformed numerical features
    """
    scaler = MinMaxScaler()
    for feature in numerical_features:
       	data[[feature]]=scaler.fit_transform(data[[feature]])
    return data

def remove_correlated_features(data):
    """
    Calculates the correlation among the numerical features and removes the highly correlated features

    Args:
        dataframe: The input Dataframe on which correlated features have to be removed

    Returns:
        A pandas Dataframe that doesn't contain highly correlated features
    """
    correlated_features=data.corr(method='pearson')
    for feature in correlated_features:
    	del data[feature]
    return data

def encode_categorical_features(data, categorical_features):
    """
    Encodes categorical features into numerical features using dummy encoding

    Args:
        dataframe: The input Dataframe on which categorical features have to be encoded
        categorical_features: The list of categorical features in the dataframe

    Returns:
        A pandas Dataframe that contains categorical features encoded into numerical features
    """
    for feature in categorical_features:
    	data = data.join(pd.get_dummies(data[feature], prefix=feature))
    	del data[feature]
    return data

def calculate_importance_of_features(data,target):
    """
    Calculates the importance of features in the dataset

    Args:
        dataframe: The input Dataframe on which the importance of the features have to be calculated

    Returns:
        A Descending sorted Dictionary that contains feature names as keys and the importance as values

    """
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None)
    rf.fit(data,target)
    importance = rf.feature_importances_
    column_names = data.columns
    dict1 = {}
    sorteddict = {}
    for i in range(len(column_names)):
    	dict1[column_names[i]] = importance[i]
    sorteddict = sorted(dict1.items(), key=operator.itemgetter(1),reverse = True)
    #print type(sorteddict)
    return sorteddict

def apply_decision_tree(trainData, targetTrain, testData, targetTest):
    """
    Applies decision tree algorithm on the dataset, by tuning various parameters

    Args:
        dataframe: The input trainData, testData and class label on which the decision tree algorithm has to be applied

    """
    # fit a CART model to the data
    dt = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
    dt.fit(trainData, targetTrain)
    print(dt)
    # make predictions
    expected = targetTest
    predicted = dt.predict(testData)
    # summarize the fit of the model
    print(accuracy_score(expected, predicted))

def apply_logistic_regression(trainData, targetTrain, testData, targetTest):
    """
    Applies logistic regression algorithm on the dataset, by tuning various parameters

    Args:
        dataframe: The input trainData, testData and class label on which the logistic regression algorithm has to be applied

    """
    # fit a logistic regression model to the data
    lr = LogisticRegression()
    lr.fit(trainData, targetTrain)
    print(lr)
    # make predictions
    expected = targetTest
    predicted = lr.predict(testData)
    # summarize the fit of the model
    print(accuracy_score(expected, predicted))

def apply_svm(trainData, targetTrain, testData, targetTest):
    """
    Applies support vector machine algorithm on the dataset, by tuning various parameters

    Args:
        dataframe: The input trainData, testData and class label on which the support vector machine algorithm has to be applied

    """
    # fit a SVM model to the data
    svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    svm.fit(trainData,targetTrain)
    print(svm)
    # make predictions
    expected = targetTest
    predicted = svm.predict(testData)
    # summarize the fit of the model
    print(accuracy_score(expected, predicted))

def apply_extra_trees_classifier(trainData, targetTrain, testData, targetTest):
    """
    Applies decision tree algorithm on the dataset, by tuning various parameters

    Args:
        dataframe: The input trainData, testData and class label on which the decision tree algorithm has to be applied

    """
    # fit a CART model to the data
    etc = ExtraTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
          max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
          min_samples_split=2, min_weight_fraction_leaf=0.0,
          random_state=None, splitter='random')
    etc.fit(trainData, targetTrain)
    print(etc)
    # make predictions
    expected = targetTest
    predicted = etc.predict(testData)
    # summarize the fit of the model
    print(accuracy_score(expected, predicted))

def apply_random_forest_classifier(trainData, targetTrain, testData, targetTest):
    """
    Applies decision tree algorithm on the dataset, by tuning various parameters

    Args:
        dataframe: The input trainData, testData and class label on which the decision tree algorithm has to be applied

    """
    # fit a CART model to the data
    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    rfc.fit(trainData, targetTrain)
    print(rfc)
    # make predictions
    expected = targetTest
    predicted = rfc.predict(testData)
    # summarize the fit of the model
    print(accuracy_score(expected, predicted))

def main():
    data = load_data('F:\Shweta\Study\Big Data Course\Folders\project-003\code\python\supervisedlearning\kdd.csv')
    cleanData = delete_outliers(data)
    numerical_features=['src_bytes','dst_bytes','duration','hot','num_failed_logins','num_compromised','num_root','num_file_creations','num_access_files','count_f','srv_count','dst_host_count','dst_host_srv_count','srv_count']
    transformedData = transform_data(cleanData, numerical_features)
    correlatedCleanData = remove_correlated_features(transformedData)
    categorical_features=['protocol_type','service','flag']
    encodedData = encode_categorical_features(correlatedCleanData,categorical_features)
    msk = np.random.rand(len(encodedData)) < 0.7
    trainData = encodedData[msk]
    testData = encodedData[~msk]
    targetTrain = trainData['Result']
    del trainData['Result']
    targetTest = testData['Result']
    del testData['Result']
    sortedDict = calculate_importance_of_features(trainData,targetTrain)
    '''Printing sortedDict to analyse the most important features'''
    #print sortedDict
    '''Removing last 3 features as it less important in classification'''
    predictors = sortedDict[:-3]
    #print len(predictors)
    columns=[]
    for i in range(0,len(predictors)):
        columns.append(predictors[i][0])
        #print columns
    #print len(columns)
    trainData_classifier=trainData[columns]
    testData_classifier=testData[columns]
    apply_decision_tree(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_logistic_regression(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_svm(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_extra_trees_classifier(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_random_forest_classifier(trainData_classifier,targetTrain,testData_classifier,targetTest)
main()