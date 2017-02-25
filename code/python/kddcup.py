import os
import operator
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cluster
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

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
    correlated_matrix=data.corr(method='pearson')
    correlated_features = ['srv_rerror_rate', 'dst_host_srv_rerror_rate', 'dst_host_rerror_rate', 'dst_host_same_src_port_rate', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'same_srv_rate', 'srv_serror_rate']
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
    # make predictions
    expected = targetTest
    predicted = rfc.predict(testData)
    # summarize the fit of the model
    print(accuracy_score(expected, predicted))

def unsupervised_preprocessing(train_data):
    '''
    Applies all the preprocessing steps required for the training an unsupervised learning model

    Args:
        train_data: The input training data dataframe that will be read from a csv file
    '''
    del train_data['srv_rerror_rate']
    del train_data['dst_host_srv_rerror_rate']
    del train_data['dst_host_rerror_rate']
    # deleting outlier :
    train_data = train_data.query('src_bytes != 693375640')

    # part of pre-processing ..decided to do logarithemic transformations on all the numeric values.
    #but log tranformation fail if any column takes 0 as its value. So I have converted 0 to 1 . then applying log on that particular column brings 
    # to 0..its original value.
    train_data["src_bytes"][train_data["src_bytes"] == 0] = 1
    train_data["dst_bytes"][train_data["dst_bytes"] == 0] = 1

    train_data["src_bytes"] = train_data['src_bytes'][train_data["src_bytes"] != 0].apply(np.log)
    train_data["dst_bytes"] = train_data['dst_bytes'][train_data["dst_bytes"] != 0].apply(np.log)

    train_data["duration"][train_data["duration"] == 0] = 1
    train_data["duration"] = train_data['duration'][train_data["duration"] != 0].apply(np.log)

    train_data["hot"][train_data["hot"] == 0] = 1
    train_data["hot"] = train_data['hot'][train_data["hot"] != 0].apply(np.log)


    train_data["num_failed_logins"][train_data["num_failed_logins"] == 0] = 1
    train_data["num_failed_logins"] = train_data['num_failed_logins'][train_data["num_failed_logins"] != 0].apply(np.log)

    train_data["num_compromised"][train_data["num_compromised"] == 0] = 1
    train_data["num_compromised"] = train_data['num_compromised'][train_data["num_compromised"] != 0].apply(np.log)

    train_data["num_root"][train_data["num_root"] == 0] = 1
    train_data["num_root"] = train_data['num_root'][train_data["num_root"] != 0].apply(np.log)


    train_data["num_file_creations"][train_data["num_file_creations"] == 0] = 1
    train_data["num_file_creations"] = train_data['num_file_creations'][train_data["num_file_creations"] != 0].apply(np.log)

    train_data["num_access_files"][train_data["num_access_files"] == 0] = 1
    train_data["num_access_files"] = train_data['num_access_files'][train_data["num_access_files"] != 0].apply(np.log)

    train_data["count_f"][train_data["count_f"] == 0] = 1
    train_data["count_f"] = train_data['count_f'][train_data["count_f"] != 0].apply(np.log)

    train_data["srv_count"][train_data["srv_count"] == 0] = 1
    train_data["srv_count"] = train_data['srv_count'][train_data["srv_count"] != 0].apply(np.log)


    train_data["dst_host_count"][train_data["dst_host_count"] == 0] = 1
    train_data["dst_host_count"] = train_data['dst_host_count'][train_data["dst_host_count"] != 0].apply(np.log)

    train_data["dst_host_srv_count"][train_data["dst_host_srv_count"] == 0] = 1
    train_data["dst_host_srv_count"] = train_data['dst_host_srv_count'][train_data["dst_host_srv_count"] != 0].apply(np.log)

    train_data["srv_count"][train_data["srv_count"] == 0] = 1
    train_data["srv_count"] = train_data['srv_count'][train_data["srv_count"] != 0].apply(np.log)

    # converting categorical data into dummy :
    train_data = train_data.join(pd.get_dummies(train_data['protocol_type'], prefix="protocol_type"))
    del train_data["protocol_type"]

    train_data = train_data.join(pd.get_dummies(train_data['service'], prefix="service"))
    del train_data["service"]

    train_data = train_data.join(pd.get_dummies(train_data['flag'], prefix="flag"))
    del train_data["flag"]

    target = train_data['Result']
    del train_data['Result']
    return train_data, target

def unsupervised_importances(train_data, target):
    '''
    Calculates the important features of the training data and returns a dataframe that contains only the important features
    
    Args:
        train_data: The training data which was preprocesses
        target: The target labels of the training data
    '''
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None)
    rf.fit(train_data,target)
    importance = rf.feature_importances_ 

    column_names = train_data.columns 
    dict1 = {}
    sorteddict = {}
    for i in range(len(column_names)):
        dict1[column_names[i]] = importance[i]
    import operator 

    sorteddict = sorted(dict1.items(), key=operator.itemgetter(1),reverse = True)
    predictors = ['service_ecr_i','srv_count','protocol_type_icmp','count_f','src_bytes','protocol_type_tcp','diff_srv_rate','flag_SF','dst_host_diff_srv_rate','dst_host_srv_count','service_private','flag_S0','dst_bytes','service_http','serror_rate','dst_host_count','dst_host_srv_diff_host_rate','logged_in','srv_diff_host_rate','protocol_type_udp','hot','rerror_rate','num_compromised','wrong_fragment','duration','service_eco_i','service_other','flag_REJ','service_ftp_data','service_smtp','flag_RSTR','service_ftp','is_guest_login','service_domain_u','service_telnet','flag_SH']
    train_dataf = train_data[predictors]
    return train_dataf

def apply_kmeans(train_data):
    k_means = cluster.KMeans(n_clusters=20)
    k_means.fit(train_data)
    train_data['clusters'] = k_means.labels_ 
    print(train_data.head())

def apply_dbscan(train_dataf, target):
    '''
    This meth
    '''

    data_f1 = train_dataf[53590:60000]#neotune
    target_f1 = target[53590:60000]
    data_f2 = train_dataf[7780:10000]#smurf
    target_f2 = target[7780:10000]
    data_f3 = train_dataf[10:3000]#normal + 2 records buffer
    target_f3 = target[10:3000]
    data_f4 = train_dataf[19300:19303]#teardrop
    target_f4 = target[19300:19303]
    data_f5 = train_dataf[43080:43082]#nmap
    target_f5 = target[43080:43082]
    data_f6 = train_dataf[39752:39790]#nmap
    target_f6 = target[39752:39790]

    data = pd.concat((data_f1,data_f2,data_f3,data_f4,data_f5,data_f6), ignore_index=True)
    target_f = pd.concat((target_f1,target_f2,target_f3,target_f4,target_f5,target_f6),ignore_index=True)
    dbscan = DBSCAN(eps=3, algorithm = 'kd_tree',min_samples=5)    
    #dbscan = DBSCAN(eps=4.5, algorithm = 'kd_tree',min_samples=5)    - 750   
    #dbscan = DBSCAN(eps=5.25, algorithm = 'kd_tree',min_samples=5)    - 500
    dbscan.fit(data) 
    print(dbscan.labels_)

    labels = dbscan.labels_
    data['acctual_response'] = target_f
    data['preditions'] = labels
     
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    #data.to_csv('F:/sem3/big_data/final_project/New folder/sample_test4.csv',sep = '\t')


def main():
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, '../../data/kdd.csv')
    data = load_data(filename)
    
    
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
    #print sortedDict
    #Removing last 3 features as it less important in classification
    predictors = sortedDict[:-3]
    columns=[]
    for i in range(0,len(predictors)):
        columns.append(predictors[i][0])
    trainData_classifier=trainData[columns]
    testData_classifier=testData[columns]
    apply_decision_tree(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_logistic_regression(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_svm(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_extra_trees_classifier(trainData_classifier,targetTrain,testData_classifier,targetTest)
    apply_random_forest_classifier(trainData_classifier,targetTrain,testData_classifier,targetTest)
    
    #Unsupervised learning
    train_data = load_data(filename)
    preprocessedTrainData, target = unsupervised_preprocessing(train_data)
    importantTrainData = unsupervised_importances(preprocessedTrainData, target)
    apply_kmeans(importantTrainData)
    apply_dbscan(importantTrainData, target)
main()