#Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns 
#Reading Data
train_data = pd.read_csv('F:/sem3/big_data/final_project/kdd.csv')
#train_data.info()
# this checks the correlation among all the numeric variables and helps in deciding inclusion or removal of variables.

print(train_data.corr(method='pearson'))

# removed highly correlated fields 
del train_data['srv_rerror_rate']
del train_data['dst_host_srv_rerror_rate']
del train_data['dst_host_rerror_rate']


# deleting outlier :
#train_data.info()
train_data = train_data.query('src_bytes != 693375640')
#train_data = train_data.query('src_bytes != 0')
#train_data = train_data.query('dst_bytes != 0')
#train_data.info()

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

train_data["count"][train_data["count"] == 0] = 1
train_data["count"] = train_data['count'][train_data["count"] != 0].apply(np.log)

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
#print(train_data.head())
#train_data = train_data.join(pd.get_dummies(train_data['Result'], prefix="Result"))
#del train_data["Result"]

target = train_data['Result']
del train_data['Result']
train_data.info()