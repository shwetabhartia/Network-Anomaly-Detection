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
#print(train_data.describe())
#np.isnan(train_data['src_bytes'])
#print(train_data[''].isnull().values.any())

#inds = pd.isnull(train_data).any(1).nonzero()[0]
#print(inds)
#from numpy import float32
#numpyMatrix = train_data.as_matrix()
#ne_data = numpyMatrix.astype(float32)

# feature selection through random forests :
train_data.info()

rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None)
rf.fit(train_data,target)
#print(rf.apply(train_data))
importance = rf.feature_importances_ 

column_names = train_data.columns 
dict1 = {}
sorteddict = {}
for i in range(len(column_names)):
    
    
    dict1[column_names[i]] = importance[i]
import operator 

sorteddict = sorted(dict1.items(), key=operator.itemgetter(1),reverse = True)
print(sorteddict)    


# data construncted by picking most imp features :
#train_data.info()  
#columns = list(train_data.columns)
#print(columns)
predictors = ['service_ecr_i','srv_count','protocol_type_icmp','count','src_bytes','protocol_type_tcp','diff_srv_rate','flag_SF','dst_host_diff_srv_rate','dst_host_srv_count','service_private','flag_S0','dst_bytes','service_http','serror_rate','dst_host_count','dst_host_srv_diff_host_rate','logged_in','srv_diff_host_rate','protocol_type_udp','hot','rerror_rate','num_compromised','wrong_fragment','duration','service_eco_i','service_other','flag_REJ','service_ftp_data','service_smtp','flag_RSTR','service_ftp','is_guest_login','service_domain_u','service_telnet','flag_SH']
#redictors = ['service_ecr_i','srv_count','protocol_type_icmp','count','src_bytes','dst_host_same_src_port_rate','protocol_type_tcp','diff_srv_rate','flag_SF','dst_host_diff_srv_rate','same_srv_rate','dst_host_srv_count','dst_host_same_srv_rate','service_private','flag_S0','dst_bytes','dst_bytes','service_http','dst_host_srv_serror_rate','dst_host_serror_rate','serror_rate','dst_host_count','dst_host_srv_diff_host_rate','logged_in','srv_serror_rate']

print(len(predictors))
train_dataf = train_data[predictors]  

from sklearn import cluster
k_means = cluster.KMeans(n_clusters=20)
k_means.fit(train_dataf)

train_dataf['clusters'] = k_means.labels_ 

print(train_dataf.head())


# dbscan :

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
#data_f4 = train_dataf[40726:40745]#normal
#target_f4 = target[40726:40745]
#data_f5 = train_dataf[52145:52181]# ioswee
#target_f5 = target[52145:52181]

data = pd.concat((data_f1,data_f2,data_f3,data_f4,data_f5,data_f6), ignore_index=True)
target_f = pd.concat((target_f1,target_f2,target_f3,target_f4,target_f5,target_f6),ignore_index=True)



from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
dbscan = DBSCAN(eps=3, algorithm = 'kd_tree',min_samples=5)    
#dbscan = DBSCAN(eps=4.5, algorithm = 'kd_tree',min_samples=5)    - 750   
#dbscan = DBSCAN(eps=5.25, algorithm = 'kd_tree',min_samples=5)    - 500
#print(data)
dbscan.fit(data) 
print(dbscan.labels_)

labels = dbscan.labels_
data['acctual_response'] = target_f
data['preditions'] = labels
 
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
#data.to_csv('F:/sem3/big_data/final_project/New folder/sample_test4.csv',sep = '\t')




  
            
            
            

