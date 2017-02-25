# Databricks notebook source exported at Thu, 24 Nov 2016 20:15:13 UTC
# main program :
from pyspark.sql.functions import *

#Reading Data : one option :
dataPath = "/FileStore/tables/xyi5g44r1479453877009/kdd.csv"
dataframe = sqlContext.read.format('com.databricks.spark.csv').options(header='true').options(inferSchema='true').load(dataPath)
dataframe.coalesce(6)
# feature transformations : converting 0 value of numberic features into 1 so that it wont affect log transfermations :
df = dataframe.replace(0,1, ['src_bytes','dst_bytes','duration','hot','num_failed_logins','num_compromised','num_root','num_file_creations','num_access_files','count_f','srv_count','dst_host_count','dst_host_srv_count'])

df1 = df.withColumn('src_bytes', log(df.src_bytes)).drop(df.src_bytes)
df2 = df1.withColumn('dst_bytes', log(df.dst_bytes)).drop(df.dst_bytes)

df3 = df2.withColumn('duration', log(df.duration)).drop(df2.duration)
df4 = df3.withColumn('hot', log(df.hot)).drop(df3.hot)
df5 = df4.withColumn('num_failed_logins', log(df.num_failed_logins)).drop(df4.num_failed_logins)
df6 = df5.withColumn('num_compromised', log(df.num_compromised)).drop(df5.num_compromised)
df7 = df6.withColumn('num_root', log(df.num_root)).drop(df6.num_root)
df8 = df7.withColumn('num_file_creations', log(df.num_file_creations)).drop(df7.num_file_creations)
df9 = df8.withColumn('num_access_files', log(df.num_access_files)).drop(df8.num_access_files)
df10 = df9.withColumn('count_f', log(df.count_f)).drop(df9.count_f)
df11 = df10.withColumn('srv_count', log(df.srv_count)).drop(df10.srv_count)
df12 = df11.withColumn('dst_host_count', log(df.dst_host_count)).drop(df11.dst_host_count)
df13 = df12.withColumn('dst_host_srv_count', log(df.dst_host_srv_count)).drop(df12.dst_host_srv_count)
df13.take(1)


# COMMAND ----------

# one hot encoding for coverting categorical variables into dummy variables.

from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import StringIndexer, VectorIndexer,OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# converting protocol type into dummy variable
stringIndexer = StringIndexer(inputCol="protocol_type", outputCol="indexed_protocol_type")
model = stringIndexer.fit(df13)
td = model.transform(df13)
encoder = OneHotEncoder(inputCol="indexed_protocol_type", outputCol="encoded_protocol_type")
encoder.transform(td).encoded_protocol_type
df14 = td.drop(td.protocol_type)
#df14.take(2)

# converting service into dummy variable
stringIndexer = StringIndexer(inputCol="service", outputCol="indexed_service")
model = stringIndexer.fit(df14)
td1 = model.transform(df14)
encoder = OneHotEncoder(inputCol="indexed_service", outputCol="encoded_service")
encoder.transform(td1).encoded_service
df15 = td1.drop(td1.service)
#df15.take(2)

# converting flag into dummy variable
stringIndexer = StringIndexer(inputCol="flag", outputCol="indexed_flag")
model = stringIndexer.fit(df15)
td2 = model.transform(df15)
encoder = OneHotEncoder(inputCol="indexed_flag", outputCol="encoded_flag")
#encoder.transform(td2).encoded_flag
encoder.transform(td2).head().encoded_flag
prefinalDF = td2.drop(td2.flag)

# converting Result into dummy variable
stringIndexer = StringIndexer(inputCol="Result", outputCol="indexed_result")
model = stringIndexer.fit(prefinalDF)
td4 = model.transform(prefinalDF)
encoder = OneHotEncoder(inputCol="indexed_result", outputCol="encoded_result")
#encoder.transform(td2).encoded_flag
encoder.transform(td4).head().encoded_result
finalDF = td4.drop(td4.Result)

# dropping highly correlated features :

final1 = finalDF.drop(finalDF.srv_rerror_rate)
final2 = final1.drop(final1.dst_host_srv_rerror_rate)
final3 = final2.drop(final2.dst_host_rerror_rate)
final4 = final3.drop(final3.dst_host_same_src_port_rate)
final5 = final4.drop(final4.dst_host_same_srv_rate)
final6 = final5.drop(final5.dst_host_serror_rate)
final7 = final6.drop(final6.dst_host_srv_serror_rate)
final8 = final7.drop(final7.same_srv_rate)
final_data = final8.drop(final8.srv_serror_rate)

#finalDF.select(finalDF.indexed_flag).distinct().show()
#print(type(finalDF.duration))
#final3.dtypes
#final_data.printSchema()
######## random forest implementation :

# any ml algorithm need features and lable to be given in vector form. So we are converting predictors into dense vector form in this way :
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['duration',	'src_bytes',	'dst_bytes',	'land',	'wrong_fragment',	'urgent',	'hot',	'num_failed_logins',	'logged_in',	'num_compromised',	'root_shell',	'su_attempted',	'num_root',	'num_file_creations',	'num_shells',	'num_access_files',	'num_outbound_cmds',	'is_host_login',	'is_guest_login',	'count_f',	'srv_count',	'serror_rate',	'rerror_rate',	'diff_srv_rate',	'srv_diff_host_rate',	'dst_host_count',	'dst_host_srv_count',	'dst_host_diff_srv_rate',	'dst_host_srv_diff_host_rate','indexed_protocol_type',	'indexed_service','indexed_flag'],outputCol="features")
output = assembler.transform(final_data)
traindata = output.select(output.features,output.indexed_result)

# implementing random forest on the converted features and label .....
from pyspark.ml.classification import RandomForestClassifier

stringIndexer = StringIndexer(inputCol="indexed_result", outputCol="indexed")
si_model = stringIndexer.fit(traindata)
td = si_model.transform(traindata)
rf = RandomForestClassifier(numTrees=100,labelCol="indexed",impurity="entropy", maxBins=70, seed=42)
model = rf.fit(td)
model.featureImportances


# COMMAND ----------

# importance features selected by looking at the results of random forests :
impcol = ['duration','src_bytes','dst_bytes','land','wrong_fragment','hot','logged_in','root_shell','num_root',	'num_file_creations','num_shells','is_guest_login','count_f','srv_count','serror_rate','rerror_rate','diff_srv_rate',	'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_diff_srv_rate','dst_host_srv_diff_host_rate',	'indexed_protocol_type','indexed_service',	'indexed_flag']
