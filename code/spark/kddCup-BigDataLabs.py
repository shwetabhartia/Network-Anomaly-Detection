import numpy
import datetime
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

programStartTime = datetime.datetime.now()

sparkConf = SparkConf().set("spark.ui.port", 5774)

spark = SparkSession.builder.config(conf=sparkConf).appName("KddCup").getOrCreate()

#Setting the Logging level to WARN
log4j = spark._jvm.org.apache.log4j
log4j.LogManager.getRootLogger().setLevel(log4j.Level.WARN)

#Schema for the data
kddSchema = StructType([StructField('Duration', IntegerType(), True),
                       StructField('ProtocolType', StringType(), True),
                       StructField('Service', StringType(), True),
                       StructField('Flag', StringType(), True),
                       StructField('SrcBytes',IntegerType(),True),
                       StructField('DstBytes',IntegerType(),True),
                       StructField('Land',IntegerType(), True),
                       StructField('WrongFragment', IntegerType(), True),
                       StructField('Urgent', IntegerType(), True),
                       StructField('Hot', IntegerType(), True),
                       StructField('NumFailedLogins', IntegerType(), True),
                       StructField('LoggedIn', IntegerType(), True),
                       StructField('NumCompromised', IntegerType(), True),
                       StructField('RootShell', IntegerType(), True),
                       StructField('SuAttempted', IntegerType(), True),
                       StructField('NumRoot', IntegerType(), True),
                       StructField('NumFileCreations', IntegerType(), True),
                       StructField('NumShells', IntegerType(), True),
                       StructField('NumAccessFiles', IntegerType(), True),
                       StructField('NumOutboundCmds', IntegerType(), True),
                       StructField('IsHostLogin', IntegerType(), True),
                       StructField('IsGuestLogin', IntegerType(), True),
                       StructField('Count', IntegerType(), True),
                       StructField('SrvCount', IntegerType(), True),
                       StructField('SerrorRate', DoubleType(), True),
                       StructField('SrvSerrorRate', DoubleType(), True),
                       StructField('RerrorRate', DoubleType(), True),
                       StructField('SrvRerrorRate', DoubleType(), True),
                       StructField('SameSrvRate', DoubleType(), True),
                       StructField('DiffSrvRate', DoubleType(), True),
                       StructField('SrvDiffHostRate', DoubleType(), True),
                       StructField('DstHostCount', IntegerType(), True),
                       StructField('DstHostSrvCount', IntegerType(), True),
                       StructField('DstHostSameSrvRate', DoubleType(), True),
                       StructField('DstHostDiffSrvRate', DoubleType(), True),
                       StructField('DstHostSameSrcPortRate', DoubleType(), True),
                       StructField('DstHostSrvDiffHostRate', DoubleType(), True),
                       StructField('DstHostSerrorRate', DoubleType(), True),
                       StructField('DstHostSrvSerrorRate', DoubleType(), True),
                       StructField('DstHostRerrorRate', DoubleType(), True),
                       StructField('DstHostSrvRerrorRate', DoubleType(), True),
                       StructField('AttackType', StringType(), True)
                       ])

#Reading the dataframe from the hdfs file
kddDF = spark.read.csv("/user/pramodvspk/kddcup.data", header=False, schema=kddSchema)
kddDF.createOrReplaceTempView("kddVIEW")
#Caching the DataFrame
spark.catalog.cacheTable("kddVIEW")
kddDF = spark.table("kddVIEW")
trainData, testData = kddDF.randomSplit([0.7,0.3], 24)


#Transforming the features on log scale
toTransformFeatures = ['SrcBytes','DstBytes','Duration','Hot','NumFailedLogins','NumCompromised','NumRoot','NumFileCreations','NumAccessFiles','Count','SrvCount','DstHostCount','DstHostSrvCount']
trainReplacedDF = trainData.replace(0,1,toTransformFeatures)
testReplacedDF = testData.replace(0,1,toTransformFeatures)
for feature in toTransformFeatures:
  trainReplacedDF = trainReplacedDF.withColumn(feature, log(trainReplacedDF[feature]))
  testReplacedDF = testReplacedDF.withColumn(feature, log(testReplacedDF[feature]))

#Indexing and Encoding Categorical features
pipeLineStages = []
toIndexColumns = toEncodeColumns = ["ProtocolType", "Service", "Flag", "AttackType"]

for column in toIndexColumns:
  currentIndexer = column+"_Indexer"
  currentIndexer = StringIndexer(inputCol=column, outputCol=column+"_index")
  pipeLineStages.append(currentIndexer)

for column in toEncodeColumns:
  currentEncoder = column+"_HotEncoder"
  currentEncoder = OneHotEncoder(inputCol=column+"_index", outputCol=column+"_vec")
  pipeLineStages.append(currentEncoder)

#Creating a list of all the unwanted indexed and categorical features
indexedAndCategoricalFeatures = ['ProtocolType','Service','Flag','AttackType','AttackType_vec','ProtocolType_index','Service_index','Flag_index']

#Creating a list of all Highly Correlated Columns
highlyCorrelatedFeatures = ['SrvRerrorRate', 'DstHostSrvRerrorRate', 'DstHostRerrorRate', 'DstHostSameSrcPortRate', 'DstHostSameSrvRate', 'DstHostSerrorRate', 'DstHostSrvSerrorRate', 'SrvSerrorRate', 'SameSrvRate']

#Creating a list containing the target feature
target_column = ['AttackType_index']

#Creating a list of all unimportant features by getting the list of unwanted features ran by the python proram by my team members
unimportantFeatures = ['Urgent','NumFailedLogins','NumCompromised','SuAttempted','NumAccessFiles','NumOutboundCmds','IsHostLogin','SrvSerrorRate','SrvRerrorRate','SameSrvRate','DstHostSameSrvRate','DstHostSameSrcPortRate','DstHostSerrorRate','DstHostSrvSerrorRate','DstHostRerrorRate','DstHostSrvRerrorRate']

unwanted_columns = indexedAndCategoricalFeatures + highlyCorrelatedFeatures + target_column + unimportantFeatures

#Creating an assembler of all the features
feature_assembler = VectorAssembler(inputCols = [column for column in testReplacedDF.columns if column not in unwanted_columns], outputCol="features")

#Declaring a decision tree classifier
decisionTree = DecisionTreeClassifier(labelCol='AttackType_index', featuresCol='features')
#Declaring a random forest classifier
randomForest = RandomForestClassifier(labelCol='AttackType_index', featuresCol='features', numTrees=5)
#Declaring a kMeans clustering algo
kMeans = KMeans().setK(30).setSeed(2)

pipeLineStages.append(feature_assembler)

decisionTreePipeLineStages = list(pipeLineStages)
randomForestPipeLineStages = list(pipeLineStages)
kMeansPipeLineStages = list(pipeLineStages)

decisionTreePipeLineStages.append(decisionTree)
randomForestPipeLineStages.append(randomForest)
kMeansPipeLineStages.append(kMeans)

#Creating the evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="AttackType_index", predictionCol="prediction", metricName="accuracy")

#Creating the kMeans pipeline
kMeansPipeLine = Pipeline(stages = kMeansPipeLineStages)
#Creating the Decision Tree pipeline
decisionTreePipeLine = Pipeline(stages = decisionTreePipeLineStages)
#Creating the Random Forest pipeline
randomForestPipeLine = Pipeline(stages = randomForestPipeLineStages)

decisionTreeModel = decisionTreePipeLine.fit(trainReplacedDF)
decisionTreePredictions = decisionTreeModel.transform(testReplacedDF)
decisionTreeAccuracy = evaluator.evaluate(decisionTreePredictions)
print  "Decision Tree Accuracy", decisionTreeAccuracy
print "Error", 1.0 - decisionTreeAccuracy

randomForestModel = randomForestPipeLine.fit(trainReplacedDF)
randomForestTreePredictions = randomForestModel.transform(testReplacedDF)
randomForestAccuracy = evaluator.evaluate(randomForestTreePredictions)
print  "Random Forest Accuracy", randomForestAccuracy
print "Random Forest Error", 1.0 - randomForestAccuracy

kMeansModel = kMeansPipeLine.fit(trainReplacedDF)
kMeansPredictions = kMeansModel.transform(testReplacedDF)
print kMeansPredictions.head()