import sys
import os
from collections import OrderedDict
from numpy import array
from math import sqrt
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import * 
import math

conf = SparkConf().setAppName("testApp")
sc = SparkContext(conf=conf) 
sqlCtx = SQLContext(sc)

# Data preprocessing
def sexTransformMapper(elem):
    '''Function which transform "male" into 1 and else things into 0
    - elem : string
    - return : vector
    '''
    if 'male' == elem:
        return 1
    else :
        return 0
# Data preprocessing
def survivedTransformMapper(elem):
    if 1 == elem:
        return 0
    else :
        return 1

# load raw data
print "Loading RAW data..."
raw_data = sc.textFile("/user/edureka/train_titanic.csv")
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
header = raw_data.take(1)[0]
raw_data= raw_data.filter(lambda line: line != header)

trainTitanic = raw_data.map(lambda x: x.split(',')).filter(lambda line: line[3] != '').filter(lambda line: line[5] != '').filter(lambda line: line[6] != '').filter(lambda line: line[7] != '').filter(lambda line: line[8] != '').filter(lambda line: line[9] != '').filter(lambda line: line[10] != '').filter(lambda line: line[11] != '')
rdd = trainTitanic.map(lambda x: Row(PassengerId = int(x[0]), Survived=float(survivedTransformMapper(int(x[1]))), Pclass=int(x[2]), Name=x[3], Sex=int(sexTransformMapper(x[5])),Age=float(x[6]),SibSp=int(x[7]), Parch=int(x[8]), Ticket=x[9], Fare=float(x[10]), Cabin=x[11], Embarked=x[12]))

df = sqlCtx.createDataFrame(rdd)

# Spliting in train and test set. Beware : It sorts the dataset
train = df
print train.head(5)
 
#ndex labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
#train = StringIndexer(inputCol="Sex", outputCol="indexedSex").fit(train).transform(train)
#train = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked").fit(train).transform(train)
 
train = StringIndexer(inputCol="Survived", outputCol="indexedSurvived").fit(train).transform(train)
 
# One Hot Encoder on indexed features
#train = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec").transform(train)
#train = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec").transform(train)
 
# Feature assembler as a vector
train = VectorAssembler(inputCols=["Pclass","Sex", "Age","SibSp","Fare"],outputCol="features").transform(train)
 
rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features")

print "DF before split"
print train.head(5)
#sc.stop() 
# Spliting in train and test set. Beware : It sorts the dataset
(traindf, testdf) = train.randomSplit([0.7,0.3])

model = rf.fit(traindf)
 
predictions = model.transform(testdf)


# Select example rows to display.
predictions.select(col("prediction"),col("probability"),).show(5)

# Select example rows to display.
predictions.columns 
 
# Select example rows to display.
predictions.select("prediction", "indexedSurvived", "features").show(5)
 
# Select (prediction, true label) and compute test error
predictions = predictions.select(col("Survived"),col("prediction"))
#evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
#accuracy = evaluator.evaluate(predictions)
#print("Test Error = %g" % (1.0 - accuracy))
 
 
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
 
#evaluatorf1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
#f1 = evaluatorf1.evaluate(predictions)
#print("f1 = %g" % f1)
 
#evaluatorwp = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
#wp = evaluatorwp.evaluate(predictions)
#print("weightedPrecision = %g" % wp)
 
#evaluatorwr = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
#wr = evaluatorwr.evaluate(predictions)
#print("weightedRecall = %g" % wr)
 
# close sparkcontext
sc.stop()
