#Aim:  Our aim for this data set is to build a network intrusion detector, a predictive model capable of distinguishing between "bad'' connections, called intrusions or attacks, and ``good'' normal connections.

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, Row
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from time import time
from pyspark.mllib.regression import LabeledPoint
from numpy import array

conf = SparkConf().setAppName("SensorNetworkPOC")
sc = SparkContext()
hiveCtx = HiveContext(sc)

rawTrain = sc.textFile("/user/edureka/kddcup.data")

print rawTrain.first()

rawTest = sc.textFile("/user/edureka/corrected")

csv = rawTrain.map(lambda line: line.split(","))
check_csv = rawTest.map(lambda line: line.split(","))

protocols = csv.map(lambda x: x[1]).distinct().collect()
services = csv.map(lambda x: x[2]).distinct().collect()
flags = csv.map(lambda x: x[3]).distinct().collect()

def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[0:41]

    # convert protocol to numeric categorical variable
    try:
        clean_line_split[1] = protocols.index(clean_line_split[1])
    except:
        clean_line_split[1] = len(protocols)

    # convert service to numeric categorical variable
    try:
        clean_line_split[2] = services.index(clean_line_split[2])
    except:
        clean_line_split[2] = len(services)

    # convert flag to numeric categorical variable
    try:
        clean_line_split[3] = flags.index(clean_line_split[3])
    except:
        clean_line_split[3] = len(flags)

    # convert label to binary label
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0

    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data = csv.map(create_labeled_point)
test_data = check_csv.map(create_labeled_point)

# Build the model
t0 = time()
tree_model = RandomForest.trainClassifier(training_data, numClasses=2,
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                          numTrees=10,featureSubsetStrategy="auto",
                                          impurity='entropy', maxDepth=4, maxBins=100)


tt = time() - t0

print ("Classifier trained in {} seconds".format(round(tt,3)))

predictions = tree_model.predict(test_data.map(lambda p: p.features))
labels_and_preds = test_data.map(lambda p: p.label).zip(predictions)

t0 = time()
test_accuracy = labels_and_preds.filter(lambda vp: vp[0] == vp[1]).count() / float(test_data.count())
tt = time() - t0

print ("Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4)))

print (tree_model.toDebugString())

