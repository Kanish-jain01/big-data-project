import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext, SparkSession,Row,Column
from pyspark.sql.functions import *
from pyspark.streaming import StreamingContext
#from pyspark.sql.DataFrame import  randomSplit
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, VectorAssembler,Tokenizer, OneHotEncoder

#creating a spark context
sc = SparkContext("local[2]", "Sentiment")

#creating a streaming context to read the incoming streaming data
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

'''
def pre_process(df):
	#to clean and preprocess the data
	
	print("IN")
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
'''

#function to convert the datastream converted to list into a dataframe object for further processing
def to_df(data):
	if data.isEmpty():
		return
	
	#creating of spark session for the streaming data input
	ss = SparkSession(data.context)
	data = data.collect()[0]
	columns = [f"feature{i}" for i in range(len(data[0]))]
	
	#create dataframe from the streaming data
	df = ss.createDataFrame(data, columns)
	df.show()
	
	print("IN")
	
	# regular expression tokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="words", pattern="\\W")
	# stop words
	add_stopwords = ["http","https","amp","rt","t","c","the"] 
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
	# bag of words count
	#countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=90000, minDF=5)
	
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=4) #minDocFreq: remove sparse terms
	
	label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx, nb])
	
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, nb])
	
	# Fit the pipeline to training documents.
	(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 100)

	lrModel = pipeline.fit(trainingData)
	
	predictions = lrModel.transform(testData)
	
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	
	print ("Test Area Under ROC: ", accuracy)
	
	'''
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
	'''

#function to convert the datastream to a list
def map_data(data):

	#load the incoming json file
	json_data=json.loads(data)
	list_rec = list()
	
	#convert the json file to tupple which is appended to a list and returned
	for rec in json_data:
		to_tuple = tuple(json_data[rec].values())
		list_rec.append(to_tuple)
	return list_rec 	

#creating a socket to read the data
lines = ssc.socketTextStream("localhost",6100).map(map_data).foreachRDD(to_df)

#start streaming
ssc.start() 
ssc.awaitTermination()

