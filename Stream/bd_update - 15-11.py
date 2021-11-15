# importing required libraries
import sys, pyspark
from pyspark import SparkContext
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler
#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


sc = SparkContext("local[2]", "Sentiment")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

lines = ssc.socketTextStream("localhost",6100)

def dstream_to_rdd(data_word):
	try:
		data_word = data_word.filter(lambda x: len(x) > 0)
		rowRdd = data_word.map(lambda w: Row(row_word=w))
		tweet_df = spark.createDataFrame(rowRdd)
		pre_model = preprocess(row_word)
	except : 
		print('No data in here')

def preprocess(row):

	stage_1 = RegexTokenizer(inputCol= 'row_word' , outputCol= 'tokens', pattern= '\\W')
	stage_2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	pipeline= Pipeline(stages= [stage_1, stage_2, stage_3])
	print(pipeline) '''
	words = lines.select(explode(split(" ", "t_end")).alias("word"))
	words = words.na.replace('', None)
	words = words.na.drop()
	words = words.withColumn('word', F.regexp_replace('word', r'http\S+', ''))
	words = words.withColumn('word', F.regexp_replace('word', '@\w+', ''))
	words = words.withColumn('word', F.regexp_replace('word', '#', ''))
	words = words.withColumn('word', F.regexp_replace('word', 'RT', ''))
	words = words.withColumn('word', F.regexp_replace('word', ':', ''))
	print(words)
	
	return words '''


words = lines.flatMap(lambda line : line.split(" "))
words.foreachRDD(dstream_to_rdd)

ssc.start() 
ssc.awaitTermination()
