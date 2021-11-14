# importing required libraries
import sys, pyspark
import pyspark.sql.streaming
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession,Row,Column
from pyspark.ml.feature import StringIndexer, VectorAssembler
#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp

sc = SparkContext("local[2]", "Sentiment")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

lines = ssc.socketTextStream("localhost",6100)
 
stage_1 = RegexTokenizer(inputCol= 'tweet' , outputCol= 'tokens', pattern= '\\W')

stage_2 = StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')

stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)

model = LogisticRegression(featuresCol= 'vector', labelCol= 'label')

pipeline = Pipeline(stages= [stage_1, stage_2, stage_3])

print(pipeline)

#pipelineFit = pipeline.fit(sys.argv[1])


def jsontordd(tweet_text):
	try:
		tweet_text = tweet_text.filter(lambda x: len(x) > 0)
		rowRdd = tweet_text.map(lambda w: Row(tweet=w))
		wordsDataFrame = spark.createDataFrame(rowRdd)
		print(wordsDataFrame)
		pipelineFit.transform(wordsDataFrame).select('tweet','prediction').show()
	except : 
		print('No data')

words = lines.flatMap(lambda line : line.split(" "))
words.foreachRDD(jsontordd)

ssc.start()             # Start the computation
ssc.awaitTermination()

'''
host = localhost
portt = 6100

spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()

# Create DataFrame representing the stream of input lines from connection to host:port
lines = spark.readStream.format('socket')\
.option('localhost', host)\
.option('port', port)\
.load()

    # Split the lines into words
 words = lines.select(explode(split(lines.value, ' ')).alias('word'))
    
for i in words:
	print words
'''
