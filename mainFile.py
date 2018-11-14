from pyspark.sql import SparkSession
import loadDataFrame as df
import Preprocess as pr


spark = SparkSession.builder.appName("SNCF2").getOrCreate()
df = df.loadDataframe(spark)
#df.transData()
pr = pr.Process()
pr.getDataSplit(df.transData())
