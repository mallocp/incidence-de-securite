from pyspark.sql import SparkSession,DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql import Row


class loadDataframe():
    def __init__(self, spark:SparkSession):
        self.df = spark.read.format("csv").load("incidents-securite.csv", sep=";", header=True)
        self.df = self.df.na.drop()

    def getDF(self):
        return self.df

    def transData(self):
        locals = self.df.select("Localisation").collect()
        liste_types = []
        for i in self.df.select("Type").collect():
            for p in i:
                liste_types.append(p)
        liste_types = list(set(liste_types))

        def dictLocal():
            liste = []
            dict_local = {}

            for i in  locals:
                for p in i:
                    liste.append(p)

            liste = list(set(liste))

            for p in liste:
                dict_local[p] = str(liste.index(p))

            return dict_local

        def esrUdf(esr):
            if esr == 'oui':
                return "1"
            else:
                return "0"

        def labelUdf(date):
            dates = date.split("/")[0:2]
            return str(dates[-1])


        def localisationUdf(lieu):
            dict_l = dictLocal()
            return dict_l[lieu]

        def typeUdf(type):
            if type: return 1.0
            else: return 0.0


        transEsr = udf(esrUdf, StringType())
        translabel = udf(labelUdf, StringType())
        translocalisation = udf(localisationUdf, StringType())

        for i in liste_types:
            def gUdf(type):
                if i == type:
                    return 1.0
                else:
                    return 0.0

            transtype = udf(gUdf, DoubleType())
            self.df = self.df.withColumn(i, transtype(self.df.Type).cast("Double")) \
                     .withColumn("ESRs", transEsr(self.df.ESR).cast("Double")) \
                    .withColumn("Local", translocalisation("Localisation").cast("Double"))\
                    .withColumn("label", translabel("Date").cast("Double"))


        self.df = self.df.na.drop()
        return self.df

