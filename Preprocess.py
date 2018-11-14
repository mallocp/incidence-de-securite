from pyspark.ml.feature import StringIndexer,IndexToString, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
class Process:
    def getDataSplit(self,df:DataFrame):

        df = df.select(
            "label","Circulation d’un Transport Exceptionnel sans annonce","Incendie en gare","Restitution d'une Demande de Fermeture de Voie", "Glissement de Terrain","Déraillement sans engagement de la voie principale",                                                                                                                                                                               "Dégagement intempestif du domaine fermé",
            "Autre", "Dérive", "Franchissement de signal", "Non respect du point d’arrêt", "Non alimentation de la conduite principale","Dérive et collision", "Désordre d'ouvrage en terre", "Non mise en protection", "Désordre ouvrage d’art",
            "accident du travail", "Boîte chaude", "Matières dangereuses", "DVL","Dérangement des installations de sécurité", "Circulation sans autorisation", "Restitution DFV sans dégagement", "Accident de voyageur",
            "Modification d'itinéraire sous un train", "Reception sur voie non-electrifiée", "Départ sans autorisation", "Incident grave de passage à niveau","Avarie au matériel", "Déshuntage", "Modification intempestive d'itinéraire",
            "Electrocution", "Dégagement de fumée sur matériel roulant", "Porte non vérroullée", "Collision contre obstacle à un passage à niveau","Incident de frein", "Point limite d’un pantographe dépassé par une circulation électrique",
            "Collision contre obstacle", "Perte d’organe du matériel roulant", "Accident de marchandise dangereuse")

        training, test = df.randomSplit([0.8, 0.2])
        indexed = StringIndexer(inputCol="label", outputCol="labelindexed")
        vec_assembler = VectorAssembler(inputCols=df.columns[1:],outputCol="features")
        layer = [39,10,12]
        mpl = MultilayerPerceptronClassifier(labelCol="labelindexed", featuresCol="features", layers=layer)
        pipe = Pipeline(stages=[indexed,vec_assembler,mpl])
        model_piped = pipe.fit(training)
        predictions = model_piped.transform(test)
        predictions.show(3000)
        evaluator = MulticlassClassificationEvaluator(labelCol="labelindexed", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))











