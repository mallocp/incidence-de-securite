from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
import numpy as np


class logis:

    lr = LogisticRegression()
    evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

    grid = tune.ParamGridBuilder()
    grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
    grid = grid.addGrid(lr.elasticNetParam, [0, 1])
    grid = grid.build()

    cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator
                         )


    #best_lr = cv.fit(dataSplit[0])

    #test_results = best_lr.transform(dataSplit[1])

    #print(evaluator.evaluate(test_results))


