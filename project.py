from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import feature, classification, evaluation

schema = StructType([
    StructField("sample_code", IntegerType(), True),
    StructField("clump_thickness", IntegerType(), True),
    StructField("uniformity_of_cell_size", IntegerType(), True),
    StructField("uniformity_of_cell_shape", IntegerType(), True),
    StructField("marginal_adhesion", IntegerType(), True),
    StructField("single_epithelial_cell_size", IntegerType(), True),
    StructField("bare_nuclei", IntegerType(), True),
    StructField("bland_chromatin", IntegerType(), True),
    StructField("normal_nucleoli", IntegerType(), True),
    StructField("mitoses", IntegerType(), True),
    StructField("class", IntegerType(), True)
])
spark = SparkSession.builder.appName("big_data").getOrCreate()
df = spark.read.csv("breast-cancer-wisconsin.data", header=False, schema=schema)

df = df.na.drop(how='any')

df_train, df_test = df.randomSplit([0.8, 0.2], 125)

feat_vect = feature.VectorAssembler(inputCols=df.columns[1:-1], outputCol='feat')
df_train = feat_vect.transform(df_train).select('class', 'feat')

scaler = feature.StandardScaler(inputCol='feat', outputCol='features')
scaler_ = scaler.fit(df_train)
df_train = scaler_.transform(df_train)

forest = classification.RandomForestClassifier(labelCol='class', maxDepth=8, minInstancesPerNode=5, seed=125)
forest_ = forest.fit(df_train)
df_pred_train = forest_.transform(df_train)

evaluator = evaluation.MulticlassClassificationEvaluator(metricName='accuracy', labelCol='class')
result = evaluator.evaluate(df_pred_train)
print('\nPrawdopodobieństwo poprawnego sklasyfikowania rodzaju nowotworu: {result:.2f}%\n'.format(result=result*100))

df_test = feat_vect.transform(df_test).select('class', 'feat')
df_test = scaler_.transform(df_test)
df_pred_test = forest_.transform(df_test)

correct_prediction_count = df_pred_test.select(['class','prediction']).where(df_pred_test['class'] == df_pred_test['prediction']).count()
print(f'{correct_prediction_count}/{df_test.count()} poprawnie sklasyfikowanych rodzajów nowotworu w zbiorze testowym')
