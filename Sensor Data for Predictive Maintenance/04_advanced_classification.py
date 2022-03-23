# Databricks notebook source
# MAGIC %md
# MAGIC <div >
# MAGIC   <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/solution-accelerator-logo.png"; width="50%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Warning: The Spark-NLP team is currenty resolving an issue with the cross-validator. This notebook will not run until this issue is resolved. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC ### In this notebook you:
# MAGIC * Configure the environment
# MAGIC * Explore the training data
# MAGIC * Prep the training data
# MAGIC * Understand sentence embeddings
# MAGIC * Build embedding and classification pipelines
# MAGIC * Define a parameter grid
# MAGIC * Create cross validator for hyperparameter tuning
# MAGIC * Track model training with MLflow
# MAGIC   * Train
# MAGIC   * Tune
# MAGIC   * Evaluate
# MAGIC   * Register

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 1: Configure the Environment
# MAGIC * To use this notebook, the cluster must be configured to support Spark NLP. Instructions for this configuration can be found [here](https://nlp.johnsnowlabs.com/docs/en/install#install-spark-nlp-on-databricks) and are summarized below.
# MAGIC   * **Use Databricks Runtime:** `7.6 ML`
# MAGIC   
# MAGIC   * **Install libraries:**
# MAGIC     * PyPi: `spark-nlp`
# MAGIC     * Maven Coordinates:
# MAGIC       * CPU: `com.johnsnowlabs.nlp:spark-nlp_2.12:3.0.0`
# MAGIC       * GPU: `com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.0`
# MAGIC   
# MAGIC   * **Add critical Spark config**:
# MAGIC     * `spark.serializer org.apache.spark.serializer.KryoSerializer`
# MAGIC     * `spark.kryoserializer.buffer.max 2000M`
# MAGIC     
# MAGIC   * **A note on cluster instances**: A CPU or GPU cluster can be used to run this notebook. We found that leveraging GPU-based clusters with the GPU spark-nlp jar from maven trained in 1/3rd of the time compared to the CPU-based training

# COMMAND ----------

import sparknlp
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import MultilabelClassificationEvaluator
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters

from pyspark.sql.functions import lit,when,col,array,array_contains,array_remove,regexp_replace,size,when
from pyspark.sql.types import ArrayType,DoubleType,StringType
from pyspark.sql import DataFrame

from mlflow.tracking import MlflowClient
import mlflow
import mlflow.spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Explore the Training Dataset
# MAGIC 
# MAGIC The training dataset from Jigsaw contains 6 columns that denote what labels are associated with a given comment. This is denoted by a 0 for false and a 1 for true.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM Gaming.Toxicity_training WHERE toxic = 0 LIMIT 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Training Data Prep

# COMMAND ----------

dataPrepDF = spark.table("Gaming.Toxicity_training")\
  .withColumnRenamed("toxic","toxic_true")\
  .withColumnRenamed("severe_toxic","severe_toxic_true")\
  .withColumnRenamed("obscene","obscene_true")\
  .withColumnRenamed("threat","threat_true")\
  .withColumnRenamed("insult","insult_true")\
  .withColumnRenamed("identity_hate","identity_hate_true")\
  .withColumn('toxic',when(col('toxic_true') == '1','toxic').otherwise(0))\
  .withColumn('severe_toxic',when(col('severe_toxic_true') == '1','severe_toxic').otherwise(0))\
  .withColumn('obscene',when(col('obscene_true') == '1','obscene').otherwise(0))\
  .withColumn('threat',when(col('threat_true') == '1','threat').otherwise(0))\
  .withColumn('insult',when(col('insult_true') == '1','insult').otherwise(0))\
  .withColumn('identity_hate',when(col('identity_hate_true') == '1','identity_hate').otherwise(0))\
  .withColumn('labels',array_remove(array('toxic','severe_toxic','obscene','threat','insult','identity_hate'),'0')\
              .astype(ArrayType(StringType())))\
  .drop('toxic','severe_toxic','obscene','threat','insult','identity_hate')\
  .withColumn('label_true', array(
    col('toxic_true').cast(DoubleType()),
    col('severe_toxic_true').cast(DoubleType()),
    col('obscene_true').cast(DoubleType()),
    col('threat_true').cast(DoubleType()),
    col('insult_true').cast(DoubleType()),
    col('identity_hate_true').cast(DoubleType()))
  )
  
train, val = dataPrepDF.randomSplit([0.8,0.2],42)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1: Display training data

# COMMAND ----------

display(train.limit(1).filter(size(col('labels')) == 0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Sentence Embeddings Pipelines
# MAGIC 
# MAGIC Lets jump in and build our embeddings pipeline! 
# MAGIC * We are building an embeddings pipeline separate from our classification model to prevent wasted compute on generating the same embeddings repeatedly in the hyperparameter tuning section process.
# MAGIC 
# MAGIC * We are creating two seperate embeddings that are used to add features to the training dataset. You can use one or multiple embeddings depending on your needs.
# MAGIC   * UniversalEmbeddingsPipeline
# MAGIC   * BertEmbeddingsPipeline  
# MAGIC   
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/embeddings-pipeline.png"; width="70%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.1: Universal Sentence Embeddings
# MAGIC 
# MAGIC Per the [documentation from John Snow Labs on Universal Sentence Embeddings](https://nlp.johnsnowlabs.com/2020/04/17/tfhub_use.html):
# MAGIC 
# MAGIC * "The model is trained and optimized for greater-than-word length text, such as sentences, phrases or short paragraphs. It is trained on a variety of data sources and a variety of tasks with the aim of dynamically accommodating a wide variety of natural language understanding tasks. The input is variable length English text and the output is a 512 dimensional vector. We apply this model to the STS benchmark for semantic similarity, and the results can be seen in the example notebook made available. The universal-sentence-encoder model is trained with a deep averaging network (DAN) encoder.""

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.2: BERT LaBSE Sentence Embeddings
# MAGIC 
# MAGIC Per the [documentation from John Snow Labs on BERT LaBSE Sentence Embeddings](https://nlp.johnsnowlabs.com/2020/09/23/labse.html):
# MAGIC * "Language-agnostic BERT sentence embedding model supporting 109 languages."
# MAGIC * "The language-agnostic BERT sentence embedding encodes text into high dimensional vectors. The model is trained and optimized to produce similar representations exclusively for bilingual sentence pairs that are translations of each other. So it can be used for mining for translations of a sentence in a larger corpus."

# COMMAND ----------

document_assembler = DocumentAssembler() \
  .setInputCol("comment_text") \
  .setOutputCol("document")

universal = UniversalSentenceEncoder.pretrained() \
  .setInputCols(["document"]) \
  .setOutputCol("universal_embeddings")  

bert = BertSentenceEmbeddings.pretrained() \
  .setInputCols(["document"]) \
  .setOutputCol("bert_embeddings")  

UniversalEmbeddingsPipeline = Pipeline(stages=[
  document_assembler,
  universal,
  bert
])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4.3: Apply embeddings feature

# COMMAND ----------

embeddingsDF = UniversalEmbeddingsPipeline.fit(train).transform(train)
display(embeddingsDF.limit(1))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 5: Classifier Pipeline
# MAGIC 
# MAGIC Lets jump in and build our classification pipeline!
# MAGIC   
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/classifier-pipeline.png"; width="70%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.1: Classifier Stage
# MAGIC [MultiClassifier DL Approach](https://nlp.johnsnowlabs.com/docs/en/annotators#multiclassifierdl-multi-label-text-classification) is a Multi-label Text Classification. MultiClassifierDL uses a Bidirectional GRU with Convolution model that was built inside TensorFlow.

# COMMAND ----------

ClassifierDL = MultiClassifierDLApproach() \
  .setInputCols(["universal_embeddings"]) \
  .setOutputCol("class") \
  .setLabelColumn("labels") \
  .setMaxEpochs(10) \
  .setLr(1e-3) \
  .setBatchSize(32) \
  .setThreshold(0.7) \
  .setOutputLogsPath('./') \
  .setEnableOutputLogs(False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.2: Custom Transformer Stage
# MAGIC The output of the MultiClassifierDLApproach needs to be converted into a vector for input into the multilabel classifier evaluator.

# COMMAND ----------

class CustomTransformer(Transformer, HasInputCol, HasOutputCol):
  input_col = Param(Params._dummy(), "input_col", "input column name.", typeConverter=TypeConverters.toString)
  output_col = Param(Params._dummy(), "output_col", "output column name.", typeConverter=TypeConverters.toString)
  
  @keyword_only
  def __init__(self, input_col: str = "input", output_col: str = "output"):
    super(CustomTransformer, self).__init__()
    self._setDefault(input_col=None, output_col=None)
    kwargs = self._input_kwargs
    self.set_params(**kwargs)
    
  @keyword_only
  def set_params(self, input_col: str = "input", output_col: str = "output"):
    kwargs = self._input_kwargs
    self._set(**kwargs)
    
  def get_input_col(self):
    return self.getOrDefault(self.input_col)
  
  def get_output_col(self):
    return self.getOrDefault(self.output_col)
  
  def _transform(self, df: DataFrame):
    input_col = self.get_input_col()
    output_col = self.get_output_col()
    
    # The custom action: create array of doubles from the class.result to leverage in the MultilabelClassificationEvaluator
    df = df.withColumn(output_col, array(
        when(array_contains(col(input_col),'toxic'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col(input_col),'severe_toxic'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col(input_col),'obscene'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col(input_col),'threat'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col(input_col),'insult'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col(input_col),'identity_hate'),1).otherwise(0).cast(DoubleType())
      )
    )
    
    print(df)

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.3: Instantiate Transformer Stage

# COMMAND ----------

custom_transformer = CustomTransformer(input_col="class.result", output_col="label_pred")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.4: Build the classifier & transformer pipeline

# COMMAND ----------

ClassificationPipeline = Pipeline(stages=[
  ClassifierDL,
  custom_transformer
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Build the parameter grid
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/ml-tuning.html

# COMMAND ----------

#The Spark-NLP team are actively working on an issue with ParamGridBuilder & CrossValidator. This code will fail to run until the fix has been applied. Run the below code that does not leverage any parameters in the grid.
ParamGrid = ParamGridBuilder()\ 
  .addGrid(classifierdl.InputCols, ["bert_embeddings","universal_embeddings"])\
  .addGrid(classifierdl.batchSize, [32,64]). \
  .addGrid(classifierdl.setThreshold, [0.7, 0.1]) \
  .build()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Multilabel Classification Training
# MAGIC 
# MAGIC Steps in the classification workflow
# MAGIC * Start tracking the runs & enable [autologging for spark](https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html)
# MAGIC 
# MAGIC * Leveraging the CrossValidator, the process begins by splitting the dataset into a set of folds which are used as separate training and test datasets. 
# MAGIC   * E.g. with k=5 folds, CrossValidator will generate 5 (training, test) dataset pairs, each of which uses 4/5 of the data for training and 1/5 for testing. To evaluate a particular ParamMap, CrossValidator computes the average evaluation metric for the 5 Models produced by fitting the Estimator on the 5 different (training, test) dataset pairs.
# MAGIC 
# MAGIC * After identifying the best ParamMap, CrossValidator re-fits the Estimator using the best ParamMap and the entire dataset.
# MAGIC 
# MAGIC * We then log the best model from the CrossValidator into MLflow's model registry to manage the models lifecycle post experimentation
# MAGIC 
# MAGIC * After, we evaluate the model with the test data and log the model metrics.
# MAGIC 
# MAGIC * As a final step, we end the tracked run. Results will then be viewable in the experiments tab on the top right of the UI.
# MAGIC 
# MAGIC 
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow.png"; width="60%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 7.1: Create Experiment & Start Autologging
# MAGIC 
# MAGIC We want experiments to persist outside of this notebook and to allow others to collaborate with their work on the same project.
# MAGIC * Create experiment in users folder to hold model artifacts and parameters
# MAGIC 
# MAGIC Note: When running this code for production change the experiment path to a location outside of a users personal folder.

# COMMAND ----------

# Create an experiment with a name that is unique and case sensitive.
client = MlflowClient()

mlflow.set_experiment(f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')}/Toxicity_Classification")

mlflow.spark.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC Under your user folder, you will find an experiment created to log the runs with parameters and metrics. The model will also be logged in the model registry during the run, similar to the image on the right.
# MAGIC <div><img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow-experiments.png"; width="55%">     ->     <img src="https://cme-solution-accelerators-images.s3-us-west-2.amazonaws.com/toxicity/mlflow-model-registry.png"; width="40%"></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 7.2: Training with Cross-Validation
# MAGIC 
# MAGIC CrossValidator begins by splitting the dataset into a set of folds which are used as separate training and test datasets. 
# MAGIC * E.g. with k=3 folds, CrossValidator will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. To evaluate a particular ParamMap, CrossValidator computes the average evaluation metric for the 3 Models produced by fitting the Estimator on the 3 different (training, test) dataset pairs.
# MAGIC 
# MAGIC [Documentation](https://spark.apache.org/docs/latest/ml-tuning.html)

# COMMAND ----------

with mlflow.start_run():
  
  evaluator = MultilabelClassificationEvaluator(labelCol="label_true",predictionCol="label_pred")
  
  cv = CrossValidator(estimator=ClassificationPipeline, \
                      estimatorParamMaps=ParamGrid, \
                      evaluator=evaluator, \
                      numFolds=2) #3+ folds for production
  
  cvModel = cv.fit(train)
  
  mlflow.spark.log_model(cvModel.bestModel,"spark-model",registered_model_name='Toxicity MultiLabel Classification')
 
  predictions = cvModel.transform(val)\
    .withColumn('label_pred', array(
        when(array_contains(col('class.result'),'toxic'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'severe_toxic'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'obscene'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'threat'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'insult'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'identity_hate'),1).otherwise(0).cast(DoubleType())
      )
    )
  
  # Evaluate best model
  score = evaluator.evaluate(predictions)
  mlflow.log_metric('f1',score)
  print(score)
  
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: Training with TrainValidationSplit
# MAGIC Using TrainValidationSplit instead of Cross Validator
# MAGIC 
# MAGIC An alternative to CrossValidator for hyper-parameter tuning is TrainValidationSplit. TrainValidationSplit only evaluates each combination of parameters once, as opposed to k times in the case of CrossValidator. TrainValidationSplit is less expensive to run but will not produce as reliable results when the training dataset is not sufficiently large.
# MAGIC 
# MAGIC [Documentation](https://spark.apache.org/docs/latest/ml-tuning.html)

# COMMAND ----------

with mlflow.start_run():
  
  cv = TrainValidationSplit(estimator=ClassificationPipeline, \
                      estimatorParamMaps=ParamGrid, \
                      evaluator=MultilabelClassificationEvaluator(labelCol="label_true",predictionCol="label_pred")
                           )
  
  cvModel = cv.fit(embeddingsDF)
  
  mlflow.spark.log_model(cvModel,"spark-model",registered_model_name='Toxicity MultiLabel Classification')
 
  predictions = cvModel.transform(val)\
    .withColumn('label_pred', array(
        when(array_contains(col('class.result'),'toxic'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'severe_toxic'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'obscene'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'threat'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'insult'),1).otherwise(0).cast(DoubleType()),
        when(array_contains(col('class.result'),'identity_hate'),1).otherwise(0).cast(DoubleType())
      )
    )
  
  # Evaluate best model
  score = evaluator.evaluate(predictions)
  mlflow.log_metric('f1',score)
  print(score)
  
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Create inference pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Spark-nlp|Apache-2.0 License| https://nlp.johnsnowlabs.com/license.html | https://www.johnsnowlabs.com/
# MAGIC |Kaggle|Apache-2.0 License |https://github.com/Kaggle/kaggle-api/blob/master/LICENSE|https://github.com/Kaggle/kaggle-api|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
