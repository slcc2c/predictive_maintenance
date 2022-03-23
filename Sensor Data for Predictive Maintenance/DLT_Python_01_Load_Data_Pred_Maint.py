# Databricks notebook source
# MAGIC %md
# MAGIC Databricks not quite ready for primetime accelerator for Predictive Maintenance

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview
# MAGIC 
# MAGIC Unplanned maintenance is commonly the result of equipment failure that was not anticipated. Unexpected equipment failure costs Energy companies an estimated $38 million in losses per year. Using sensors to remotely equipment enables companies to reduce losses by enabling proactive maintenance prior to failure. 
# MAGIC 
# MAGIC There should be another paragraph or more sentences here. We will end with something like the following sentence: In this solution accelerator, we use data from  51 sensors on a piece of machinary to predict the probability of a breakdown.
# MAGIC 
# MAGIC 
# MAGIC ** Authors**
# MAGIC - Hayley Horn [<hayley.horn@databricks.com>]
# MAGIC - Spencer Cook [<spencer.cook@databeicks.com>]

# COMMAND ----------

# MAGIC %md
# MAGIC ## About This Series of Notebooks
# MAGIC 
# MAGIC * This series of notebooks is intended to help you ingest IOT data to use with classification algorithms to predict equipment failure.
# MAGIC 
# MAGIC * In support of this goal, we will:
# MAGIC  * Load labeled sensor data from a piece of industrial equipment that experienced multiple types of failure
# MAGIC  * Create one pipeline for streaming and batch to predict failure in near real-time and/or on an ad-hoc basis. 
# MAGIC        * This pipeline can then be used to monitor and alert operators to imminent failure and plan proactive maintenance.
# MAGIC  * Create alerts and a dashboard for monitoring equipment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## About the Data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Predictive Maintenance Dataset
# MAGIC 
# MAGIC The predictive maintenance dataset used in this accelerator is a synthetic dataset that reflects real predictive maintenance encountered in industry created by Stephan Matzka in 2020. It was published as 'Explainable Artificial Intelligence for Predictive Maintenance Applications' at the Third International Conference on Artificial Intelligence for Industries (AI4I 2020).  <br>
# MAGIC <br>
# MAGIC 
# MAGIC The dataset was donated to UCI Machine Learning Repository and made available as a csv file containing 10,000 data points with 14 feature columns.
# MAGIC 
# MAGIC 
# MAGIC * There is a  binary label column, 'Machine Failure' which indicates failure of any type
# MAGIC * There are also five machine failure subcategory columns, each with binary values indicating the type of failure:
# MAGIC   * Tool Wear Failure(TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), overstrain failure (OSF), Random Failure (RNF)
# MAGIC   
# MAGIC Further details about this dataset
# MAGIC   * Dataset title: The AI4I 2020 Predictive Maintenance Dataset
# MAGIC   * Dataset source URL: https://archive-beta.ics.uci.edu/ml/datasets/ai4i+2020+predictive+maintenance+dataset
# MAGIC   * Dataset source description: Synthetic Multivariate, Time-Series dataset
# MAGIC   * Dataset license: Creative Commons Attribution 4.0 International
# MAGIC   * Relevant Papers:Stephan Matzka, 'Explainable Artificial Intelligence for Predictive Maintenance Applications', Third International Conference on Artificial Intelligence for Industries (AI4I 2020), 2020

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
driver_to_dbfs_path = 'dbfs:/home/{}/predictive_maintenance/ai4i2020.csv'.format(user)
dbutils.fs.cp('file:/databricks/driver/ai4i2020.csv', driver_to_dbfs_path)

# COMMAND ----------

@dlt.create_table()
def bronze_pm_data():
  return (
    spark.readStream.format("cloudFiles")
      .option("cloudFiles.format", "csv")
      .load('dbfs:/home/spencer.cook@databricks.com/predictive_maintenance/ai4i2020.csv')
  )

# COMMAND ----------

# MAGIC %md #####1.6 Check for missing numbers

# COMMAND ----------

import missingno as msno

msno.matrix(bronze_df.toPandas())


# COMMAND ----------

# MAGIC %md Due to the data being synthetic, it is fairly clean, but we will need to create separate silver tables for the different classification problems.

# COMMAND ----------

@dlt.table()
def silver_binaryClass_pm_data():
  return (
    spark.read('bronze_pm_data').select('ProductID', 'Type', 'AirTemp_K','ProcessTemp_K','RotationalSpeed_RPM','Torque_NM','ToolWear_Min','MachineFailure').
  )


# COMMAND ----------

from pyspark.sql.functions import concat,col

# silver_m_df = bronze_df.withColumn("MachineFailureType", con(bronze_df.TWF, bronze_df.HDF,bronze_df.PWF,bronze_df.OSF,bronze_df.RNF)).drop("MachineFailure","TWF","HDF","PWF","OSF","RNF")

silver_m_df=bronze_df.select(concat(bronze_df.TWF, bronze_df.HDF,bronze_df.PWF,bronze_df.OSF,bronze_df.RNF).alias("MachineFailureType"),"ProductID","Type","AirTemp_K","ProcessTemp_K","RotationalSpeed_RPM","Torque_NM","ToolWear_Min")
              
         
silver_m_df.display()


silver_m_df.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("predictive_maintenance.silver_multiClass_pm_data")

# Save table off as a JSON file
df_final.coalesce(1).write.format('json').save('/path/file_name.json')

# COMMAND ----------

# MAGIC %md #####4.2 Accelerate EDA using Databricks AutoML
# MAGIC 
# MAGIC While Databricks AutoML is used to quickly generate baseline models and notebooks for citizen data scientists and ML experts, it also generates an EDA notebook that can accelerate the data exploration process. Our dataset contains a label field, so it is ready load into the AutoML process. If the dataset did not include a label, we could easily add a field to the bronze sendor data table using the [ALTER TABLE](https://docs.databricks.com/spark/2.x/spark-sql/language-manual/alter-table-or-view.html#add-columns) command in SQL.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC * In the next notebook, we will do automl or something

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|

# COMMAND ----------


