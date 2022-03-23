# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1: Download the data
# MAGIC 
# MAGIC In this step, we will:
# MAGIC   1. Download the Predictive Maintenance Dataset
# MAGIC   2. Move the data into object storage.
# MAGIC   3. Create a database and table structure for the data.
# MAGIC   4. Write the data into Delta format.
# MAGIC   5. Create tables for easy access and querability.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.1: Download Predictive Maintenance Dataset

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.2:  Move Predictive Maintenance Dataset into object storage
# MAGIC Move the sensor data from the driver node to object storage

# COMMAND ----------



# COMMAND ----------

import time
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
driver_to_dbfs_path = 'dbfs:/home/{}/predictive_maintenance/ai4i2020_{}.csv'.format(user, int( time.time() ))
dbutils.fs.cp('file:/databricks/driver/ai4i2020.csv', driver_to_dbfs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1.3: Create a database and table structure for the data.

# COMMAND ----------

# Load libraries
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType,StructField,DoubleType, StringType, IntegerType, FloatType

# Set database name, file paths, and table names
database_name = 'predictive_maintenance'

# Set table paths the Delta tables
bronze_tbl_path = '/home/{}/predictive_maintenance/bronze/'.format(user)
silver_tbl_path = '/home/{}/predictive_maintenance/silver/'.format(user)
gold_tbl_path = '/home/{}/predictive_maintenance/gold/'.format(user)
pm_preds_path = '/home/{}/predictive_maintenance/preds/'.format(user)

# Set Delta table names
bronze_tbl_name = 'bronze_pm_data'
silver_b_tbl_name = 'silver_binaryClass_pm_data'
silver_mc_tbl_name = 'silver_multiClass_pm_data'
gold_tbl_name = 'gold_pm_data'
pm_preds_tbl_name = 'pm_preds'

# Delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS {} CASCADE'.format(database_name))

# Create database to house tables
_ = spark.sql('CREATE DATABASE {}'.format(database_name))

# Drop any old delta lake files if needed (e.g. re-running this notebook with the same bronze_tbl_path and silver_tbl_path)
shutil.rmtree('/dbfs'+bronze_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+silver_tbl_path, ignore_errors=True)
shutil.rmtree('/dbfs'+pm_preds_path, ignore_errors=True)

# COMMAND ----------

# MAGIC %md ##### 1.4: Write the data into a pyspark datafram and a Delta table

# COMMAND ----------

schema = StructType([StructField('RowID', StringType(), True),
                     StructField('ProductID', StringType(), True),
                     StructField('Type', StringType(), True),
                     StructField('AirTemp_K', FloatType(), True),
                     StructField('ProcessTemp_K', FloatType(), True),
                     StructField('RotationalSpeed_RPM', IntegerType(), True),
                     StructField('Torque_NM', FloatType(), True),
                     StructField('ToolWear_Min', IntegerType(), True),
                     StructField('MachineFailure', IntegerType(), True),
                     StructField('TWF', IntegerType(), True),
                     StructField('HDF', IntegerType(), True),
                     StructField('PWF', IntegerType(), True),
                     StructField('OSF', IntegerType(), True),                    
                     StructField('RNF', IntegerType(), True)])

bronze_df = spark.read.csv(driver_to_dbfs_path,header=True,escape='"',schema=schema,multiLine=True)
bronze_df.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("predictive_maintenance.bronze_pm_data")

display(bronze_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Under the Data tab, you will see a predictive_maintenance database with a table named bronze_pm_data. Although we have defined the table names and paths of other tables, they will not be populated until after the data is explored and refined.

# COMMAND ----------

# MAGIC %md #####1.5 View shape of dataset and summary statistics

# COMMAND ----------

print("There are",len(bronze_df.columns) ,"columns and",bronze_df.count(),"rows in this dataset")
bronze_df.select(col("AirTemp_K"),col("ProcessTemp_K"),col("RotationalSpeed_RPM"),col("Torque_NM"), col("ToolWear_Min"), col("MachineFailure"),col("TWF"),col("HDF"),col("PWF"),col("OSF"),col("RNF")).summary("count", "mean", "stddev", "min", "max").display()

# COMMAND ----------

# MAGIC %md #####1.6 Check for missing numbers

# COMMAND ----------

import missingno as msno

msno.matrix(bronze_df.toPandas())


# COMMAND ----------

# MAGIC %md Due to the data being synthetic, it is fairly clean, but we will need to create separate silver tables for the different classification problems.

# COMMAND ----------

silver_b_df = spark.sql("SELECT ProductID, Type, AirTemp_K,ProcessTemp_K,RotationalSpeed_RPM,Torque_NM,ToolWear_Min,MachineFailure FROM predictive_maintenance.bronze_pm_data")
silver_b_df.show()

silver_b_df.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("predictive_maintenance.silver_binaryClass_pm_data")

# Save table off as a JSON file- create a stream
silver_b_df.coalesce(1).write.format('json').save('/path/file_name.json')

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


