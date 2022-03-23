# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC This notebook performs exploratory data analysis on the dataset.
# MAGIC To expand on the analysis, attach this notebook to the **hh_demo_ml** cluster,
# MAGIC edit [the options of pandas-profiling](https://pandas-profiling.github.io/pandas-profiling/docs/master/rtd/pages/advanced_usage.html), and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/3588141836682562/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/3588141836682560) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC 
# MAGIC Runtime Version: _9.1.x-cpu-ml-scala2.12_

# COMMAND ----------

# MAGIC %md
# MAGIC > **NOTE:** The dataset loaded below is a sample of the original dataset.
# MAGIC Stratified sampling using pyspark's [sampleBy](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrameStatFunctions.sampleBy.html)
# MAGIC method is used to ensure that the distribution of the target column is retained.
# MAGIC <br/>
# MAGIC > Rows were sampled with a sampling fraction of **0.8842107820133087**

# COMMAND ----------

import os
import uuid
import shutil
import pandas as pd
import databricks.automl_runtime

from mlflow.tracking import MlflowClient

# Download input data from mlflow into a pandas DataFrame
# create temp directory to download data
temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(temp_dir)

# download the artifact and read it
client = MlflowClient()
training_data_path = client.download_artifacts("b5bb025b79bd4034bf8a9f62e7ed5532", "data", temp_dir)
df = pd.read_parquet(os.path.join(training_data_path, "training_data"))

# delete the temp data
shutil.rmtree(temp_dir)

target_col = "machine_status"

# Convert columns detected to be of semantic type datetime
datetime_columns = ["timestamp"]
df[datetime_columns] = df[datetime_columns].apply(pd.to_datetime, errors="coerce")

# Convert columns detected to be of semantic type numeric
numeric_columns = ["_c0", "sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06", "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21", "sensor_22", "sensor_23", "sensor_24", "sensor_25", "sensor_26", "sensor_27", "sensor_28", "sensor_29", "sensor_30", "sensor_31", "sensor_32", "sensor_33", "sensor_34", "sensor_35", "sensor_36", "sensor_37", "sensor_38", "sensor_39", "sensor_40", "sensor_41", "sensor_42", "sensor_43", "sensor_44", "sensor_45", "sensor_46", "sensor_47", "sensor_48", "sensor_49", "sensor_50", "sensor_51"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df, minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

import missingno as msno

  
# Visualize missing values as a matrix
msno.matrix(df)

# COMMAND ----------

msno.heatmap(df)

# COMMAND ----------


