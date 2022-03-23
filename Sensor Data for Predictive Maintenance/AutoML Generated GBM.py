# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM training
# MAGIC This is an auto-generated notebook. To reproduce these results, attach this notebook to the **hh_demo_ml** cluster and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/3588141836682562/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Navigate to the parent notebook [here](#notebook/3588141836682560) (If you launched the AutoML experiment using the Experiments UI, this link isn't very useful.)
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.
# MAGIC 
# MAGIC Runtime Version: _9.1.x-cpu-ml-scala2.12_

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# Use MLflow to track experiments
mlflow.set_experiment("/Users/hayley.horn@databricks.com/databricks_automl/machine_status_bronze_sensor_data-2022_01_22-20_33")

target_col = "machine_status"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC > **NOTE:** The dataset loaded below is a sample of the original dataset.
# MAGIC Stratified sampling using pyspark's [sampleBy](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrameStatFunctions.sampleBy.html)
# MAGIC method is used to ensure that the distribution of the target column is retained.
# MAGIC <br/>
# MAGIC > Rows were sampled with a sampling fraction of **0.8842107820133087**

# COMMAND ----------

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_client = MlflowClient()
input_data_path = input_client.download_artifacts("b5bb025b79bd4034bf8a9f62e7ed5532", "data", input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC %md ### Datetime Preprocessor
# MAGIC For each datetime column, extract relevant information from the date:
# MAGIC - Unix timestamp
# MAGIC - whether the date is a weekend
# MAGIC - whether the date is a holiday
# MAGIC 
# MAGIC Additionally, extract extra information from columns with timestamps:
# MAGIC - hour of the day (one-hot encoded)
# MAGIC 
# MAGIC For cyclic features, plot the values along a unit circle to encode temporal proximity:
# MAGIC - hour of the day
# MAGIC - hours since the beginning of the week
# MAGIC - hours since the beginning of the month
# MAGIC - hours since the beginning of the year

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from databricks.automl_runtime.sklearn import TimestampTransformer

for col in ["timestamp"]:
    timestamp_transformer = TimestampTransformer()
    ohe_transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), [timestamp_transformer.HOUR_COLUMN_INDEX])],
        remainder="passthrough")
    timestamp_preprocessor = Pipeline([
        ("extractor", timestamp_transformer),
        ("onehot_encoder", ohe_transformer)
    ])
    transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean for consistency

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer", SimpleImputer(strategy="mean"))
])

transformers.append(("numerical", numerical_pipeline, ["_c0", "sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06", "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21", "sensor_22", "sensor_23", "sensor_24", "sensor_25", "sensor_26", "sensor_27", "sensor_28", "sensor_29", "sensor_30", "sensor_31", "sensor_32", "sensor_33", "sensor_34", "sensor_35", "sensor_36", "sensor_37", "sensor_38", "sensor_39", "sensor_40", "sensor_41", "sensor_42", "sensor_43", "sensor_44", "sensor_45", "sensor_46", "sensor_47", "sensor_48", "sensor_49", "sensor_50", "sensor_51"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training - Validation Split
# MAGIC Split the input data into training and validation data

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=501761183, stratify=split_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3588141836682562/s?orderByKey=metrics.%60val_f1_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from lightgbm import LGBMClassifier

help(LGBMClassifier)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display="diagram")

lgbmc_classifier = LGBMClassifier(
  colsample_bytree=0.7933999697822255,
  lambda_l1=0.002681807610747215,
  lambda_l2=0.2532599221069806,
  learning_rate=0.3396598066822584,
  max_bin=32,
  min_child_samples=2,
  n_estimators=404,
  num_leaves=438,
  subsample=0.9555687933629089,
  random_state=501761183,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", lgbmc_classifier),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="lightgbm") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    lgbmc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                                prefix="val_")
    display(pd.DataFrame(lgbmc_val_metrics, index=[0]))

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None
None

import holidays
import lightgbm
from mlflow.tracking import MlflowClient

lgbmc_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(lgbmc_temp_dir)
lgbmc_client = MlflowClient()
lgbmc_model_env_path = lgbmc_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", lgbmc_temp_dir)
lgbmc_model_env_str = open(lgbmc_model_env_path)
lgbmc_parsed_model_env_str = yaml.load(lgbmc_model_env_str, Loader=yaml.FullLoader)

lgbmc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"holidays=={holidays.__version__}")
lgbmc_parsed_model_env_str["dependencies"][-1]["pip"].append(f"lightgbm=={lightgbm.__version__}")

with open(lgbmc_model_env_path, "w") as f:
  f.write(yaml.dump(lgbmc_parsed_model_env_str))
lgbmc_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=lgbmc_model_env_path, artifact_path="model")
shutil.rmtree(lgbmc_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC 
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC 
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC 
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC 
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")
