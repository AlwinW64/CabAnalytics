

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

DATA_PATH = "data/processed/january_2025/taxi_trips_features.parquet"
# MODEL_PATH = "models/january_2025_linear_regression"

import os

MODEL_PATH = os.path.abspath(
    "models/january_2025_linear_regression"
).replace("\\", "/")
MODEL_PATH = "file:///" + MODEL_PATH



spark = (
    SparkSession.builder
    .appName("NYCTaxiModelDiagnostics")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(DATA_PATH)

df = df.select(
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pulocationid",
    "dolocationid",
    "fare_amount"
).filter(
    F.col("fare_amount").isNotNull()
    & F.col("trip_distance").isNotNull()
    & (F.col("trip_distance") > 0)
    & F.col("pickup_hour").isNotNull()
    & F.col("pickup_day_of_week").isNotNull()
)

_, test_df = df.randomSplit([0.8, 0.2], seed=42)

model = PipelineModel.load(MODEL_PATH)
predictions = model.transform(test_df)

# Absolute error for each prediction.
predictions = predictions.withColumn(
    "absolute_error",
    F.abs(F.col("fare_amount") - F.col("prediction"))
)

print("\n=== Error by Trip Distance ===")

predictions.withColumn(
    "distance_band",
    F.when(F.col("trip_distance") < 1, "Under 1 mile")
    .when(F.col("trip_distance") < 3, "1 to under 3 miles")
    .when(F.col("trip_distance") < 5, "3 to under 5 miles")
    .when(F.col("trip_distance") < 10, "5 to under 10 miles")
    .otherwise("10 miles or more")
).groupBy("distance_band").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("absolute_error"), 2).alias("mae"),
    F.round(
        F.sqrt(F.avg(F.pow(
            F.col("fare_amount") - F.col("prediction"), 2
        ))),
        2
    ).alias("rmse")
).orderBy(
    F.min("trip_distance")
).show(truncate=False)

print("\n=== Largest Prediction Errors ===")

predictions.select(
    "fare_amount",
    "prediction",
    "absolute_error",
    "trip_distance",
    "pickup_hour",
    "pulocationid",
    "dolocationid"
).orderBy(
    F.col("absolute_error").desc()
).show(20, truncate=False)

spark.stop()
