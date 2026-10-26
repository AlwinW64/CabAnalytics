
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "january_2025"
    / "taxi_trips_features.parquet"
)
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("CabAnalytics-PredictionPlots")
    .master("local[4]")
    .config("spark.driver.memory", "6g")
    .config("spark.driver.maxResultSize", "1g")
    .getOrCreate()
)

try:
    df = spark.read.parquet(str(DATA_PATH))

    columns = [
        "fare_amount",
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pulocationid",
        "dolocationid",
    ]
    df = df.select(*columns).dropna()

    # Reproduce the reported split.
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    lr_model = PipelineModel.load(
        str(MODEL_DIR / "january_2025_linear_regression")
    )
    rf_model = PipelineModel.load(
        str(MODEL_DIR / "january_2025_random_forest")
    )

    # Score both models on the same test DataFrame.
    lr_scored = lr_model.transform(test_df).select(
        "fare_amount",
        F.col("prediction").alias("lr_prediction"),
    )

    rf_scored = rf_model.transform(test_df).select(
        F.col("prediction").alias("rf_prediction"),
    )

    # Both transforms preserve row order for this same input DataFrame.
    # Zip the outputs using a temporary row ID instead for a guaranteed join.
    # The robust approach is to add the ID before scoring both models.
    # See the corrected version in the next step before running.
finally:
    spark.stop()
