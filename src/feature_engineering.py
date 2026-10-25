

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("TaxiFeatureEngineering")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = "data/processed/processed1/nyc_taxi_cleaned.parquet"
OUTPUT_PATH = "data/processed/processed1/nyc_taxi_features.parquet"

df = spark.read.parquet(INPUT_PATH)

# Derive temporal features from pickup timestamp
features_df = (
    df
    .withColumn(
        "pickup_hour",
        F.hour("tpep_pickup_datetime")
    )
    .withColumn(
        "pickup_day_of_week",
        F.dayofweek("tpep_pickup_datetime")
    )
    .withColumn(
        "is_weekend",
        F.when(F.col("pickup_day_of_week").isin(1, 7), 1)
         .otherwise(0)
    )
    .withColumn(
        "pickup_month",
        F.month("tpep_pickup_datetime")
    )
    .withColumn(
        "fare_per_mile",
        F.col("fare_amount") / F.col("trip_distance")
    )
)

# Select fields needed for analysis and modeling
features_df = features_df.select(
    "tpep_pickup_datetime",
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "pulocationid",
    "dolocationid",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "pickup_month",
    "fare_amount",
    "total_amount",
    "fare_per_mile"
)

features_df.write.mode("overwrite").parquet(OUTPUT_PATH)

print("Feature engineering complete.")
print("Record count:", features_df.count())
features_df.printSchema()

features_df.select(
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "fare_per_mile"
).show(10, truncate=False)

spark.stop()
