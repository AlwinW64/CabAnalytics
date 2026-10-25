

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_PATH = "data/processed/january_2025/taxi_trips_clean.parquet"

spark = (
    SparkSession.builder
    .appName("NYCTaxiFareOutlierInvestigation")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(INPUT_PATH)

print("\n=== Fare Threshold Counts ===")
df.agg(
    F.count("*").alias("total_trips"),
    F.sum(F.when(F.col("fare_amount") > 100, 1).otherwise(0)).alias("fare_over_100"),
    F.sum(F.when(F.col("fare_amount") > 200, 1).otherwise(0)).alias("fare_over_200"),
    F.sum(F.when(F.col("fare_amount") > 500, 1).otherwise(0)).alias("fare_over_500"),
    F.sum(F.when(F.col("fare_amount") > 1000, 1).otherwise(0)).alias("fare_over_1000"),
    F.sum(F.when(F.col("fare_amount") > 5000, 1).otherwise(0)).alias("fare_over_5000")
).show(truncate=False)

print("\n=== Highest Fare Records ===")
df.orderBy(F.col("fare_amount").desc()).select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "trip_duration_minutes",
    "fare_amount",
    "total_amount",
    "pulocationid",
    "dolocationid"
).show(20, truncate=False)

print("\n=== Fare Distribution ===")
df.select("fare_amount").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "90%", "95%", "99%", "max"
).show(truncate=False)

spark.stop()
