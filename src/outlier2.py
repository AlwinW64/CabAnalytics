from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("TaxiOutlierInvestigation")
    .getOrCreate()
)


df = spark.read.parquet(
    "data/processed/processed1/nyc_taxi_cleaned.parquet"
)

# Inspect unusually long-duration trips
df.filter(
    F.col("trip_duration_minutes") > 180
).select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_duration_minutes",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "pulocationid",
    "dolocationid"
).orderBy(
    F.desc("trip_duration_minutes")
).show(50, truncate=False)

# Count records in duration bands
df.select(
    F.count("*").alias("total_records"),
    F.sum(
        F.when(F.col("trip_duration_minutes") > 60, 1).otherwise(0)
    ).alias("over_60_minutes"),
    F.sum(
        F.when(F.col("trip_duration_minutes") > 180, 1).otherwise(0)
    ).alias("over_180_minutes"),
    F.sum(
        F.when(F.col("trip_duration_minutes") > 720, 1).otherwise(0)
    ).alias("over_12_hours")
).show()