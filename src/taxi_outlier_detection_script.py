
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

df.orderBy(
    F.col("trip_duration_minutes").desc()
).select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_duration_minutes",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "pulocationid",
    "dolocationid",
).show(15, truncate=False)

spark.stop()