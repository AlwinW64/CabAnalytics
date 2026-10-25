

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_PATH = "data/processed/january_2025/taxi_trips_clean.parquet"
OUTPUT_PATH = "data/processed/january_2025/taxi_trips_model.parquet"

spark = (
    SparkSession.builder
    .appName("NYCTaxiModelDataPreparation")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(INPUT_PATH)

# Exclude extreme fare targets from the initial model.
model_df = df.filter(
    (F.col("fare_amount") > 0)
    & (F.col("fare_amount") <= 200)
    & (F.col("trip_distance") > 0)
    & (F.col("trip_distance") <= 100)
    & (F.col("trip_duration_minutes") > 0)
    & (F.col("trip_duration_minutes") <= 180)
)

print("\n=== Model Dataset Summary ===")
print(f"Cleaned input rows: {df.count():,}")
print(f"Model-ready rows: {model_df.count():,}")

print("\n=== Fare Statistics ===")
model_df.select("fare_amount").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "90%", "95%", "99%", "max"
).show(truncate=False)

print("\n=== Date Coverage ===")
model_df.select(
    F.min("tpep_pickup_datetime").alias("first_pickup"),
    F.max("tpep_pickup_datetime").alias("last_pickup"),
    F.countDistinct(F.to_date("tpep_pickup_datetime")).alias("days_covered")
).show(truncate=False)

model_df.write.mode("overwrite").parquet(OUTPUT_PATH)

print(f"\nModel-ready dataset saved to: {OUTPUT_PATH}")

spark.stop()
