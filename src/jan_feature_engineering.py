

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_PATH = "data/processed/january_2025/taxi_trips_model.parquet"
OUTPUT_PATH = "data/processed/january_2025/taxi_trips_features.parquet"

spark = (
    SparkSession.builder
    .appName("NYCTaxiFeatureEngineering")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(INPUT_PATH)

# Time-based features
df = (
    df
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn(
        "pickup_day_of_week",
        F.dayofweek("tpep_pickup_datetime")
    )
    .withColumn(
        "is_weekend",
        F.when(F.col("pickup_day_of_week").isin(1, 7), 1).otherwise(0)
    )
    .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
)

# Remove rows without essential model features.
# Invalid categorical values are handled in the model pipeline later.
df = df.filter(
    F.col("trip_distance").isNotNull()
    & (F.col("trip_distance") > 0)
    # & F.col("passenger_count").isNotNull()
    # & (F.col("passenger_count") > 0)
    & F.col("fare_amount").isNotNull()
)

# Keep the target and candidate input columns.
# Actual duration and post-trip monetary fields are retained in the
# dataset for analysis, but will not be used as model inputs.
df.write.mode("overwrite").parquet(OUTPUT_PATH)

print("\n=== Feature Dataset ===")
print(f"Rows: {df.count():,}")
print(f"Columns: {len(df.columns)}")

df.select(
    "tpep_pickup_datetime",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "pickup_month",
    "trip_distance",
    "passenger_count",
    "fare_amount"
).show(10, truncate=False)

print("\n=== Trips by Pickup Hour ===")
df.groupBy("pickup_hour").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 2).alias("average_fare")
).orderBy("pickup_hour").show(24)

print("\n=== Trips by Weekday ===")
df.groupBy("pickup_day_of_week").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 2).alias("average_fare")
).orderBy("pickup_day_of_week").show()

print(f"\nFeatures saved to: {OUTPUT_PATH}")

spark.stop()
