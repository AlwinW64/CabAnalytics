

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("TaxiDataCleaning")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = "data/processed/processed1/nyc_taxi_10000.parquet"
OUTPUT_PATH = "data/processed/processed1/nyc_taxi_cleaned.parquet"

df = spark.read.parquet(INPUT_PATH)

# Calculate trip duration in minutes.
df = df.withColumn(
    "trip_duration_minutes",
    (
        F.col("tpep_dropoff_datetime").cast("long")
        - F.col("tpep_pickup_datetime").cast("long")
    ) / 60.0
)

original_count = df.count()

# Define explicit eligibility rules for fare modeling.
valid_condition = (
    (F.col("fare_amount") > 0)
    & (F.col("trip_distance") > 0)
    & (F.col("trip_duration_minutes") > 0)
)

clean_df = df.filter(valid_condition)


# Exclude only trips with durations exceeding 3 hours
# from the primary modeling dataset.
clean_df = clean_df.filter(
    F.col("trip_duration_minutes") <= 180
)

print("Before duration filtering:", clean_df.count())
print("After duration filtering:", clean_df.count())
print("Excluded:", clean_df.count() - clean_df.count())


clean_count = clean_df.count()
removed_count = original_count - clean_count

print(f"Original records: {original_count:,}")
print(f"Records retained: {clean_count:,}")
print(f"Records excluded: {removed_count:,}")
print(
    f"Retention rate: "
    f"{clean_count / original_count * 100:.2f}%"
)

print("\nCleaned data summary:")
clean_df.select(
    "trip_distance",
    "trip_duration_minutes",
    "fare_amount",
    "total_amount",
).summary(
    "count", "min", "25%", "50%", "75%", "max", "mean"
).show(truncate=False)

# Check that the cleaned dataset satisfies the rules.
violations = clean_df.filter(
    (F.col("fare_amount") <= 0)
    | (F.col("trip_distance") <= 0)
    | (F.col("trip_duration_minutes") <= 0)
).count()

print("Remaining rule violations:", violations)

# Save a separate dataset; preserve the raw and original Parquet data.
(
    clean_df.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

clean_df.select(
    F.min("tpep_pickup_datetime").alias("first_pickup"),
    F.max("tpep_pickup_datetime").alias("last_pickup"),
).show(truncate=False)

print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")

spark.stop()
