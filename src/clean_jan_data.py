

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

INPUT_PATH = "data/processed/january_2025/taxi_trips.parquet"
OUTPUT_PATH = "data/processed/january_2025/taxi_trips_clean.parquet"

spark = (
    SparkSession.builder
    .appName("NYCTaxiJanuaryCleaning")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(INPUT_PATH)

# Remove metadata/corrupt rows if the old Parquet still contains them.
if "_corrupt_record" in df.columns:
    df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

df = df.withColumn(
    "trip_duration_minutes",
    (
        F.col("tpep_dropoff_datetime").cast("long")
        - F.col("tpep_pickup_datetime").cast("long")
    ) / 60.0
)

original_count = df.count()

# Report each rule independently. Counts overlap.
checks = df.agg(
    F.sum(
        F.when(
            F.col("tpep_pickup_datetime").isNull()
            | F.col("tpep_dropoff_datetime").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_timestamps"),

    F.sum(
        F.when(F.col("trip_duration_minutes") <= 0, 1).otherwise(0)
    ).alias("nonpositive_duration"),

    F.sum(
        F.when(F.col("trip_distance") <= 0, 1).otherwise(0)
    ).alias("nonpositive_distance"),

    F.sum(
        F.when(F.col("fare_amount") <= 0, 1).otherwise(0)
    ).alias("nonpositive_fare"),

    F.sum(
        F.when(F.col("trip_distance") > 100, 1).otherwise(0)
    ).alias("distance_over_100"),

    F.sum(
        F.when(F.col("trip_duration_minutes") > 180, 1).otherwise(0)
    ).alias("duration_over_180")
)

print("\n=== Independent Cleaning Rules ===")
checks.show(truncate=False)

# Apply all filters together.
clean = df.filter(
    F.col("tpep_pickup_datetime").isNotNull()
    & F.col("tpep_dropoff_datetime").isNotNull()
    & F.col("trip_duration_minutes").isNotNull()
    & (F.col("trip_duration_minutes") > 0)
    & F.col("trip_distance").isNotNull()
    & (F.col("trip_distance") > 0)
    & F.col("fare_amount").isNotNull()
    & (F.col("fare_amount") > 0)
    & (F.col("trip_distance") <= 100)
    & (F.col("trip_duration_minutes") <= 180)
)

clean_count = clean.count()
removed_count = original_count - clean_count

print("\n=== Cleaning Summary ===")
print(f"Input records: {original_count:,}")
print(f"Records retained: {clean_count:,}")
print(f"Records removed: {removed_count:,}")
print(f"Retention rate: {clean_count / original_count * 100:.2f}%")

print("\n=== Cleaned Data Preview ===")
clean.select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "trip_duration_minutes",
    "fare_amount",
    "total_amount"
).show(10, truncate=False)

print("\n=== Cleaned Data Summary ===")
clean.select(
    "trip_distance",
    "trip_duration_minutes",
    "fare_amount",
    "total_amount"
).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show(
    truncate=False
)

clean.write.mode("overwrite").parquet(OUTPUT_PATH)

print(f"\nCleaned Parquet saved to: {OUTPUT_PATH}")

spark.stop()
