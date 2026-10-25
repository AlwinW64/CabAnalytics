

from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("JanuaryTaxiParquetConversion")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT_DIR = "data/raw/raw2_january_2025/*.jsonl"
OUTPUT_DIR = "data/processed/january_2025/taxi_trips.parquet"

start = perf_counter()

# Read all 31 JSONL files together.
df = spark.read.json(INPUT_DIR)

print("Detected input schema:")
df.printSchema()

# Parse timestamps and cast numeric fields.
df = (
    df
    .withColumn(
        "tpep_pickup_datetime",
        F.to_timestamp("tpep_pickup_datetime")
    )
    .withColumn(
        "tpep_dropoff_datetime",
        F.to_timestamp("tpep_dropoff_datetime")
    )
)

integer_columns = [
    "vendorid",
    "pulocationid",
    "dolocationid",
    "payment_type",
]

double_columns = [
    "passenger_count",
    "trip_distance",
    "ratecodeid",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

for column in integer_columns:
    if column in df.columns:
        df = df.withColumn(
            column, F.col(column).cast("int")
        )

for column in double_columns:
    if column in df.columns:
        df = df.withColumn(
            column, F.col(column).cast("double")
        )

# Persist the typed data without applying cleaning rules yet.
df.write.mode("overwrite").parquet(OUTPUT_DIR)

# Read the saved output to verify it.
parquet_df = spark.read.parquet(OUTPUT_DIR)

print("\n=== Conversion Verification ===")
print("Record count:", parquet_df.count())
parquet_df.printSchema()

print("\n=== Pickup Date Range ===")
parquet_df.select(
    F.min("tpep_pickup_datetime").alias("first_pickup"),
    F.max("tpep_pickup_datetime").alias("last_pickup"),
).show(truncate=False)

print("\n=== Key Column Summary ===")
parquet_df.select(
    "trip_distance",
    "fare_amount",
    "total_amount",
).summary().show()

elapsed = perf_counter() - start
print(f"Elapsed time: {elapsed:.1f} seconds")

spark.stop()
