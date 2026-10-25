

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("JanuaryDataInvestigation")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

PATH = "data/processed/january_2025/taxi_trips.parquet"
df = spark.read.parquet(PATH)

total = df.count()

print(f"Total rows: {total:,}")

# Check malformed JSON records
if "_corrupt_record" in df.columns:
    corrupt_count = df.filter(
        F.col("_corrupt_record").isNotNull()
    ).count()

    print(f"Rows with corrupt-record content: {corrupt_count:,}")

    df.filter(
        F.col("_corrupt_record").isNotNull()
    ).select(
        "_corrupt_record"
    ).show(10, truncate=False)

# Check missing values in important columns
important_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "pulocationid",
    "dolocationid",
]

print("\n=== Missing Values ===")

df.select([
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in important_columns
]).show(truncate=False)

# Inspect extreme values without deleting anything
print("\n=== Highest Trip Distances ===")

df.orderBy(
    F.desc("trip_distance")
).select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "pulocationid",
    "dolocationid",
).show(10, truncate=False)

print("\n=== Highest Fare Amounts ===")

df.orderBy(
    F.desc("fare_amount")
).select(
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
).show(10, truncate=False)

print("\n=== Extreme-Value Counts ===")

df.select(
    F.sum(
        F.when(F.col("trip_distance") > 100, 1).otherwise(0)
    ).alias("distance_over_100_miles"),
    F.sum(
        F.when(F.col("fare_amount") > 200, 1).otherwise(0)
    ).alias("fare_over_200"),
    F.sum(
        F.when(F.col("trip_distance") < 0, 1).otherwise(0)
    ).alias("negative_distance"),
    F.sum(
        F.when(F.col("fare_amount") <= 0, 1).otherwise(0)
    ).alias("nonpositive_fare"),
).show(truncate=False)

spark.stop()
