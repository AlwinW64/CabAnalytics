

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("TaxiDataProfiling")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(
    "data/processed/processed1/nyc_taxi_10000.parquet"
)

print("Total records:", df.count())

# Check missing values
print("\nMissing values by column:")

missing_exprs = [
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]

df.select(missing_exprs).show(truncate=False)

# Create trip duration in minutes
df = df.withColumn(
    "trip_duration_minutes",
    (
        F.col("tpep_dropoff_datetime").cast("long")
        - F.col("tpep_pickup_datetime").cast("long")
    ) / 60.0
)

# Count suspicious records
checks = [
    ("Negative fare", F.col("fare_amount") < 0),
    ("Zero or negative fare", F.col("fare_amount") <= 0),
    ("Zero or negative distance", F.col("trip_distance") <= 0),
    ("Negative total amount", F.col("total_amount") < 0),
    ("Zero or negative duration",
     F.col("trip_duration_minutes") <= 0),
    ("Duration over 3 hours",
     F.col("trip_duration_minutes") > 180),
    ("Fare over $200", F.col("fare_amount") > 200),
    ("Distance over 100 miles", F.col("trip_distance") > 100),
]

print("\nData quality checks:")

for label, condition in checks:
    count = df.filter(condition).count()
    print(f"{label}: {count}")

print("\nFare distribution:")
df.select("fare_amount").summary(
    "count", "min", "25%", "50%", "75%", "max", "mean"
).show()

print("\nDistance distribution:")
df.select("trip_distance").summary(
    "count", "min", "25%", "50%", "75%", "max", "mean"
).show()

print("\nExamples of negative fares:")
df.filter(F.col("fare_amount") < 0).select(
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
).show(10, truncate=False)

spark.stop()
