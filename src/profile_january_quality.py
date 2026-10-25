

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("JanuaryTaxiQualityProfile")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

PATH = "data/processed/january_2025/taxi_trips.parquet"
df = spark.read.parquet(PATH)

# Remove metadata/corrupt rows from the profiling population.
if "_corrupt_record" in df.columns:
    trips = df.filter(F.col("_corrupt_record").isNull())
else:
    trips = df

# Calculate trip duration.
trips = trips.withColumn(
    "trip_duration_minutes",
    (
        F.col("tpep_dropoff_datetime").cast("long")
        - F.col("tpep_pickup_datetime").cast("long")
    ) / 60.0
)

print("=== Population ===")
print("Rows in Parquet:", df.count())
print("Rows after corrupt-record removal:", trips.count())

# Count each rule independently. These counts overlap.
print("\n=== Independent Quality Checks ===")

checks = trips.select(
    F.sum(
        F.when(
            F.col("tpep_pickup_datetime").isNull()
            | F.col("tpep_dropoff_datetime").isNull(),
            1
        ).otherwise(0)
    ).alias("missing_timestamps"),

    F.sum(
        F.when(
            F.col("tpep_dropoff_datetime")
            <= F.col("tpep_pickup_datetime"),
            1
        ).otherwise(0)
    ).alias("nonpositive_duration"),

    F.sum(
        F.when(F.col("trip_distance") <= 0, 1).otherwise(0)
    ).alias("nonpositive_distance"),

    F.sum(
        F.when(F.col("fare_amount") < 0, 1).otherwise(0)
    ).alias("negative_fare"),

    F.sum(
        F.when(F.col("fare_amount") == 0, 1).otherwise(0)
    ).alias("zero_fare"),

    F.sum(
        F.when(F.col("fare_amount") > 200, 1).otherwise(0)
    ).alias("fare_over_200"),

    F.sum(
        F.when(F.col("trip_distance") > 100, 1).otherwise(0)
    ).alias("distance_over_100"),

    F.sum(
        F.when(F.col("trip_duration_minutes") > 180, 1)
         .otherwise(0)
    ).alias("duration_over_180_minutes"),

    F.sum(
        F.when(
            F.col("trip_duration_minutes") > 720, 1
        ).otherwise(0)
    ).alias("duration_over_12_hours")
)

checks.show(truncate=False)

print("\n=== Duration Distribution ===")
trips.select(
    "trip_duration_minutes"
).filter(
    F.col("trip_duration_minutes") > 0
).summary(
    "count", "25%", "50%", "75%", "90%", "95%", "99%"
).show(truncate=False)

print("\n=== Most Extreme Distances ===")
trips.orderBy(
    F.desc("trip_distance")
).select(
    "trip_distance",
    "trip_duration_minutes",
    "fare_amount",
    "total_amount"
).show(15, truncate=False)

spark.stop()
