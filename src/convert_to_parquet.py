
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("TaxiDataConversion")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = "data/raw1_10000/nyc_taxi_10000.jsonl"
output_path = "data/processed1/nyc_taxi_10000.parquet"

# Read raw JSON without imposing a numeric schema.
df = spark.read.json(input_path)

# Convert timestamps and numeric columns explicitly.
timestamp_columns = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]

integer_columns = [
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

for column in timestamp_columns:
    df = df.withColumn(
        column,
        F.to_timestamp(F.col(column), "yyyy-MM-dd'T'HH:mm:ss.SSS")
    )

for column in integer_columns:
    df = df.withColumn(column, F.col(column).cast("int"))

for column in double_columns:
    df = df.withColumn(column, F.col(column).cast("double"))

print("Converted schema:")
df.printSchema()

print("Sample records:")
df.select(
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
).show(5, truncate=False)

print("Record count:", df.count())

# Write corrected Parquet output.
(
    df.write
    .mode("overwrite")
    .parquet(output_path)
)

# Verify the saved data.
parquet_df = spark.read.parquet(output_path)

print("Saved record count:", parquet_df.count())

parquet_df.select(
    "trip_distance",
    "fare_amount",
    "total_amount",
).summary("count", "min", "max", "mean").show()

spark.stop()
