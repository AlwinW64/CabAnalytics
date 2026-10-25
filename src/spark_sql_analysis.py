

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("TaxiSparkSQLAnalysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = "data/processed/processed1/nyc_taxi_features.parquet"

df = spark.read.parquet(INPUT_PATH)

# Register the DataFrame as a temporary SQL view
df.createOrReplaceTempView("taxi_trips")


# 1. Trip demand by pickup hour
print("\n=== 1. Trip Count by Pickup Hour ===")

spark.sql("""
    SELECT
        pickup_hour,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS avg_fare
    FROM taxi_trips
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""").show(24, truncate=False)


# 2. Average fare by trip-distance band
print("\n=== 2. Average Fare by Distance Band ===")

spark.sql("""
    SELECT
        CASE
            WHEN trip_distance < 1 THEN 'Under 1 mile'
            WHEN trip_distance < 3 THEN '1-3 miles'
            WHEN trip_distance < 5 THEN '3-5 miles'
            WHEN trip_distance < 10 THEN '5-10 miles'
            ELSE '10+ miles'
        END AS distance_band,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS avg_fare,
        ROUND(AVG(total_amount), 2) AS avg_total_amount
    FROM taxi_trips
    GROUP BY
        CASE
            WHEN trip_distance < 1 THEN 'Under 1 mile'
            WHEN trip_distance < 3 THEN '1-3 miles'
            WHEN trip_distance < 5 THEN '3-5 miles'
            WHEN trip_distance < 10 THEN '5-10 miles'
            ELSE '10+ miles'
        END
    ORDER BY MIN(trip_distance)
""").show(truncate=False)


# 3. Most common pickup locations
print("\n=== 3. Top 10 Pickup Locations ===")

spark.sql("""
    SELECT
        pulocationid,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS avg_fare
    FROM taxi_trips
    GROUP BY pulocationid
    ORDER BY trip_count DESC
    LIMIT 10
""").show(truncate=False)


# 4. Weekday versus weekend
print("\n=== 4. Weekday vs Weekend ===")

spark.sql("""
    SELECT
        CASE
            WHEN is_weekend = 1 THEN 'Weekend'
            ELSE 'Weekday'
        END AS day_type,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS avg_fare,
        ROUND(AVG(trip_distance), 2) AS avg_distance,
        ROUND(AVG(trip_duration_minutes), 2) AS avg_duration_minutes
    FROM taxi_trips
    GROUP BY is_weekend
    ORDER BY is_weekend
""").show(truncate=False)

spark.stop()
