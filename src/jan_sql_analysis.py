

from pyspark.sql import SparkSession

INPUT_PATH = "data/processed/january_2025/taxi_trips_features.parquet"

spark = (
    SparkSession.builder
    .appName("NYCTaxiSparkSQLAnalysis")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Load the feature-engineered Parquet dataset.
df = spark.read.parquet(INPUT_PATH)

# Register the dataset for Spark SQL queries.
df.createOrReplaceTempView("taxi_trips")

print("\n=== Dataset Overview ===")
print(f"Total records: {df.count():,}")
print(f"Total columns: {len(df.columns)}")

df.printSchema()


def run_query(title, query):
    print(f"\n=== {title} ===")
    spark.sql(query).show(30, truncate=False)


# 1. Hourly demand and average fare
run_query(
    "Hourly Demand and Average Fare",
    """
    SELECT
        pickup_hour,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS average_fare,
        ROUND(AVG(trip_distance), 2) AS average_distance
    FROM taxi_trips
    GROUP BY pickup_hour
    ORDER BY pickup_hour
    """
)

# 2. Weekday demand and average fare
run_query(
    "Weekday Demand and Average Fare",
    """
    SELECT
        pickup_day_of_week,
        CASE pickup_day_of_week
            WHEN 1 THEN 'Sunday'
            WHEN 2 THEN 'Monday'
            WHEN 3 THEN 'Tuesday'
            WHEN 4 THEN 'Wednesday'
            WHEN 5 THEN 'Thursday'
            WHEN 6 THEN 'Friday'
            WHEN 7 THEN 'Saturday'
        END AS day_name,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS average_fare
    FROM taxi_trips
    GROUP BY pickup_day_of_week
    ORDER BY pickup_day_of_week
    """
)

# 3. Fare analysis by distance band
run_query(
    "Fare by Trip Distance",
    """
    SELECT
        CASE
            WHEN trip_distance < 1 THEN 'Under 1 mile'
            WHEN trip_distance < 3 THEN '1 to under 3 miles'
            WHEN trip_distance < 5 THEN '3 to under 5 miles'
            WHEN trip_distance < 10 THEN '5 to under 10 miles'
            ELSE '10 miles or more'
        END AS distance_band,
        COUNT(*) AS trip_count,
        ROUND(AVG(trip_distance), 2) AS average_distance,
        ROUND(AVG(fare_amount), 2) AS average_fare,
        ROUND(MIN(fare_amount), 2) AS minimum_fare,
        ROUND(MAX(fare_amount), 2) AS maximum_fare
    FROM taxi_trips
    GROUP BY
        CASE
            WHEN trip_distance < 1 THEN 'Under 1 mile'
            WHEN trip_distance < 3 THEN '1 to under 3 miles'
            WHEN trip_distance < 5 THEN '3 to under 5 miles'
            WHEN trip_distance < 10 THEN '5 to under 10 miles'
            ELSE '10 miles or more'
        END
    ORDER BY MIN(trip_distance)
    """
)

# 4. Top pickup locations
run_query(
    "Top 10 Pickup Locations",
    """
    SELECT
        pulocationid,
        COUNT(*) AS pickup_count,
        ROUND(AVG(fare_amount), 2) AS average_fare
    FROM taxi_trips
    WHERE pulocationid IS NOT NULL
    GROUP BY pulocationid
    ORDER BY pickup_count DESC
    LIMIT 10
    """
)

# 5. Top drop-off locations
run_query(
    "Top 10 Drop-off Locations",
    """
    SELECT
        dolocationid,
        COUNT(*) AS dropoff_count,
        ROUND(AVG(fare_amount), 2) AS average_fare
    FROM taxi_trips
    WHERE dolocationid IS NOT NULL
    GROUP BY dolocationid
    ORDER BY dropoff_count DESC
    LIMIT 10
    """
)

# 6. Passenger-count analysis
run_query(
    "Trips by Passenger Count",
    """
    SELECT
        passenger_count,
        COUNT(*) AS trip_count,
        ROUND(AVG(fare_amount), 2) AS average_fare,
        ROUND(AVG(trip_distance), 2) AS average_distance
    FROM taxi_trips
    GROUP BY passenger_count
    ORDER BY passenger_count
    """
)

# 7. Overall dataset statistics
run_query(
    "Overall Trip Statistics",
    """
    SELECT
        COUNT(*) AS total_trips,
        ROUND(AVG(fare_amount), 2) AS average_fare,
        ROUND(AVG(trip_distance), 2) AS average_distance,
        ROUND(AVG(trip_duration_minutes), 2) AS average_duration_minutes,
        ROUND(MIN(fare_amount), 2) AS minimum_fare,
        ROUND(MAX(fare_amount), 2) AS maximum_fare
    FROM taxi_trips
    """
)

spark.stop()
