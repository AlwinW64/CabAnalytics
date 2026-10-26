
from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

INPUT_PATH = "data/processed/january_2025/taxi_trips_features.parquet"
MODEL_PATH = "models/january_2025_random_forest"

spark = (
    SparkSession.builder
    .appName("NYCTaxiRandomForest")
    .master("local[4]")
    .config("spark.driver.memory", "6g")
    .config("spark.driver.maxResultSize", "1g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(INPUT_PATH)

# Use the same columns and filters as the Linear Regression model.
df = df.select(
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pulocationid",
    "dolocationid",
    "fare_amount"
).filter(
    df["fare_amount"].isNotNull()
    & df["trip_distance"].isNotNull()
    & (df["trip_distance"] > 0)
    & df["pickup_hour"].isNotNull()
    & df["pickup_day_of_week"].isNotNull()
).persist(StorageLevel.MEMORY_AND_DISK)

total = df.count()
print(f"Total records available: {total:,}")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {train_df.count():,}")
print(f"Test records: {test_df.count():,}")

# Encode location IDs as categorical features.
categorical_cols = ["pulocationid", "dolocationid"]
index_cols = ["pickup_location_index", "dropoff_location_index"]
encoded_cols = ["pickup_location_encoded", "dropoff_location_encoded"]

indexer = StringIndexer(
    inputCols=categorical_cols,
    outputCols=index_cols,
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=index_cols,
    outputCols=encoded_cols,
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=[
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week"
    ] + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="fare_amount",
    predictionCol="prediction",
    numTrees=40,
    maxDepth=8,
    maxBins=64,
    seed=42
)

pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])

print("\nTraining Random Forest model...")
model = pipeline.fit(train_df)

print("\nGenerating test predictions...")
predictions = model.transform(test_df)

print("\n=== Sample Predictions ===")
predictions.select(
    "fare_amount",
    "prediction",
    "trip_distance",
    "pickup_hour"
).show(15, truncate=False)

print("\n=== Random Forest Evaluation ===")

for metric in ["rmse", "mae", "r2"]:
    evaluator = RegressionEvaluator(
        labelCol="fare_amount",
        predictionCol="prediction",
        metricName=metric
    )
    score = evaluator.evaluate(predictions)
    print(f"Test {metric.upper()}: {score:.4f}")

model.write().overwrite().save(MODEL_PATH)
print(f"\nSaved model to: {MODEL_PATH}")

spark.stop()
