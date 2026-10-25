

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

INPUT_PATH = "data/processed/january_2025/taxi_trips_features.parquet"
MODEL_PATH = "models/january_2025_linear_regression"

spark = (
    SparkSession.builder
    .appName("NYCTaxiFarePrediction")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Load the corrected feature dataset.
df = spark.read.parquet(INPUT_PATH)

# Use only fields appropriate for the initial fare-estimation scenario.
# Actual trip duration and post-trip monetary fields are excluded.
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
)

print(f"Records available for modeling: {df.count():,}")

# Deterministic random split.
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training records: {train_df.count():,}")
print(f"Test records: {test_df.count():,}")

# Treat location IDs as categories, not continuous numbers.
categorical_cols = ["pulocationid", "dolocationid"]

index_cols = [f"{c}_index" for c in categorical_cols]
encoded_cols = [f"{c}_encoded" for c in categorical_cols]

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

numeric_cols = [
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week"
]

assembler = VectorAssembler(
    inputCols=numeric_cols + encoded_cols,
    outputCol="features",
    handleInvalid="keep"
)

regression = LinearRegression(
    featuresCol="features",
    labelCol="fare_amount",
    predictionCol="prediction",
    maxIter=50,
    regParam=0.1,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=[
    indexer,
    encoder,
    assembler,
    regression
])

print("\nTraining Linear Regression model...")
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

# Regression evaluation metrics.
for metric in ["rmse", "mae", "r2"]:
    evaluator = RegressionEvaluator(
        labelCol="fare_amount",
        predictionCol="prediction",
        metricName=metric
    )
    score = evaluator.evaluate(predictions)
    print(f"Test {metric.upper()}: {score:.4f}")

# Save the fitted preprocessing and regression pipeline.
model.write().overwrite().save(MODEL_PATH)
print(f"\nSaved model pipeline to: {MODEL_PATH}")

spark.stop()
