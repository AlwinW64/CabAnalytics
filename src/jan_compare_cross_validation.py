from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import time
PAUSE_SECONDS = 30

# Run from the project root:
# spark-submit data/jan_compare_cross_validation.py
INPUT_PATH = "data/processed/january_2025/taxi_trips_features.parquet"

spark = (
    SparkSession.builder
    .appName("CabAnalyticsThreeFoldCrossValidation")
    .master("local[4]")
    .config("spark.driver.memory", "6g")
    .config("spark.driver.maxResultSize", "1g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = None
folds = []

try:
    df = spark.read.parquet(INPUT_PATH)

    # Same target, predictors, and filters as the existing training scripts.
    df = df.select(
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pulocationid",
        "dolocationid",
        "fare_amount",
    ).filter(
        df["fare_amount"].isNotNull()
        & df["trip_distance"].isNotNull()
        & (df["trip_distance"] > 0)
        & df["pickup_hour"].isNotNull()
        & df["pickup_day_of_week"].isNotNull()
    ).persist(StorageLevel.MEMORY_AND_DISK)

    total = df.count()
    print(f"Records available for cross-validation: {total:,}")

    # Create mutually exclusive folds once and reuse them for both models.
    folds = [
        part.persist(StorageLevel.MEMORY_AND_DISK)
        for part in df.randomSplit([1.0, 1.0, 1.0], seed=42)
    ]
    fold_counts = [fold.count() for fold in folds]
    for i, count in enumerate(fold_counts, start=1):
        print(f"Fold {i}: {count:,} records")

    evaluators = {
        metric: RegressionEvaluator(
            labelCol="fare_amount",
            predictionCol="prediction",
            metricName=metric,
        )
        for metric in ("rmse", "mae", "r2")
    }

    def build_pipeline(model_type):
        categorical_cols = ["pulocationid", "dolocationid"]
        index_cols = ["pickup_location_index", "dropoff_location_index"]
        encoded_cols = ["pickup_location_encoded", "dropoff_location_encoded"]

        indexer = StringIndexer(
            inputCols=categorical_cols,
            outputCols=index_cols,
            handleInvalid="keep",
        )
        encoder = OneHotEncoder(
            inputCols=index_cols,
            outputCols=encoded_cols,
            handleInvalid="keep",
        )
        assembler = VectorAssembler(
            inputCols=[
                "trip_distance",
                "pickup_hour",
                "pickup_day_of_week",
            ] + encoded_cols,
            outputCol="features",
            handleInvalid="keep",
        )

        if model_type == "Linear Regression":
            estimator = LinearRegression(
                featuresCol="features",
                labelCol="fare_amount",
                predictionCol="prediction",
                maxIter=50,
                regParam=0.01,
                elasticNetParam=0.0,
            )
        else:
            estimator = RandomForestRegressor(
                featuresCol="features",
                labelCol="fare_amount",
                predictionCol="prediction",
                numTrees=40,
                maxDepth=8,
                maxBins=64,
                seed=42,
            )

        return Pipeline(stages=[indexer, encoder, assembler, estimator])

    results = []

    for model_type in ("Linear Regression", "Random Forest"):
        print(f"\n{'=' * 18} {model_type} {'=' * 18}")

        for test_idx in range(3):
            test_df = folds[test_idx]
            train_df = folds[(test_idx + 1) % 3].unionByName(
                folds[(test_idx + 2) % 3]
            )

            train_count = fold_counts[(test_idx + 1) % 3] + fold_counts[(test_idx + 2) % 3]
            test_count = fold_counts[test_idx]
            print(
                f"\nTraining {model_type}, fold {test_idx + 1}/3 "
                f"(train={train_count:,}, test={test_count:,})"
            )

            pipeline = build_pipeline(model_type)
            model = pipeline.fit(train_df)
            predictions = model.transform(test_df).persist(
                StorageLevel.MEMORY_AND_DISK
            )

            row = {
                "model": model_type,
                "fold": test_idx + 1,
                "train_records": train_count,
                "test_records": test_count,
            }
            for metric, evaluator in evaluators.items():
                row[metric] = evaluator.evaluate(predictions)

            results.append(row)
            print(
                f"Fold {test_idx + 1}: "
                f"RMSE={row['rmse']:.4f}, "
                f"MAE={row['mae']:.4f}, "
                f"R2={row['r2']:.4f}"
            )

            predictions.unpersist()
            del model, pipeline, predictions, train_df, test_df

            print(f"Waiting {PAUSE_SECONDS} seconds before the next fit...")
            time.sleep(PAUSE_SECONDS)

    print("\n========== CROSS-VALIDATION RESULTS ==========")
    print("Model               Fold   Train rows   Test rows      RMSE      MAE       R2")
    for row in results:
        print(
            f"{row['model']:<19} {row['fold']:<6} "
            f"{row['train_records']:<12,} {row['test_records']:<11,} "
            f"{row['rmse']:>8.4f} {row['mae']:>8.4f} {row['r2']:>8.4f}"
        )

    print("\n========== MEAN AND SAMPLE STANDARD DEVIATION ==========")
    for model_type in ("Linear Regression", "Random Forest"):
        model_rows = [r for r in results if r["model"] == model_type]
        print(f"\n{model_type}\n")
        for metric in ("rmse", "mae", "r2"):
            values = [r[metric] for r in model_rows]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            stddev = variance ** 0.5
            print(f"  {metric.upper()}: mean={mean:.4f}, sample SD={stddev:.4f}")

    print(
        "\nInterpretation: lower RMSE/MAE is better; higher R2 is better. "
        "These are cross-validation estimates, not an untouched final-test score."
    )
finally:
    for fold in folds:
        fold.unpersist()
    if df is not None:
        df.unpersist()
    spark.stop()
