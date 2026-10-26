

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

model_path = str(
    Path("models/january_2025_random_forest").resolve()
)

spark = (
    SparkSession.builder
    .appName("ActualModelLoadTest")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("Loading:", model_path)

    model = PipelineModel.load(model_path)

    print("SUCCESS: Pipeline loaded!")
    print("Number of stages:", len(model.stages))

    for i, stage in enumerate(model.stages):
        print(f"Stage {i}: {stage.__class__.__name__}")

finally:
    spark.stop()
