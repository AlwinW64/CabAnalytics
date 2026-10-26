
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

MODEL_PATH = Path("models/january_2025_random_forest").resolve()

print(f"Model path: {MODEL_PATH}")
print(f"Model directory exists: {MODEL_PATH.is_dir()}")
print(f"Metadata directory exists: {(MODEL_PATH / 'metadata').is_dir()}")

if not (MODEL_PATH / "metadata").is_dir():
    raise FileNotFoundError(
        f"Model metadata not found at {MODEL_PATH / 'metadata'}. "
        "Check the path used by the training script."
    )

spark = (
    SparkSession.builder
    .appName("ModelLoadDiagnosis")
    .master("local[4]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    model = PipelineModel.load(str(MODEL_PATH))
    print("SUCCESS: Random Forest pipeline loaded.")
    print(f"Pipeline stages: {len(model.stages)}")
finally:
    spark.stop()
