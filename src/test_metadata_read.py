

from pyspark.sql import SparkSession
from pathlib import Path

spark = (
    SparkSession.builder
    .appName("MetadataReadTest")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

metadata_path = str(
    Path(
        "models/january_2025_random_forest/metadata/part-00000"
    ).resolve()
)

try:
    print("Metadata path:", metadata_path)

    lines = sc.textFile(
        metadata_path,
        minPartitions=1
    ).first()

    print("SUCCESS: Metadata read through Spark RDD")
    print(lines)

except Exception:
    import traceback
    traceback.print_exc()

finally:
    spark.stop()
