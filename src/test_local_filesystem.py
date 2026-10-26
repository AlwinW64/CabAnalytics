

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("LocalFilesystemTest")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

jvm = sc._jvm
conf = sc._jsc.hadoopConfiguration()

try:
    fs = jvm.org.apache.hadoop.fs.FileSystem.getLocal(conf)
    path = jvm.org.apache.hadoop.fs.Path(
        "models/january_2025_random_forest/metadata/part-00000"
    )

    print("Filesystem:", fs.getClass().getName())
    print("File exists:", fs.exists(path))
    print("File status:", fs.getFileStatus(path))

except Exception as e:
    print("Filesystem test failed:")
    print(e)

finally:
    spark.stop()
