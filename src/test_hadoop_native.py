
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("HadoopNativeTest")
    .getOrCreate()
)

jvm = spark.sparkContext._jvm

try:
    result = jvm.org.apache.hadoop.io.nativeio.NativeIO.POSIX.stat(
        r"C:\Windows\win.ini"
    )
    print("NativeIO stat succeeded:", result)
except Exception as e:
    print("NativeIO stat failed:")
    print(type(e).__name__, e)
finally:
    spark.stop()