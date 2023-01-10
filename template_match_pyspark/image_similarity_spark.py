import os
import pyspark

conf = pyspark.SparkConf()
conf.set('spark.ui.proxyBase', '/user/' + os.environ['JUPYTERHUB_USER'] + '/proxy/4041')
conf.set('spark.sql.repl.eagerEval.enabled', True)
conf.set('spark.driver.memory','4g')

sc = pyspark.SparkContext(conf=conf)
spark = pyspark.SQLContext.getOrCreate(sc)

# Import dependent packages
import numpy as np
from PIL import Image
from pyspark.sql.functions import col
from pyspark.sql.functions import split, udf, pandas_udf
from pyspark.sql.types import FloatType, IntegerType
from collections import Counter
from pyspark.sql.functions import lit

def read_image_as_array(img_name, root_dir="data"):
    path = os.path.join(root_dir, img_name)
    img = Image.open(path)
    img_arr = np.asarray(img)
    return img_arr

# Load the data corpus

df1 = spark.read\
    .option('delimiter',',')\
    .option('header', True)\
    .option("inferschema", True)\
    .csv("data/filelist.csv")

img1 = "1.jpg"
# df1 = df1.withColumn('reference_img', lit(img1))
img1 = read_image_as_array(img1)

df1

@udf(returnType=FloatType())
def calculate_image_similarity(img2):
    img2 = read_image_as_array(img2)
    # Image 1
    flat_array_1 = img1.flatten()
    RH1 = Counter(flat_array_1)
    H1 = []
    for i in range(256):
        H1.append(RH1[i] if i in RH1.keys() else 0)

    # Image 2
    flat_array_2 = img2.flatten()
    RH1 = Counter(flat_array_2)
    H2 = []
    for i in range(256):
        H2.append(RH1[i] if i in RH1.keys() else 0)
    
    distance=0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    dist_test_ref_1 = np.sqrt(distance)
    return float(dist_test_ref_1)

df1 = df1.withColumn("similarity_index", calculate_image_similarity(col("filename")))
df1.sort(col("similarity_index")).show()