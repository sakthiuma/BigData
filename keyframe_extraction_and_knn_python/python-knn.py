#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[3]:


import numpy as np
from PIL import Image
from pyspark.sql.functions import col
from pyspark.sql.functions import split, udf, pandas_udf
from pyspark.sql.types import FloatType, IntegerType,BooleanType
from collections import Counter
from pyspark.sql.functions import lit
from sklearn.neighbors import NearestNeighbors


# In[ ]:


def read_image_as_array(img_name, root_dir="data"):
    path = os.path.join(root_dir, img_name)
    img = Image.open(path)
    img_arr = np.asarray(img)
    return img_arr


# In[ ]:


img_dir = "C:/Users/Hp/Downloads/archive-dataset/images"

df = spark.read\
    .option('delimiter',',')\
    .option('header', True)\
    .option("inferschema", True)\
    .csv("C:/Users/Hp/Downloads/archive-dataset/images-list.csv")
df.show(5)


# In[5]:


img_dir = "C:/Users/Hp/Downloads/archive-dataset/images"
df = spark.read.format("image").option("dropInvalid", True).load(img_dir)
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
df_withfeatures = featurizer.transform(df).cache()
df_withfeatures


# In[ ]:


import os
 
dir_list = os.listdir(img_dir)
with open(r'C:/Users/Hp/Downloads/archive-dataset/images-list.csv', 'w') as fp:
    for item in dir_list:
        fp.write("%s\n" % item)
    print('Done')


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
import matplotlib.cm as cm
import re
import os.path

from io import StringIO
from collections import OrderedDict
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import Row
import pyspark.sql.functions as F
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler


# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import functools
import math


# In[ ]:


ds = tfds.load("imagenette")
def extract_image(example):
    image = example['image']
    return image

def preprocess_image(image, height, width):
    image = tf.image.resize_with_crop_or_pad(image, target_height=height, target_width=width)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def get_image_batches(batch_size=128, height=256, width=256):
    partial_preprocess_image = functools.partial(preprocess_image, height=height, width=width)
    train_ds = ds['train']
    train_ds = ( train_ds.map(extract_image)
                .map(partial_preprocess_image)
                .cache()
                .shuffle(buffer_size=1000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
                )
    return train_ds


BATCH_SIZE = 32
IMG_WIDTH = IMG_HEIGHT = 256
train_ds = get_image_batches(batch_size=BATCH_SIZE, height=IMG_HEIGHT, width=IMG_WIDTH)


# In[ ]:


print(train_ds,type(train_ds))
crashdata = tfds.as_numpy(train_ds)
crashdata


# In[ ]:


images = np.array([img for batch in train_ds.take(20) for img in batch])
print(images.shape)


# In[ ]:


vectorizer = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5", trainable=False)
])
vectorizer.build([None, IMG_HEIGHT, IMG_WIDTH, 3])


# In[ ]:


features = vectorizer.predict(images, batch_size=BATCH_SIZE)
print(features.shape)


# In[ ]:


from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(features)


# In[ ]:


from PIL import Image
from numpy import asarray

img = preprocess_image(Image.open('nyc.jpg'),IMG_WIDTH,IMG_HEIGHT)
numpydata = asarray(img)
image = numpydata
image = np.expand_dims(image, 0)
feature = vectorizer.predict(image)

distances, nbors = knn.kneighbors(feature)
distances, nbors = distances[0], nbors[0]

nbor_images = [images[i] for i in nbors]
fig, axes = plt.subplots(1, len(nbors)+1, figsize=(10, 5))

for i in range(len(nbor_images)+1):
    ax = axes[i]
    ax.set_axis_off()
    if i == 0:
        ax.imshow(image.squeeze(0))
        ax.set_title("Input")
    else:
        ax.imshow(nbor_images[i-1])
        ax.set_title(f"Sim: {1 - distances[i-1]:.2f}")

