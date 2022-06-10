import os
import pprint
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from typing import Dict, Text
from crud import select

base = os.path.abspath('..')

total_data = len(select())
train_data = int(total_data * 0.8)
test_data = int(total_data - train_data)

user_id=[]
product_id=[]
rating=[]
for x in select():
  user_id.append(str(x[0]))
  product_id.append(str(x[1]))
  rating.append(int(x[2]))

ratings = tf.data.Dataset.from_tensor_slices({
    "user_id": user_id,
    "Product_id": product_id,
    "rating": rating
})

ratings = ratings.map(lambda x: {
    "Product_id": x["Product_id"],
    "user_id": x["user_id"],
})

products = tf.data.Dataset.from_tensor_slices({
    "Product_id": product_id
})

products = products.map(lambda x: x["Product_id"])
products = products.unique()

tf.random.set_seed(42)
shuffled = ratings.shuffle(total_data, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(train_data)
test = shuffled.skip(train_data).take(test_data)

user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
product_ids = products.batch(1_000)

unique_user_ids = np.unique(np.concatenate(list(user_ids)))
unique_product_ids = np.unique(np.concatenate(list(product_ids)))

embedding_dimension = 32

user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

products_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_product_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_product_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=products.batch(128).map(products_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class ProductsModel(tfrs.Model):

  def __init__(self, user_model, products_model):
    super().__init__()
    self.products_model: tf.keras.Model = products_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])
    positive_products_embeddings = self.products_model(features["Product_id"])
    return self.task(user_embeddings, positive_products_embeddings)

model = ProductsModel(user_model, products_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(total_data).batch(8192).cache()
cached_test = test.batch(4096).cache()

history = model.fit(cached_train, epochs=10)

model.evaluate(cached_test, return_dict=True)
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
  tf.data.Dataset.zip((products.batch(100), products.batch(100).map(model.products_model)))
)

_, titles = index(tf.constant(["42"]))
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(base, "model")
  tf.saved_model.save(index, path)