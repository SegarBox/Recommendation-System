import os
p = os.path.abspath('./Dataset')
base = os.path.abspath('.')
print(p)

total_data = 40
train_data = int(total_data * 0.8)
test_data = int(total_data - train_data)

import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

#Loading the dataset

url = p+'/ratings_Beauty_2.csv'
train = pd.read_csv(url)
train = train.dropna()
train.head()

train.dtypes

train.UserID = train.UserId.astype(str)
train.Rating = train.Rating.astype(int)
train.head(2)

train.dtypes

ratings = tf.data.Dataset.from_tensor_slices({
    "user_id": train.UserID.tolist(),
    "Product_id": train.ProductId.tolist(),
    "rating": train.Rating.tolist(),
    "timestamp": train.Timestamp.tolist()
})

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

list(ratings.take(2).as_numpy_iterator())

ratings = ratings.map(lambda x: {
    "Product_id": x["Product_id"],
    "user_id": x["user_id"],
})
list(ratings.take(2).as_numpy_iterator())

products = tf.data.Dataset.from_tensor_slices({
    "Product_id": train.ProductId.tolist()
})

list(products.take(2).as_numpy_iterator())

products = products.map(lambda x: x["Product_id"])
list(products.take(2).as_numpy_iterator())

products = products.unique()
list(products.take(2).as_numpy_iterator())

tf.random.set_seed(42)
shuffled = ratings.shuffle(total_data, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(train_data)
test = shuffled.skip(train_data).take(test_data)

user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
product_ids = products.batch(1_000)

unique_user_ids = np.unique(np.concatenate(list(user_ids)))
unique_product_ids = np.unique(np.concatenate(list(product_ids)))

unique_product_ids[:10]

len(unique_user_ids), len(unique_product_ids)

embedding_dimension = 32

user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
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
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_products_embeddings = self.products_model(features["Product_id"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_products_embeddings)

class NoBaseClassProductsModel(tf.keras.Model):

  def __init__(self, user_model, products_model):
    super().__init__()
    self.products_model: tf.keras.Model = products_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["user_id"])
      positive_movie_embeddings = self.products_model(features["Product_id"])
      loss = self.task(user_embeddings, positive_products_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["user_id"])
    positive_products_embeddings = self.products_model(features["Product_id"])
    loss = self.task(user_embeddings, positive_products_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

model = ProductsModel(user_model, products_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(total_data).batch(8192).cache()
cached_test = test.batch(4096).cache()

history = model.fit(cached_train, epochs=10)

model.evaluate(cached_test, return_dict=True)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((products.batch(100), products.batch(100).map(model.products_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(base, "model")

  # Save the index.
  tf.saved_model.save(index, path)

  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)

  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])

  print(f"Recommendations: {titles[0][:3]}")


print(history)
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["regularization_loss"])
plt.title("Model losses during training")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "test"], loc="upper right")
plt.show()