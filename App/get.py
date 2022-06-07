import os
import tensorflow as tf
p = os.path.abspath('.')
print(p)

path = os.path.join(p, "model")

# Save the index.
# tf.saved_model.save(
#   index,
#   path,
#   options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
# )

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

# Pass a user id in, get top predicted movie titles back.
scores, titles = loaded(["s"])

print(f"Recommendations: {titles[0][:3]}")
# print(scores)