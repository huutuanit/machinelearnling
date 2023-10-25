import numpy as np
import tensorflow as tf

# Create a dataset from a tensor
data = tf.constant([1, 2, 3, 4, 5])
dataset = tf.data.Dataset.from_tensor_slices(data)

# Create an iterator for the dataset
iterator = iter(dataset)
# Now, you can iterate through the dataset
for item in iterator:
    print(item.numpy())