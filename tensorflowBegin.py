import numpy as np
import tensorflow as tf

# Create a placeholder-like variable
input_data = tf.constant([1, 2, 3], dtype=tf.float32)

# Now you can use input_data in your computations
result = tf.square(input_data)

# To get the result as a NumPy array, you can call numpy() on the tensor
result_array = result.numpy()

print(result_array)