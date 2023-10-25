# I. Cài đặt thư viện
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# II. Tải dữ liệu
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))
# load data to memory
train_df = pd.read_csv(train_dataset_fp)

# Tạo tf dataset từ csv
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

# III. Xây dựng tensorflow dataset từ csv
batch_size = 32
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))
print(features)
# Một cách khác
# list(train_dataset.take(1))
# Hiển thị dữ liệu
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels,
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
# Hiện tại trong dataset các cột đang tách rời nhau, ta sẽ sử dụng hàm stack để ghép các cột

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
print(features)

# Xây dựng mô hình bao gồm
# 2 tầng ẩn mỗi tầng có 10 nơ ron + activation là relu
# Lớp Softmax phân loại thành 3 nhãn
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2) ,
  tf.keras.layers.Dense(3)
])
# Định nghĩa hàm mất mát
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))
# Định nghĩa thuật toán tối ưu
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# Định nghĩa quá trình tính Loss Function + Gradie
model.trainable_variables
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# Thử nghiệp quá trình này
loss_value, grads = grad(model, features, labels)
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(), loss(model, features, labels, training=True).numpy()))

# Tiến hành training
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # lặp qua các batch và đổ vào mô hình
  for x, y in train_dataset:
    # Tính gradient
    loss_value, grads = grad(model, x, y)

    # Cập nhật tham số của mô hình thông qua Gradient
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Tính giá trị sai lệch
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    
    # Tính độ chính xác
    epoch_accuracy.update_state(y, model(x, training=True))
  
  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))