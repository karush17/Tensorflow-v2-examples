import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
tf.config.run_functions_eagerly(True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

units = 100
activation = 'relu'

def mnist_dataset():
  (x,y), _ = datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x,y))
  ds = ds.map(mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds

def mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255
  y = tf.cast(y, tf.int64)
  return (x,y)


class Model(tf.Module):
  def __init__(self, input_dims):
    super(Model, self).__init__()
    self.input_dims = input_dims
    self.inputs = layers.Flatten()
    self.layer1 = layers.Dense(units, activation)
    self.layer2 = layers.Dense(units, activation)
    self.layer3 = layers.Dense(10)


  def __call__(self, imgs):
    x = self.inputs(imgs)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return x

optimizer = optimizers.Adam()
model = Model(28)

@tf.function
def compute_loss(logits, labels):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
  prediction = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

@tf.function
def train_one_step(optimizer, x,y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(logits, y)
    
    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    accuracy = compute_accuracy(logits, y)
  return loss, accuracy

def train(epoch, optimizer):
  train_ds = mnist_dataset()
  loss = 0
  accuracy = 0
  for step, (x,y) in enumerate(train_ds):
    loss, accuracy = train_one_step(optimizer, x, y)
    
    if step % 500 == 0:
      print("epoch", epoch, "loss", loss.numpy(), "accuracy", accuracy.numpy())
  return loss, accuracy
    

## train the model in main loop
for epoch in range(20):
  loss, accuracy = train(epoch, optimizer)
