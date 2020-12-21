import os
import time
import tensorflow as tf
import numpy as np
# from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers, metrics
tf.config.run_functions_eagerly(True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

def mnist_datasets():
  (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = tf.cast(tf.one_hot(y_train, 10), tf.int64), tf.cast(tf.one_hot(y_test, 10), tf.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  return train_dataset, test_dataset

class CNN(tf.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.l1 = layers.Reshape(input_shape=(28,28,), target_shape=[28,28,1])
    self.l2 = layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu)
    self.l3 = layers.MaxPooling2D((2,2),(2,2), padding='same')
    self.l4 = layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu)
    self.l5 = layers.MaxPooling2D((2,2),(2,2), padding='same')
    self.l6 = layers.Flatten()
    self.l7 = layers.Dense(32, activation=tf.nn.relu)
    self.l8 = layers.Dropout(rate=0.4)
    self.l9 = layers.Dense(10)
    
  def __call__(self, imgs):
    x = self.l9(self.l8(self.l7(self.l6(self.l5(self.l4(self.l3(self.l2(self.l1(imgs)))))))))
    return x
  
@tf.function
def train_one_step(model, optimizer, images, labels):
  with tf.GradientTape() as tape:
    logits = model(images)
    loss = compute_loss(labels, logits)
    acc = compute_accuracy(labels, logits)
    
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss, acc

@tf.function
def compute_loss(labels, logits):
  # print(tf.shape(labels), tf.shape(logits))
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

@tf.function
def compute_accuracy(labels, logits):
  prediction = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(labels, axis=1)), tf.float32))


def train(model, dataset, optimizer, log_freq=600):
  
  for img, labels in dataset:
    loss, acc = train_one_step(model, optimizer, img, labels)
    
  return loss, acc
        

def main():
  train_ds, test_ds = mnist_datasets()
  train_ds = train_ds.shuffle(60000).batch(100)
  test_ds = test_ds.batch(100)
  
  optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.5)
  model = CNN()
  
  NUM_EPOCH = 5

  for i in range(0, NUM_EPOCH):
    start = time.time()
    loss, acc = train(model, train_ds, optimizer)
    print("Loss: ", np.float64(loss),"Accuracy: ", np.float64(acc))
    end = time.time()
    print("Training complete, Time Taken:", int(end) - int(start))

if __name__ == "__main__":
  main()
