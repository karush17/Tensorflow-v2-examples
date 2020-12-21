import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)

class Regressor(tf.Module):
  def __init__(self):
    super(Regressor, self).__init__()
    self.w = tf.Variable(tf.random.normal((13, 1)), name='weight', shape=[13,1])
    self.b = tf.Variable(tf.random.normal((1,)), name='bias', shape=[1])
          
  def __call__(self, features):
    x = tf.matmul(features, self.w) + self.b
    return x
  
@tf.function
def compute_loss(logits, labels):
  return tf.reduce_mean(tf.nn.l2_loss(logits - labels))


def main():
  tf.random.set_seed(22)
  (x_train, y_train), (x_val, y_val) = datasets.boston_housing.load_data()
  x_train, x_val = tf.cast(x_train, tf.float32), tf.cast(x_val, tf.float32)
  y_train, y_val = tf.cast(y_train, tf.float32), tf.cast(y_val, tf.float32)
  db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
  db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
  
  model = Regressor()
  optimizer = optimizers.Adam(lr=1e-2)
  
  for epoch in range(200):
    for step, (x,y) in enumerate(db_train):
      with tf.GradientTape() as tape:
        logits = model(x)
        logits = tf.squeeze(logits, axis=1)
        loss = compute_loss(logits, y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("epoch: ", int(epoch), "loss: ", loss.numpy())
    
    
if __name__=="__main__":
  main()



