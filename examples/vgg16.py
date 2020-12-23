import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, regularizers, models

class VGG16(models.Model):

    def __init__(self, input_shape):
        """
        :param input_shape: [32, 32, 3]
        """
        super(VGG16, self).__init__()

        weight_decay = 0.000
        self.num_classes = 10

        model = models.Sequential()

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                         input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes))
        # model.add(layers.Activation('softmax'))

        self.model = model

    def call(self, x):
        x = self.model(x)
        return x


def normalize(X_train, X_test):
  X_train = X_train / 255
  X_test = X_test / 255
  
  mean = np.mean(X_train, axis=(0,1,2,3))
  std = np.std(X_train, axis=(0,1,2,3))
  
  X_train = (X_train - mean) / (std + 1e-7)
  X_test = (X_test - mean) / (std + 1e-7)
  return X_train, X_test

def prepare_cifar(x, y):
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.int64)
  return x, y

@tf.function
def compute_loss(logits, labels):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def main():
  tf.random.set_seed(22)
  
  print("loading data...")
  (x_train,y_train), (x_test, y_test) = datasets.cifar10.load_data()
  x_train, x_test = normalize(x_train, x_test)
  train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

  test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
  
  print("loading complete!")
  
  model = VGG16([32,32,3])
  
  optimizer = optimizers.Adam(learning_rate=0.0001)
  
  for epoch in range(10):
    for step, (x,y) in enumerate(train_loader):
      y = tf.squeeze(y, axis=1)
      
      with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)
        
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 15) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
    print("Epoch: ", epoch, "Loss: ", loss.numpy())
    
    
if __name__=="__main__":
  main()
        
