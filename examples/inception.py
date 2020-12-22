import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_test, tf.float32) / 255
y_train, y_test = tf.cast(y_train, tf.int64), tf.cast(y_test, tf.int64)
x_train, x_test = tf.expand_dims(x_train, axis=3), tf.expand_dims(x_test, axis=3)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)


class ConvBNRelu(tf.Module):
  def __init__(self, ch, kernelsize=3,  strides=1, padding='same'):
    super(ConvBNRelu, self).__init__()
    self.l1 = keras.layers.Conv2D(ch, kernelsize, strides=strides, padding=padding)
    self.l2 = keras.layers.BatchNormalization()
    self.l3 = keras.layers.ReLU()
    
  def __call__(self, x):
    img = self.l3(self.l2(self.l1(x)))
    return img
  
class InceptionBlk(tf.Module):
    
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.conv1 = ConvBNRelu(ch, strides=strides)
        self.conv2 = ConvBNRelu(ch, kernelsize=3, strides=strides)
        self.conv3_1 = ConvBNRelu(ch, kernelsize=3, strides=strides)
        self.conv3_2 = ConvBNRelu(ch, kernelsize=3, strides=1)
        self.pool = keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBNRelu(ch, strides=strides)
        
        
    def __call__(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x3_1)
        x4 = self.pool(x)
        x4 = self.pool_conv(x4)
        # concat along axis=channel
        x = tf.concat([x1, x2, x3_2, x4], axis=3)        
        return x


class Inception(tf.Module):
    
    def __init__(self, num_layers, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_layers = num_layers
        self.init_ch = init_ch
        self.conv1 = ConvBNRelu(init_ch)
        self.blocks = []
        
        for block_id in range(num_layers):    
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.append(block)
            self.out_channels *= 2            
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)
        
        
    def __call__(self, x):
        out = self.conv1(x)
        for layer in self.blocks:
          out = layer(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out    
            

def compute_loss(logits, labels):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


batch_size = 32
epochs=10
model = Inception(2, 10)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(epochs):
  for step, (x,y) in enumerate(db_train):
    with tf.GradientTape() as tape:
      logits = model(x)
      loss = compute_loss(logits, y)
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      
  print("Epoch: ", epoch, "Loss", los.numpy())



