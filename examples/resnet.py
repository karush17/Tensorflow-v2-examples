import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
tf.config.run_functions_eagerly(True)

tf.random.set_seed(22)
os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"


def load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_test, tf.float32) / 255
    x_train, x_test = tf.expand_dims(x_train, axis=3), tf.expand_dims(x_test, axis=3)
    y_train = tf.cast(y_train, tf.int64)
    y_test = tf.cast(y_test, tf.int64)
    # y_train = tf.one_hot(y_train, depth=10).numpy()
    # y_test = tf.one_hot(y_test, depth=10).numpy()
    return x_train, y_train, x_test, y_test

def conv3x3(channels, stride=1, kernel=(3,3)):
    return keras.layers.Conv2D(channels,kernel, strides=stride, padding="same", use_bias=False)

class ResnetBlock(tf.Module):
    def __init__(self, channels, strides=1, residual_path=False):
        self.channels = channels
        self.stride = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels=self.channels, stride=self.stride)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels=self.channels, stride=self.stride)
        self.bn2 = keras.layers.BatchNormalization()

        if self.residual_path:
            self.down_conv = conv3x3(channels, strides, kernel=(1,1))
            self.down_bn = keras.layers.BatchNormalization()

    def __call__(self, inputs):
        residual = inputs
        x = self.bn1(inputs)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_bn(inputs)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)        
        x = x + residual
        return x
    
class ResNet(tf.Module):
    def __init__(self, block_list, num_classes, initial_filter=16, **kwargs):
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.in_channels = initial_filter
        self.out_channels = initial_filter
        self.conv_initial = conv3x3(self.out_channels)
        self.blocks = []

        for block_id in range(self.num_blocks):
            for layer_id in range(block_list[block_id]):
                if block_id!=0 and layer_id==0:
                    block = ResnetBlock(self.out_channels, strides=1, residual_path=True)
                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels, residual_path=residual_path)

                self.in_channels = self.out_channels
                self.blocks.append(block)
            self.out_channels *= 2

        self.final_bn = keras.layers.BatchNormalization()
        self.avg_pool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def __call__(self, inputs):
        out = self.conv_initial(inputs)
        for block in self.blocks:
            out = block(out)
        out = self.final_bn(out)
        out = tf.nn.relu(out)
        out = self.avg_pool(out)
        out = self.fc(out)
        return out

@tf.function
def compute_loss(logits, labels):
  return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
  prediction = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

def main():
    num_classes = 10
    batch_size = 32
    epochs = 10

    model = ResNet([2,2,2], num_classes)
    optimizer = keras.optimizers.Adam()

    x_train, y_train, x_test, y_test = load_dataset()
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds = ds.shuffle(60000).batch(batch_size)

    for epoch in range(epochs):
        for step, (x,y) in enumerate(ds):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = compute_loss(logits=logits, labels=y)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                accuracy = compute_accuracy(logits=logits, labels=y)
        
        print("Loss: ",loss.numpy(), "Acc: ",acc.numpy())

if __name__=="__main__":
  main()

  