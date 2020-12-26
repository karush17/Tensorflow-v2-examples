from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, datasets

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
num_classes = 4
num_gpus = 2
learning_rate = 0.001
training_steps = 1000
batch_size = 1024*num_gpus

conv1_filters = 64
conv2_filters = 128
conv3_filters = 256
fc1_units = 2048

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_test, tf.float32) / 255
y_train, y_test = tf.cast(tf.reshape(y_train, (-1)), tf.int64), tf.cast(tf.reshape(y_test, (-1)), tf.int64)
db = tf.data.Datset.from_tensor_slices((x_train, y_train)).repeat().shuffle(batch_size*10).batch(batch_size).prefetch(num_gpus)

class ConvNet(tf.Module):
    # Set layers.
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv1_1 = layers.Conv2D(conv1_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv1_2 = layers.Conv2D(conv1_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        # Convolution Layer with 128 filters and a kernel size of 3.
        self.conv2_1 = layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv2_2 = layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv2_3 = layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        # Convolution Layer with 256 filters and a kernel size of 3.
        self.conv3_1 = layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv3_2 = layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        self.conv3_3 = layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)
        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()
        # Fully connected layer.
        self.fc1 = layers.Dense(1024, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)
        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    @tf.function
    def __call__(self, x, is_training=False):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

@tf.function
def cross_entropy_loss(x, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x))

@tf.function
def accuracy(y_pred, y_true):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), y_true), tf.float32), axis=-1)    

def backprop(batch_x, batch_y, conv_net, trainable_variables):
    with tf.GradientTape() as tape:
        pred = conv_net(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        grads = tape.gradient(loss, trainable_variables)
    return grads

@tf.function
def average_gradients(tower_grads):
    avg_grads = []
    for tgrads in zip(*tower_grads):
        grads = []
        for g in tgrads:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        avg_grads.append(grad)
    return avg_grads

with tf.device('/cpu:0'):
    conv_net = ConvNet()
    optimizer = tf.optimizers.Adam(learning_rate)

def run_optimization(x, y):
    tower_grads = []
    trainable_variables = conv_net.trainable_variables

    with tf.device('/cpu:0'):
        for i in range(num_gpus):
            gpu_batch_size = int(batch_size / num_gpus)
            batch_x = x[i*gpu_batch_size:(i+1)*gpu_batch_size]
            batch_y = y[i*gpu_batch_size:(i+1)*gpu_batch_size]

            with tf.device('/gpu:%i' %i):
                grad = backprop(batch_x, batch_y, conv_net, trainable_variables)
                tower_grads.append(grad)

                if i==num_gpus-1:
                    gradients = average_gradients(tower_grads)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

ts = time.time()
for step, (batch_x, batch_y) in enumerate(db.take(training_steps), 1):
    run_optimization(batch_x, batch_y)

    if step % 100==0 or step==1:
        dt = time.time() - ts
        speed = batch_size*100 / dt
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Step: ", step, "Loss: ", loss.numpy(), "Acc: ", acc.numpy(), "Speed: ", speed)




