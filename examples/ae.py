import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_train, tf.float32) / 255

image_size = 28*28
h_dim = 20
num_epochs = 20
batch_size = 100
learning_rate = 1e-3

class AE(tf.Module):
    def __init_(self):
        super(AE, self).__init__()
        self.fc1 = keras.layers.Dense(512)
        self.fc2 = keras.layers.Dense(h_dim)
        self..fc3 = keras.layers.Dense(512)
        self.fc4 = keras.layers.Dense(image_size)

    def encode(self, x):
        x = tf.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def decode_logits(self, x):
        x = tf.nn.relu(self.fc3(h))
        x = self.fc4(x)
        return x

    def decode(self, x):
        return tf.nn.sigmoid(self.decode_logits(x))
    
    def __call__(self, inputs):
        h = self.encoder(inputs)
        x_recon = self.decoder(h)
        return x_recon
    
@tf.function
def compute_loss(labels, logits, batch_size):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labesl, logits=logits)) / batch_size
model = AE()
optimizer = keras.optimizers.Adam(learning_rate)

ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batch_size*5).batch(batch_size)
num_batches = x_train.shape[0] // batch_size

for epoch in range(epochs):
    for step, (x,y) in enumerate(ds):
        x = tf.reshape(x, [-1, image_size])
        with tf.GradientTape() as tape:
            x_recon = model(x)
            loss = compute_loss(labels=y, logits=x_recon, batch_size=batch_size)
            grads = tape.gradient(loss, model.trainable_variables)
            grads,_ = tf.clip_by_global_norm(grads, 15)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Epoch: ", epoch, "Loss: ", loss.numpy())



