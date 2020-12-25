import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

tf.random.set_seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = tf.cast(x_train, tf.float32) / 255, tf.cast(x_test, tf.float32) / 255

image_size = 28*28
h_dim = 512
z_dim = 20
num_epochs = 10
batch_size = 100
learning_rate = 1e-3

class VAE(tf.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.fc1 = keras.layers.Dense(h_dim)

        self.fc2 = keras.layers.Dense(z_dim)
        self.fc3 = keras.layers.Dense(z_dim)

        self.fc4 = keras.layers.Dense(h_dim)

        self.fc5 = keras.layers.Dense(image_size)

    def encode(self, x):
        h = tf.nn.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = tf.exp(log_var * 0.5)
        eps = tf.random.normal(std.shape)
        return mu + eps*std
    
    def decode_logits(self, z):
        h = tf.nn.relu(self.fc4(z))
        return self.fc5(h)
    
    def decode(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def __call__(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        x_reconstructed_logits = self.decode_logits(z)
        return x_reconstructed_logits, mu, log_var

model = VAE()
optimizer = keras.optimizers.Adam(learning_rate)
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batch_size*5).batch(batch_size)

num_batches = x_train.shape[0] // batch_size

for epoch in range(num_epochs):
    for step, x in enumerate(dataset):
        x = tf.reshape(x, [-1, image_size])
        with tf.GradientTape() as tape:
            x_reconstructed_logits, mu, log_var = model(x)
            reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstructed_logits)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss) / batch_size

            kl_div = -0.5 * tf.reduce_sum(1+log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
            kl_div = tf.reduce_mean(kl_div)

            loss = tf.reduce_mean(reconstruction_loss) + kl_div

            grads = tape.gradient(loss, model.trainable_variables)
            for g in grads:
                tf.clip_by_norm(g, 15)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    print("Epoch: ", epoch, "Loss: ", loss.numpy())

z = tf.random.normal((batch_size, z_dim))
out = model.decode(z)
out = tf.reshape(out, [-1,2,8,28,1]).numpy() * 255




