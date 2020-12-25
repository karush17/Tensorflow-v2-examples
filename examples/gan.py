import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

class Generator(tf.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_f = 512
        self.n_k = 4

        self.dense1 = keras.layers.Dense(3*3*self.n_f)
        self.conv2 = keras.layers.Conv2DTranspose(self.n_f // 2, 3,2, 'valid')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2DTranspose(self.n_f // 4, self.n_k, 2, padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2DTranspose(1, self.n_k, 2, padding='same')

    def __call__(self, inputs):
        x = tf.nn.leaky_relu(tf.reshape(self.dense1(inputs), shape=[-1,3,3,self.n_f]))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x)))
        x = tf.tanh(self.conv4(x))
        return x

class Discriminator(tf.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_f = 64
        self.n_k = 4

        self.conv1 = keras.layers.Conv2D(self.n_f, self.n_k, 2, 'same')
        self.conv2 = keras.layers.Conv2D(self.n_f*2, self.n_k, 2, 'same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(self.n_f*4, self.n_k, 2, 'same')
        self.bn3 = keras.layers.BatchNormalization()
        self.flatten4 = keras.layers.Flatten()
        self.dense4 = keras.layers.Dense(1)

    def __call__(self, inputs):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dense4(self.flatten4(x))
        return x

def celoss_ones(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits)*(1-smooth)))

def celoss_zeros(logits, smooth=0.0):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits)*(1-smooth)))

def d_loss_fn(generator, discriminator, input_noise, real_image):
    fake_image = generator(input_noise)
    d_real_logits = discriminator(real_image)
    d_fake_logits = discriminator(fake_image)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    loss = d_loss_real + d_loss_fake
    return loss

def g_loss_fn(generator, discriminator, input_noise):
    fake_image = generator(input_noise)
    d_fake_logits = discriminator(fake_image)
    loss = celoss_ones(d_fake_logits)
    return loss

def main():
    tf.random.set_seed(22)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "22"

    z_dim = 100
    epochs = 10
    batch_size = 128
    learning_rate = 0.0002

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = tf.cast(x_train, tf.float32) / 255
    db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(batch_size*4).batch(batch_size).repeat()
    db_iter = iter(db)
    input_shape = [-1, 28,28,1]

    generator = Generator()
    discriminator = Discriminator()

    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        batch_x = next(db_iter)
        batch_x = tf.reshape(batch_x, shape=input_shape)
        batch_x = batch_x*2 - 1

        batch_z = tf.random.uniform(shape=[batch_size, z_dim], minval=-1, maxval=1)
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        print("Epoch: ", epoch, "G_Loss: ",g_loss, "D_Loss:", d_loss)

if __name__=="__main__":
  main()

  

  