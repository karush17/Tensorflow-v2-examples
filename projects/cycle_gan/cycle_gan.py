import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class Encoder(tf.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, kernel_size=7, strides=1)
        self.conv2 = keras.layers.Conv2D(64, kernel_size=7, strides=1, padding='same')
        self.conv3 = keras.layers.Conv2D(128, kernel_size=7, strides=1, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def __call__(self, inputs):
        x = tf.pad([inputs, [[0,0], [3,3], [3,3], [0,0]]], "REFLECT")\
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        return x

class Residual(tf.Module):
    def __init__(self):
        super(Residual, self).__init__()
        self.conv1 = keras.layers.Conv2D(128, kernel_size=3, strides=1)
        self.conv2 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
    
    def __call__(self, inputs):
        x = tf.pad(inputs, [[0,0], [1,1], [1,1], [0,0]], "REFLECT")
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], "REFLECT")
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = tf.add(x, inputs)
        return x

class Decoder(tf.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same',
                                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same',
                                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(3, kernel_size=7, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def __call__(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.tanh(x)
        return x


class Generator(tf.Module):

    def __init__(self, img_size=256, skip=False):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.skip = skip  # TODO: Add skip
        self.encoder = Encoder()
        if (img_size == 128):
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
        else:
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
            self.res7 = Residual()
            self.res8 = Residual()
            self.res9 = Residual()
        self.decoder = Decoder()

    def __call__(self, inputs, training=True):
        x = self.encoder(inputs, training)
        if (self.img_size == 128):
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
        else:
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
            x = self.res7(x, training)
            x = self.res8(x, training)
            x = self.res9(x, training)
        x = self.decoder(x, training)
        return x


class Discriminator(tf.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.leaky = keras.layers.LeakyReLU(0.2)
        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def __call__(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.leaky(x)
        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.leaky(x)
        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = self.leaky(x)
        x = self.conv5(x)
        # x = tf.nn.sigmoid(x) # use_sigmoid = not lsgan
        return x


def discriminator_loss(disc_of_real_output, disc_of_gen_output, lsgan=True):
    if lsgan:
        real_loss = keras.layers.mean_squared_error(disc_of_real_output, tf.ones_like(disc_of_real_output))
        generated_loss = tf.reduce_mean(tf.square(disc_of_gen_output))
        total_disc_loss = (real_loss+generated_loss)*0.5
    else:
        raise NotImplementedError
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_real_output), logits=disc_of_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_of_gen_output), logits=disc_of_gen_output)
        total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_of_gen_output, lsgan=True):
    if lsgan:
        gen_loss = keras.losses.mean_squared_error.(disc_of_gen_output, tf.ones_like(disc_of_gen_output))
    else:
        raise NotImplementedError
        gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_gen_output), logits=disc_of_gen_output)
    return gen_loss

def cycle_consistency_loss(data_A, data_B, recon_A, recon_B, cyc_lambda=10):
    loss tf.reduce_mean(tf.abs(data_A - recon_A) + tf.abs(data_B - recon_B))
    return cyc_lambda*loss


tf.random.set_seed(22)

learning_rate = 0.0002
batch_size = 1
img_size = 256
cyc_lambda = 10
epochs = 10

path_to_zip = keras.utils.get_file('horse2zebra.zip',
                              cache_subdir=os.path.abspath('.'),
                              origin='https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
                              extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'horse2zebra/')

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtpye(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = image*2 - 1
    return image

train_datasetA = tf.data.Dataset.list_files(PATH + 'trainA/*.jpg', shuffle=False)
train_datasetA = train_datasetA.shuffle(trainA_size).repeat(epochs)
train_datasetA = train_datasetA.map(lambda x: load_image(x))
train_datasetA = train_datasetA.batch(batch_size)
train_datasetA = train_datasetA.prefetch(batch_size)
train_datasetA = iter(train_datasetA)

train_datasetB = tf.data.Dataset.list_files(PATH + 'trainB/*.jpg', shuffle=False)
train_datasetB = train_datasetB.shuffle(trainB_size).repeat(epochs)
train_datasetB = train_datasetB.map(lambda x: load_image(x))
train_datasetB = train_datasetB.batch(batch_size)
train_datasetB = train_datasetB.prefetch(batch_size)
train_datasetB = iter(train_datasetB)

a = next(train_datasetA)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

discA = Discriminator()
discB = Discriminator()
genA2B = Generator()
genB2A = Generator()

discA_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
discB_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
genA2B_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
genB2A_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)

def generate_image(A, B, B2A, A2B, epoch):
    A = tf.reshape(A, [256, 256, 3]).numpy()
    B = tf.reshape(B, [256, 256, 3]).numpy()
    A2B = tf.reshape(A2B, [256, 256, 3]).numpy()
    B2A = tf.reshape(B2A, [256, 256, 3]).numpy()

def train(train_datasetA, train_datasetB):
    for epoch in range(epochs):
        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
            try:
                trainA = next(train_datasetA)
                trainB = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, ran out of data")
                break
            genA2B_output = genA2B(trainA)
            genB2A_output = genB2A(trainB)
            discA_real_output = discA(trainA)
            discB_real_output = discB(trainB)
            discA_fake_output = discA(genB2A_output)
            discB_fake_output = discB(genA2B_output)
            reconA = genB2A(genA2B_output)
            reconB = genA2B(genB2A_output)

            discA_loss = discriminator_loss(discA_real_output, discA_fake_output,lsgan=lsgan)
            discB_loss = discriminator_loss(discB_real_output, discB_fake_output,lsgan=lsgan)
            genA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan) + cycle_consistency_loss(trainA, trainB, recon_A, recon_B)
            genB2A_loss = generator_loss(discA_fake_output, lsgan=lsgan) + cycle_consistency_loss(trainA, trainB, recon_A, recon_B)

            genA2B_grads = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)
            genB2A_grads = genB2A_tape.gradient(genB2A_loss, genB2A.trainable_variables)
            discA_grads = discA_tape.gradient(discA_loss, discA.trainable_variables)
            discB_grads = discB_tape.gradient(discB_loss, discB.trainable_variables)

            genA2B_optimizer.apply_gradients(zip(genA2B_grads, genA2B.trainable_variables))
            genB2A_optimizer.apply_gradients(zip(genB2A_grads, genB2A.trainable_variables))
            discA_optimizer.apply_gradients(zip(discA_grads, discA.trainable_variables))
            discB_optimizer.apply_gradients(zip(discB_grads, discB.trainable_variables))


        print("Epoch: ", epoch, "Total Generator Loss: ", genA2B_loss.numpy() + genB2A_loss.numpy(), "Total Discriminator Loss: ", discA_loss.numpy() + discB_loss.numpy())



