import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
tf.config.run_functions_eagerly(True)

tf.random.set_seed(22)
os.environ["TF_MIN_CPP_LOG_LEVEL"] = "2"

top_words = 1000
max_review_length = 80
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length)
x_train, x_test = tf.cast(x_train, tf.float32), tf.cast(x_test, tf.float32)
y_train, y_test = tf.expand_dims(tf.cast(y_train, tf.float32),1), tf.expand_dims(tf.cast(y_test, tf.float32),1)

class RNN(tf.Module):
    def __init__(self, units, num_classes, num_layers):
        super(RNN, self).__init__()
        self.rnn = keras.layers.LSTM(units, return_sequences=True)
        self.rnn2 = keras.layers.LSTM(units)
        self.embedding = keras.layers.Embedding(top_words, 100, input_length=max_review_length)
        self.fc = keras.layers.Dense(1)

    def __call__(self, inputs):
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.rnn2(x)
        x = self.fc(x)
        return x

@tf.function
def compute_loss(logits, labels):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
  prediction = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

def main():
    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20

    model = RNN(units, num_classes, num_layers=2)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    for epoch in range(epochs):
        for step, (x,y) in enumerate(ds):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = compute_loss(logits=logits, labels=y)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print("Epoch: ",epoch, "Loss: ", loss.numpy())

if __name__=="__main__":
  main()

