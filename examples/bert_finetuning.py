## import dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

# setup dataset urls
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

glue, info = tfds.load('glue/mrpc', with_info=True, batch_size=-1)

list(glue.keys())
info.features
info.features['label'].names

glue_train = glue['train']
for key, value in glue_train.items():
    print(f"{key:9s}: {value[0].numpy()}")

# tokenize vocabulary
tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(gs_folder_bert, "vocab.txt"), do_lower_case=True)
print("Vocab Size: ", len(tokenizer.vocab))

tokens = tokenizer.tokenize("Hello Tensorflow!")
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

# preprocess data
def encode_sentence(s):
    tokens = list(tokenizer.tokenize(s.numpy()))
    tokens.append(['[SEP]'])
    return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(glue_dict, tokenizer):
    sentence1 = tf.ragged.constant([encode_sequence(s) for s in glue_dict["sentence1"]])
    sentence2 = tf.ragged.constant([encode_sequence(s) for s in glue_dict["sentence2"]])

    # pre append classification tokens
    cls = [tokenizer.convert_tokens_to_ids(["[CLS]"])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    # make mask
    input_mask = tf.ones_like(input_word_ids).to_tensor()
    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.zeros_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    # dict input to bert model
    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }
    return inputs

glue_train = bert_encode(glue['train'], tokenizer)
glue_train_labels = glue['train']['label']

glue_validation = bert_encode(glue['validation'], tokenizer)
glue_validation_labels = glue['validation']['label']

glue_test = bert_encode(glue['test'], tokenizer)
glue_test_labels = glue['test']['label']

for key, value in glue_train.items():
  print(f'{key:15s} shape: {value.shape}')

print(f'glue_train_labels shape: {glue_train_labels.shape}')

# build model from dictionary
import json

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
bert_config = bert.configs.BertConfig.from_dict(config_dict)
config_dict

bert_classifier,bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)

# run model on test batch of 10 samples
glue_batch = {key: val[:10] for key, val in glue_train.items()}
bert_classifier(glue_batch, training=True).numpy()

# restore encoder weigths
checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

# prepare optimizer
epochs = 3
batch_size = 32
eval_batch_size = 32

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch*epochs
warmup_steps = int(epochs*train_data_size*0.1 / batch_size)

optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

# train the model
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
bert_classifier.fit(glue_train, glue_train_labels, validation_data=(glue_validation, glue_validation_labels), batch_size=32, epochs=epochs)

# run finetuned model on a pair
my_examples = bert_encode(
    glue_dict = {
        'sentence1':[
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2':[
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    },
    tokenizer=tokenizer)

result = bert_classifier(my_examples, training=False)
result = tf.argmax(result).numpy()
result

# save model
export_dir = './saved_model'
tf.saved_model.save(bert_classifier, export_dir=export_dir)

# reload saved model
reloaded = tf.saved_model.load(export_dir)





