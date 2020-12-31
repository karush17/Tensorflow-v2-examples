<p align="center"><img src="images\tensorflow-2.0.gif" height="300" width="300"></img></p>

## Introduction  
This repository is a collection of programs written as examples in Tensorflow V2. Programs are combined into following two sections-  
* Example scripts  
* Collaborative projects  

## Motivation
Primary objective of the project is to gain an in-depth understanding of functional and operational characteristics of Tensorflow V2. These include Tensorflow syntax, Keras API, control flow and dynamic computation of graphs. Example scripts will build the understanding of writing programs in the basic Tensorflow structure. Collaborative projects will lead to the development of practical Tensorflow aplications for solving a challenging problem.  

## Study Material
 - [ ] [Tensorflow Documentation](https://www.tensorflow.org/api_docs) (primary resource)  
 - [ ] [Repository of Tensorflow v2 Examples](https://github.com/dragen1860/TensorFlow-2.x-Tutorials) (primary resource)  
 - [ ] [Tensorflow for Beginners](https://www.tensorflow.org/tutorials) (secondary resource)
 - [ ] [Old Tensorflow repository](https://github.com/aymericdamien/TensorFlow-Examples) (just in case)  

## Agenda
Following is the tentative agenda-  

| Date       | Topic                            | Programs               | Reading |Remarks (weak points)   |
|:----------:|:--------------------------------:|:----------------------:|:-------:|:--------:|
| 21/12/2020 |Introduction                      |ANN, CNN, Autograph, LR | [new features](https://www.tensorflow.org/guide/effective_tf2)| use tf.float32 from now on         |
| 22/12/2020 |Deep Convolutional Models-1       |Inception               | [tf.Module()](https://www.tensorflow.org/api_docs/python/tf/Module)| optmize graph using @tf.function|
| 23/12/2020 |Deep Convolutional Models-2       |VGG, ResNet             | [tf.GradientTape()](https://www.tensorflow.org/api_docs/python/tf/GradientTape)|tf.Session() replaced with tf.GradientTape()          |
| 24/12/2020 |Memory-based Models               |RNN, LSTM               | [RNNs in Keras](https://www.tensorflow.org/guide/keras/rnn)| tf.Module() preferred         |
| 25/12/2020 |Adversarial Learning              |GAN, Pix2Pix, CycleGAN  | [DCGAN](https://www.tensorflow.org/tutorials/generative/dcgan)| list-concatenated layers for larger models         |
| 26/12/2020 |Distributed Training              |DistGPU                 | [Distributed Training in TF](https://www.tensorflow.org/guide/distributed_training)| average tower gradients for each GPU udring training, manually set tf.device() flag         |
| 27/12/2020 |Test-1                            |implement 1 program     |CycleGAN         |-          |
| 28/12/2020 |Language Models                   |GPT              | [Transformers in Tensorflow](https://www.tensorflow.org/tutorials/text/transformer)| use predefined layers for large models, revise attention mechanisms|
| 29/12/2020 |Case Study                        |Training BERT in Tensorflow| [BERT for classification](https://www.tensorflow.org/tutorials/text/classify_text_with_bert), [Fine-tuning BERT](https://www.tensorflow.org/official_models/fine_tuning_bert), [Solving GLUE tasks using BERT](https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu)|          |
| 30/12/2020 |Reinforcement Learning            |A2C            | [Dreamer](https://github.com/danijar/dreamer)| use Keras API for combined model loss, model.train_on_bathc(obs, [inp1, inp2])         |
| 31/01/2021 |Test-2                            |implement 1 program     |DDQN         |use target_variable.assign() for soft param updates in tf.Module or model.set_weights() in Keras API|
| 01/01/2021 |Project Day                       |DreamerV2               |[DreamerV2](https://github.com/danijar/dreamerv2)         |          |
| 02/01/2021 |Project Day                       |DreamerV2               |[DreamerV2](https://github.com/danijar/dreamerv2)         |          |
| 03/01/2021 |Project Day                       |DreamerV2               |[DreamerV2](https://github.com/danijar/dreamerv2)         |          |

## Potential Project Topics

- [ ] [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan)
- [ ] [DeepDream](https://www.tensorflow.org/tutorials/generative/deepdream)
- [ ] [DCGAN](https://www.tensorflow.org/tutorials/generative/dcgan)
- [ ] [Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix)
- [ ] [Adversarial FGSM](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
- [ ] [VAE](https://www.tensorflow.org/tutorials/generative/cvae)




