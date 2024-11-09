import logging
import time

import numpy as np

import tensorflow as tf

import Treecnn

class FCNN(tf.keras.Model):
  def __init__(self, *, dropout_rate=0.1):
    super().__init__()
    self.treecnn = Treecnn.TreeCNN()

    self.fcnn1 = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    self.fcnn2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),   
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(8),  
        tf.keras.layers.Dense(1)
    ])

  def call(self, inputs):
    if len(inputs) == 3:
      input, idx, mask = inputs
    else:
      input, idx = inputs
      mask = None
    output = self.treecnn(input, idx)
    
    output1 = self.fcnn1(output)
    output1 = tf.expand_dims(output1, axis=1)
    
    logits = self.fcnn2(output1+output)


    # Return the final output and the attention weights.
    return logits