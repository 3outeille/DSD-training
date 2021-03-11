import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras import datasets, layers, utils, Sequential, Model
import matplotlib.pyplot as plt

def plot_wb(model, ranges=None):
    
    for i, l in enumerate(model.layers[:-1]):
        # Plot.
        fig = plt.figure(figsize=(15,5))

        fig.add_subplot(1,2,1)
        plt.title("weights " + l.name)
        plt.ylim(top=20000)
        w = l.get_weights()[0].flatten()
        plt.hist(w, bins=100, range=ranges);

        fig.add_subplot(1,2,2)
        plt.title("biases " + l.name)
        plt.ylim(top=20)
        b = l.get_weights()[1].flatten()
        plt.hist(b, bins=100, range=ranges);

def wb_non_zero_percentage(model):
    pos_last_layer = len(model.layers) - 1

    for i, l in enumerate(model.layers):

        # Take all dense layer except last one,
        if i != pos_last_layer and  i % 2 == 0:
            w, b = model.layers[0].get_weights()

            non_zero_w = tf.math.count_nonzero(w)
            total_w = tf.reshape(w, [-1]).shape[0]
            res_w = (non_zero_w / total_w).numpy() * 100

            non_zero_b = tf.math.count_nonzero(w)
            total_b = tf.reshape(w, [-1]).shape[0]
            res_b = (non_zero_b / total_b).numpy() * 100

            print("Percentage of non-zero value " + l.name + ": w = {} | b = {}".format(res_w, res_b))