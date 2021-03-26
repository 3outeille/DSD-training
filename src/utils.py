import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

def plot_wb(model, ranges=None):
    
    # According to paper, first conv layer is ignored.
    for i, l in enumerate(model.layers[1:]):
        w = l.get_weights()[0].flatten()
        b = l.get_weights()[1].flatten()

        # Plot.
        fig = plt.figure(figsize=(15,5))

        fig.add_subplot(1,2,1)
        plt.title("weights " + l.name)
        plt.hist(w, bins=100, range=ranges);

        fig.add_subplot(1,2,2)
        plt.title("biases " + l.name)
        plt.hist(b, bins=100, range=ranges);

def wb_non_zero_percentage(model):

    for i, l in enumerate(model.layers):
        # Take all dense layer except last one,
        if i > 0 and ("conv" in l.name or "dense" in l.name):
            w, b = model.layers[0].get_weights()

            non_zero_w = tf.math.count_nonzero(w)
            total_w = tf.reshape(w, [-1]).shape[0]
            res_w = (non_zero_w / total_w).numpy() * 100

            non_zero_b = tf.math.count_nonzero(w)
            total_b = tf.reshape(w, [-1]).shape[0]
            res_b = (non_zero_b / total_b).numpy() * 100

            print("Percentage of non-zero value " + l.name + ": w = {} | b = {}".format(res_w, res_b))