import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
import numpy as np

class DSDTraining(tf.keras.Model):
    
    def __init__(self, model, sparsity):
        super(DSDTraining, self).__init__()
        
        self.model = model
        self.sparsity = sparsity
        self.train_on_sparse = False
        
        # Init masks.
        self.reset_masks()
    
    def reset_masks(self):
        
        self.masks = []
        
        for layer in self.model.layers:
            w, b = layer.get_weights()
            self.masks.append(tf.ones_like(w))
            self.masks.append(tf.ones_like(b))
        
        return self.masks
    
    def update_masks(self, trainable_vars):
        
        for i, wb in enumerate(trainable_vars):
            qk = tfp.stats.percentile(tf.math.abs(wb), q = self.sparsity * 100)
            mask = tf.where(tf.math.abs(wb) < qk, 0., 1.)
            # Keep track of masks for "Training on Sparse" step.
            self.masks[i] = mask
    
    def apply_masks(self, trainable_vars, gradients):
        
        for i, (wb, grad, mask) in enumerate(zip(trainable_vars, gradients, self.masks)):
            # Weights/biases.
            trainable_vars[i].assign(tf.multiply(wb, mask))
            # Gradients.
            gradients[i] = tf.multiply(grad, mask) 
        
        return gradients
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        # Get trainable weights/biases
        trainable_vars = self.model.trainable_variables
        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply masks
        if self.train_on_sparse:
            gradients = self.apply_masks(trainable_vars, gradients)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, x):
        x = self.model(x)
        return x

class UpdateMasks(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        trainable_vars = self.model.trainable_variables
        self.model.update_masks(trainable_vars)