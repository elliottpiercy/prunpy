import json
import numpy as np
import tensorflow as tf

"""
Magnitude weight pruning schedule - Mask all weights that are within the threshold.

EG: threshold = 0.1. ALl weights between -0.1 and 0.1 are masked with 0s

"""

class schedule(tf.keras.callbacks.Callback):
    

    # Set pruning configuration
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        # Update mask for weight vector based on new weights
        def get_mask(weights):

            # Return locations where value is between bounds
            def abs_thresholding(data):
                return np.where(np.abs(data) < self.pruning_config['threshold'])

            mask = np.ones(weights.shape)
            locations = abs_thresholding(weights)
            mask[locations] = 0
            return mask


        # Apply magnitude pruning schedule
        def apply_mask(weights, mask, epoch):
            return weights * mask
        
        

        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            
            # Create new weight matrix and set new weights
            new_weights = []
            for idx, weights in enumerate(self.model.get_weights()):


                layer_mask = get_mask(weights)
                pruned_weights = apply_mask(weights, layer_mask, epoch)
                new_weights.append(pruned_weights)

            self.model.set_weights(new_weights)

        
        
        