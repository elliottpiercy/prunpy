import numpy as np
import tensorflow as tf

"""
Magnitude percentage weight pruning schedule - Mask all weights that are within the percentage threshold.

EG: threshold = 10%. All weights in the middle 10% (approximately -5% to -10%) are prunned. Example assume gaussian distributed weights.

"""


class schedule(tf.keras.callbacks.Callback):
    

    # Set pruning configuration
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
            
        if pruning_config['threshold'] < 0  or pruning_config['threshold'] > 1:
            raise ValueError('Percentage threshold must be between 0 and 1 (inclusive)')

            
    def on_epoch_begin(self, epoch, logs=None):
        
        # Update mask for weight vector based on new weights
        def get_mask(weights):

            # Return bottom n% of weight magnitudes. Numpy argsort doesnt work on multidimensional arrays. 
            def abs_percentage_threshold(data):
                return data.argsort()[:int(len(data) * self.pruning_config['threshold'])]


            weights_shape = weights.shape    
            weights = np.abs(weights).reshape(-1)
            mask = np.ones(weights.shape)
            
            locations = abs_percentage_threshold(weights)
            mask[locations] = 0
            mask = mask.reshape(weights_shape)

            return mask

        
        # Apply magnitude pruning schedule
        def apply_mask(weights, mask, epoch):
            return weights * mask
        

        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            
            # Create new weight matrix and set new weights
            new_weights = []
            for weights in self.model.get_weights():


                layer_mask = get_mask(weights)
                pruned_weights = apply_mask(weights, layer_mask, epoch)
                new_weights.append(pruned_weights)

            self.model.set_weights(new_weights)

        

    
        
        
        
        