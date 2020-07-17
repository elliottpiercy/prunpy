import numpy as np
import tensorflow as tf
import utils.prune.helper


"""
Magnitude weight pruning schedule - Mask all weights that are within the threshold.

EG: threshold = 0.1. ALl weights between -0.1 and 0.1 are masked with 0s

"""


class schedule(tf.keras.callbacks.Callback):
    

    # Set pruning configuration
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
        self.mask_exists = False
        self.gradual_counter = 1
        self.valid_layers = 'dense'
            
            
    # Initialise pruning mask (0s of network shape)
    def create_ones_mask(self):
        
        masks = []
        for weights in self.model.get_weights():

            layer_mask = np.ones(weights.shape)
            masks.append(layer_mask)
            
        return masks
    
    
    # Update mask for weight vector based on new weights
    def get_layer_mask(self, weights, sparsity, ID):

        # Return locations where value is between bounds
        def abs_thresholding(data, sparsity):
            return np.where(np.abs(data) < sparsity)

        mask = np.ones(weights.shape)
        
                # If the layer is a valid layer to prune. Locate indexs to mask. Otherwise keep mask of 1s (no pruning)
        if utils.prune.helper._validate_layer(ID, self.valid_layers):
            locations = abs_thresholding(weights, sparsity)
            mask[locations] = 0
            
        return mask

    
    
     # Apply magnitude pruning schedule
    def apply_layer_mask(self, weights, mask):
        return weights * mask
        
            

    # Create new weight matrix using sparsity value. Return new weights and mask
    def get_masked_weights(self, sparsity):

        masks = []
        new_weights = []
        for layer, weights in zip(self.model.layers, self.model.get_weights()):


            layer_mask = self.get_layer_mask(weights, sparsity, layer.name)
            masks.append(layer_mask)

            pruned_weights = self.apply_layer_mask(weights, layer_mask)
            new_weights.append(pruned_weights)

        return new_weights, masks   
    
    
    
    # Prune at the beginning of each epoch
    def on_epoch_begin(self, epoch, logs=None):
        

        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            
            
            # Creates a static mask in one shot
            if self.pruning_config['function'] == 'one_shot_static':
                
                if self.mask_exists:
                    masks = self.model.masks 
                
                else:
 
                    new_weights, masks = self.get_masked_weights(self.pruning_config['threshold'])
                    self.model.set_weights(new_weights)
                    self.mask_exists = True
            
                
            # Creates a new mask each epoch
            elif self.pruning_config['function'] == 'one_shot':

                new_weights, masks = self.get_masked_weights(self.pruning_config['threshold'])
                self.model.set_weights(new_weights)
            
            
            
                
            # Gradual convergence results in a static mask
            elif self.pruning_config['function'] == 'gradual':
                
                sparsity = self.pruning_config['threshold'] * (self.gradual_counter / self.pruning_config['converge_over'])
                self.gradual_counter += 1
                
                if sparsity > self.pruning_config['threshold']:
                    sparsity = self.pruning_config['threshold']
                    
                if self.mask_exists:
                    masks = self.model.masks 
                
                else:
 
                    new_weights, masks = self.get_masked_weights(sparsity)
                    self.model.set_weights(new_weights)
        
                    # If we have hit max pruning -- keep a static mask
                    if sparsity == self.pruning_config['threshold']:
                        self.mask_exists = True
                    
    
    
        else:
            masks = self.create_ones_mask()
            
       
        self.model.masks = masks
        
 

