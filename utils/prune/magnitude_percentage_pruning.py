import numpy as np
import tensorflow as tf
import utils.prune.helper

"""
Magnitude percentage weight pruning schedule - Mask all weights that are within the percentage threshold.

EG: threshold = 10%. All weights in the middle 10% (approximately -5% to -10%) are prunned. Example assume gaussian distributed weights.

"""


class schedule(tf.keras.callbacks.Callback):
    

    # Set pruning configuration
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
        self.mask_exists = False
        self.gradual_counter = 1
        self.valid_layers = 'dense'
            
        if pruning_config['threshold'] < 0  or pruning_config['threshold'] > 1:
            raise ValueError('Percentage threshold must be between 0 and 1 (inclusive)')
            
        
            
    # Initialise pruning mask (0s of network shape)
    def create_ones_mask(self):
        
        masks = []
        for weights in self.model.get_weights():

            layer_mask = np.ones(weights.shape)
            masks.append(layer_mask)
            
        return masks
    
    
    # Update mask for weight vector based on new weights
    def get_layer_mask(self, weights, sparsity, ID):

        # Return bottom n% of weight magnitudes. Numpy argsort doesnt work on multidimensional arrays. 
        def abs_percentage_threshold(data, sparsity):
            return data.argsort()[:int(len(data) * sparsity)]


        weights_shape = weights.shape    
        weights = np.abs(weights).reshape(-1)
        mask = np.ones(weights.shape)
        
        # If the layer is a valid layer to prune. Locate indexs to mask. Otherwise keep mask of 1s (no pruning)
        if utils.prune.helper._validate_layer(ID, self.valid_layers):
            locations = abs_percentage_threshold(weights, sparsity)
            mask[locations] = 0
        
        mask = mask.reshape(weights_shape)
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

        
            
    # Start pruning on beginning of epoch 
    def on_epoch_begin(self, epoch, logs=None):
        
        
        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            
            
            
            
            if self.pruning_config['function'] == 'one_shot_static':
                
                if self.mask_exists:
                    masks = self.model.masks 
                    sparsity = self.model.sparsity
                
                else:
                    
                    sparsity = self.pruning_config['threshold']
                    new_weights, masks = self.get_masked_weights(sparsity)
                    self.model.set_weights(new_weights)
                    self.mask_exists = True
            
                
            elif self.pruning_config['function'] == 'one_shot':
             
                sparsity = self.pruning_config['threshold']
                new_weights, masks = self.get_masked_weights(sparsity)
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
            sparsity = 0
            
            
        self.model.masks = masks
        self.model.sparsity = sparsity
                 
