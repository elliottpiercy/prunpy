import numpy as np
import tensorflow as tf
import utils.prune.helper

"""
Learning rate rewind pruning schedule 
Mask all weights that are within the percentage threshold and reset the learing rate. Retrain to within an accuracy threshold then increase sparsity and continue pruninng until specific compression is reached.
"""

class schedule(tf.keras.callbacks.Callback):
    

    # Set pruning configuration
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
        self.mask_exists = False
        self.rolling_sparsity = 0
        self.valid_layers = 'dense'
        
        self.masks = None
        self.target_accuracy = None
        
        

    # Initialise pruning mask (0s of network shape)
    def create_ones_mask(self):
        
        masks = []
        for weights in self.model.get_weights():

            layer_mask = np.ones(weights.shape)
            masks.append(layer_mask)
            
        return masks
    
    
    # Reset learning rate to 'initial_learning_rate'
    def reset_learning_rate(self):
        self.model.optimizer.learning_rate.assign(self.pruning_config['initial_learning_rate'])

            
            
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
        
            
            
    # Get static masks for the model
    def get_model_mask(self, sparsity):

        masks = []
        for layer, weights in zip(self.model.layers, self.model.get_weights()):
            layer_mask = self.get_layer_mask(weights, sparsity, layer.name)
            masks.append(layer_mask)

        return masks


    # Apply magnitude pruning schedule
    def apply_layer_mask(self, weights, mask):
        return weights * mask
        
            
            
    # Create new weight matrix using sparsity value. Return new weights and mask
    def get_masked_weights(self, masks):

        new_weights = []
        for layer_mask, weights in zip(masks, self.model.get_weights()):

            pruned_weights = self.apply_layer_mask(weights, layer_mask)
            new_weights.append(pruned_weights)

        return new_weights   
    
    
        
    def on_epoch_begin(self, epoch, logs=None):
        

        if epoch >= self.pruning_config['epoch_threshold']:
            
            if self.target_accuracy == None:
                self.target_accuracy = np.max(self.model.history.history['accuracy'])

            
            """
            If within accuracy bounds
                1. Reset learning rate
                2. Increase sparsity
                3. Create new masks
            
            """ 
            
            # Just a quick scan -- this doesnt look right... operation order is a bit off
            if self.model.history.history['accuracy'][-1] >= (self.target_accuracy - self.pruning_config['eps']):
                print(self.model.history.history['accuracy'][-1], (self.target_accuracy - self.pruning_config['eps']))
                
                print('Inside loop')
                # Learning rate rewinding
                print('Reset learning rate')
                self.reset_learning_rate()
                
                
                # If the rolling sparsity is 0 (first iteration). Set it to the initial threshold. Else += threshold_step
                if self.rolling_sparsity == 0:
                    self.rolling_sparsity += self.pruning_config['threshold']
                else:
                    self.rolling_sparsity += self.pruning_config['threshold_step']
                    
                    
                print('Rolling sparsity:' , self.rolling_sparsity)
                self.mask_exists = True
                
                print('Creating new mask')
                self.masks = self.get_model_mask(self.rolling_sparsity)
                
                
                

            print('Pruning using existing mask')
            new_weights = self.get_masked_weights(self.masks)
            self.model.set_weights(new_weights)


    
        else:
            self.masks = self.create_ones_mask()
            
            
        self.model.masks = self.masks
            
                