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
        self.mask_exists = False
        self.gradual_counter = 1
            
        if pruning_config['threshold'] < 0  or pruning_config['threshold'] > 1:
            raise ValueError('Percentage threshold must be between 0 and 1 (inclusive)')

            
    def on_epoch_begin(self, epoch, logs=None):
        
        # Update mask for weight vector based on new weights
        def get_layer_mask(weights, sparsity):

            # Return locations where value is between bounds
            def abs_thresholding(data, sparsity):
                return np.where(np.abs(data) < sparsity)

            mask = np.ones(weights.shape)
            locations = abs_thresholding(weights, sparsity)
            mask[locations] = 0
            return mask

        
        # Apply magnitude pruning schedule
        def apply_layer_mask(weights, mask, epoch):
            return weights * mask
        
        
        
        # Create new weight matrix using sparsity value. Return new weights and mask
        def get_masked_weights(sparsity):

            masks = []
            new_weights = []
            for weights in self.model.get_weights():


                layer_mask = get_layer_mask(weights, sparsity)
                masks.append(layer_mask)

                pruned_weights = apply_layer_mask(weights, layer_mask, epoch)
                new_weights.append(pruned_weights)
                    
            return new_weights, masks   
        

        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            
            
            
            if self.pruning_config['function'] == 'one_shot_static':
                
                if self.mask_exists:
                    masks = self.model.masks 
                
                else:
 
                    new_weights, masks = get_masked_weights(self.pruning_config['threshold'])
                    self.model.set_weights(new_weights)
                    self.mask_exists = True
            
                
            elif self.pruning_config['function'] == 'one_shot':
             

                new_weights, masks = get_masked_weights(self.pruning_config['threshold'])
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
 
                    new_weights, masks = get_masked_weights(sparsity)
                    self.model.set_weights(new_weights)
        
                    # If we have hit max pruning -- keep a static mask
                    if sparsity == self.pruning_config['threshold']:
                        self.mask_exists = True
                    
#                 new_weights, masks = get_masked_weights(sparsity)
#                 self.model.set_weights(new_weights)
               
            
        else:
            
            masks = []
            for weights in self.model.get_weights():
                
                layer_mask = np.ones(weights.shape)
                masks.append(layer_mask)
                
                
        self.model.masks = masks
     
    






# import numpy as np
# import tensorflow as tf

# """
# Magnitude weight pruning schedule - Mask all weights that are within the threshold.

# EG: threshold = 0.1. ALl weights between -0.1 and 0.1 are masked with 0s

# """

# class schedule(tf.keras.callbacks.Callback):
    

#     # Set pruning configuration
#     def __init__(self, pruning_config):
#         self.pruning_config = pruning_config
    
    
#     def on_epoch_begin(self, epoch, logs=None):
        
#         # Update mask for weight vector based on new weights
#         def get_mask(weights):

#             # Return locations where value is between bounds
#             def abs_thresholding(data):
#                 return np.where(np.abs(data) < self.pruning_config['threshold'])

#             mask = np.ones(weights.shape)
#             locations = abs_thresholding(weights)
#             mask[locations] = 0
#             return mask


#         # Apply magnitude pruning schedule
#         def apply_mask(weights, mask, epoch):
#             return weights * mask
        
        

#         # If the threshold is 0 -- ignore and dont prune
#         if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            
#             # Create new weight matrix and set new weights
#             masks = []
#             new_weights = []
#             for idx, weights in enumerate(self.model.get_weights()):

                
#                 layer_mask = get_mask(weights)
#                 masks.append(layer_mask)
                
#                 pruned_weights = apply_mask(weights, layer_mask, epoch)
#                 new_weights.append(pruned_weights)

#             self.model.set_weights(new_weights)
            
#         else:
            
#             masks = []
#             for weights in self.model.get_weights():
                
#                 layer_mask = np.ones(weights.shape)
#                 masks.append(layer_mask)
                
               
#         self.model.masks = masks
            

        
        
        