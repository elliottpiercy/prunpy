import numpy as np


"""
Magnitude weight pruning schedule - Mask all weights that are within the threshold.

EG: threshold = 0.1. ALl weights between -0.1 and 0.1 are masked with 0s

"""


class schedule():
    
    
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
        self.threshold = pruning_config['threshold']
    
    
    # Update mask for weight vector based on new weights
    def update_mask(self, weights):
        
        # Return locations where value is between bounds
        def abs_thresholding(data):
            return np.where(np.abs(data) < self.pruning_config['threshold'])
#             return np.where(np.logical_and(data >= -self.pruning_config['threshold'], data <= self.pruning_config['threshold']))
        
        
        masks = {}
        for ID in weights.keys():
            
            layer_weights = weights[ID].eval()
            locations = abs_thresholding(layer_weights)
            mask = np.ones(layer_weights.shape)
            mask[locations] = 0
            masks[ID] = mask
            
        return masks
    
    
    # Apply magnitude pruning schedule
    def apply(self, model, sess, epoch):
        
        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch >= self.pruning_config['epoch_threshold']:
            masks = self.update_mask(model.weights)

            for ID in masks.keys():
                new_weights = model.masks[ID] * model.weights[ID]
                sess.run(model.weights[ID].assign(new_weights))
                
            return masks
        
        else:
            return model.masks

    
    
        
        
        
        