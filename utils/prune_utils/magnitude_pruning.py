import numpy as np


"""
Magnitude weight pruning schedule - Mask all weights that are within the threshold.

EG: threshold = 0.1. ALl weights between -0.1 and 0.1 are masked with 0s

"""


class schedule():
    
    
    def __init__(self, pruning_config):
        self.threshold = pruning_config['threshold']
    
    
    # Update mask for weight vector based on new weights
    def update_mask(self, weights, lbound, ubound):
        
        # Return locations where value is between bounds
        def between(data, lbound, ubound):
            return np.where(np.logical_and(data >= lbound, data <= ubound))
        
        
        masks = {}
        for ID in weights.keys():
            
            layer_weights = weights[ID].eval()
            locations = between(layer_weights, lbound, ubound)
            mask = np.ones(layer_weights.shape)
            mask[locations] = 0
            masks[ID] = mask
            
        return masks
    
    
    # Apply magnitude pruning schedule
    def apply(self, model, sess):
        
        # If the threshold is 0 -- ignore and dont prune
        if self.threshold != 0:
            masks = self.update_mask(model.weights, lbound=-self.threshold, ubound=self.threshold)

            for ID in masks.keys():
                new_weights = model.masks[ID] * model.weights[ID]
                sess.run(model.weights[ID].assign(new_weights))
                
            return masks
        
        else:
            return model.masks

    
    
        
        
        
        