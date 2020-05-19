import numpy as np

# Magnitude weight pruning schedule
class schedule():
    
    
    def __init__(self, threshold):
        self.threshold = threshold
    
    
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
    
    
        
        
        
        