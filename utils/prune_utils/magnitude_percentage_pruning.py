import numpy as np


"""
Magnitude percentage weight pruning schedule - Mask all weights that are within the percentage threshold.

EG: threshold = 10%. All weights in the middle 10% (approximately -5% to -10%) are prunned. Example assume gaussian distributed weights.

"""


class schedule():
    
    
    def __init__(self, pruning_config):
        self.pruning_config = pruning_config
    
    
    # Update mask for weight vector based on new weights
    def update_mask(self, weights):
        
        # Return locations where value is between bounds
        def abs_percentage_threshold(data):
            order = np.argsort(np.abs(data))[:int(len(data) * self.pruning_config['threshold'])]
            return order
        
        
        masks = {}
        for ID in weights.keys():
            
            layer_weights = weights[ID].eval()
            locations = abs_percentage_threshold(layer_weights)
            mask = np.ones(layer_weights.shape)
            mask[locations] = 0
            masks[ID] = mask
            
        return masks
    
    
    # Apply magnitude pruning schedule
    def apply(self, model, sess, epoch):
        
        # If the threshold is 0 -- ignore and dont prune
        if self.pruning_config['threshold'] != 0 and epoch > self.pruning_config['epoch_threshold']:
            masks = self.update_mask(model.weights)

            for ID in masks.keys():
                new_weights = model.masks[ID] * model.weights[ID]
                sess.run(model.weights[ID].assign(new_weights))
                
            return masks
        
        else:
            return model.masks

    
    
        
        
        
        