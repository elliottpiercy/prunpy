import os
import json
import numpy as np

'''
Visualisation helper functions
'''

# Return dictionary of masks for each epochs
def get_masks(log_path):
    
    # Get sorted list of mask patahs
    def get_masks_paths(masks_path):
        return np.sort([masks_path + path for path in os.listdir(masks_path) if path.split('-')[0] == 'mask'])
    
    mask_path = log_path + 'parameters/'
    mask_paths = get_masks_paths(mask_path)
    
    masks = {}
    for epoch, path in enumerate(mask_paths):
        with open(path, "r") as read_file:
            data = json.load(read_file)
            
        masks[epoch] = data
        
    return masks


# Validate network shapes. Raise error if there are too many units to render (omre than max nodes)
def _validate_shapes(masks, max_nodes):
    
    for idx, key in enumerate(masks[0]):
        
        # Ignore the input layer. Most likely over max_node limit
        if idx == 0:
            continue
            
        layer_mask = masks[0][key]
        if np.array(layer_mask).shape[0] > max_nodes:
            raise ValueError('Too many nodes to visualise. Pass \'validate_shapes=False\' to ignore this exception')


# Validate the log path exists and ends with '/'.
def _validate_log_path(path):
    
    if not os.path.isdir(path):
        raise ValueError('Invalid log path.')
        
    if path[-1] != '/':
        return path + '/'
    else:
        return path