import os
import json
import glob
import numpy as np
from PIL import Image


'''
Visualisation helper functions
'''


# Return dictionary of masks for each epochs
def _get_weights(log_path):
    
    # Get sorted list of mask patahs
    def get_masks_paths(weights_path):
        return np.sort([weights_path + path for path in os.listdir(weights_path) if path.split('-')[0] == 'weights'])
    
    weights_path = log_path + 'parameters/'
    weight_paths = get_masks_paths(weights_path)
    
    weights = {}
    for epoch, path in enumerate(weight_paths):
        with open(path, "r") as read_file:
            data = json.load(read_file)
            
        weights[epoch] = data
        
    return weights


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


# Render gif and save to file        
def _render(log_path, fp_in_extension, fp_out_extension):

    # File in/out paths
    fp_in = log_path + fp_in_extension
    fp_out = log_path + fp_out_extension

    # Load images for gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0) 

    print('Gif rendered. Please see ' + fp_out + ' for the gif.')
  

# Return max and min weights. eps=value added to the max/min bounds (for visualisation)
def _get_weight_bounds(weights, eps = 0.1):

    max_weight = 0
    min_weight = 0
    for epoch in weights:
        for layer in weights[epoch]:

            if np.max(weights[epoch][layer]) > max_weight:
                max_weight = np.max(weights[epoch][layer])

            if np.min(weights[epoch][layer]) < min_weight:
                min_weight = np.min(weights[epoch][layer])

    return min_weight-eps, max_weight+eps


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