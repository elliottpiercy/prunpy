import os
import cv2
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
def _get_masks(log_path):
    
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


# Return network loss and accuracy (model.history.history)
def _get_history(log_path):
    
    log_path += 'history.json'
    with open(log_path, "r") as read_file:
            history = json.load(read_file)
    return history
    
    
# Return dictionary of sparsity value for each epochs
def _get_sparsity(log_path):
    
    # Get sorted list of mask patahs
    def get_sparsity_paths(sparsity_paths):
        return np.sort([sparsity_paths + path for path in os.listdir(sparsity_paths) if path.split('-')[0] == 'sparsity'])
    
    sparsity_path = log_path + 'parameters/'
    sparsity_paths = get_sparsity_paths(sparsity_path)
    
    sparsity = {}
    for epoch, path in enumerate(sparsity_paths):
        with open(path, "r") as read_file:
            data = json.load(read_file)
            
        sparsity[epoch] = data
        
    return sparsity


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
    
    
    

# Give each rendered image an epoch counter 
def _label_image(path, epoch, history, sparsity):
    
    loss = np.round(history['loss'][epoch], 3)
    accuracy = np.round(history['accuracy'][epoch], 3)
    sparsity = np.round(sparsity[epoch]['sparsity'], 3)

    image = cv2.imread(path)
    labels = ['Epoch: ' + str(epoch),
              'Loss: ' + str(loss),
              'Accuracy: ' + str(accuracy),
              'Sparsity: ' + str(sparsity)]
    
    dy = 50
    for i, line in enumerate(labels):
        
        x_position = image.shape[1]-300# 
        y_position =image.shape[0]-200 + (dy * i)
        position = (x_position, y_position)

        img = cv2.putText(
             image, #numpy array on which text is written
             line, #text
             position, #position at which writing has to start
             cv2.FONT_HERSHEY_SIMPLEX, #font family
             1, #font size
             (0, 0, 0, 0), #font color
             3) #font stroke
    
    cv2.imwrite(path, image)