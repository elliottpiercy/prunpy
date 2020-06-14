import os
import json
import glob
import numpy as np
import seaborn as sns

from PIL import Image
from graphviz import Digraph


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


def _validate_shapes(masks, max_nodes):
    raise NotImplementedError('Not yet implemented')


def visualise(masks, save_path, validate_shapes=False):

    if validate_shapes:
        _validate_shapes(masks, max_nodes = 64)

    for epoch in masks:

        filename = save_path + 'graph_' + str(epoch)


        n = 0
        epoch_nodes = []
        epoch_mask = masks[epoch]
        g = Digraph('g', filename=filename, format='png')

        for layer_idx in epoch_mask:
            layer_mask = np.array(epoch_mask[layer_idx])

            # If the first layer (input layer), IGNORE. Too many nodes to visualise so show the hidden/output layers
            if layer_idx == '0':
                continue 

            if len(layer_mask.shape) < 2:# and int(layer_idx) != (len(epoch_mask)-1):
                epoch_nodes.append([])
                continue


            layer_nodes = []
            g.graph_attr.update(splines="false", nodesep='0.1', ranksep='5')

            with g.subgraph(name=str(layer_idx)) as c:

                    the_label = 'layer_' + str(layer_idx)
                    if layer_mask.shape[0] > max_nodes:
    #                     continue
                        raise ValueError('Too many nodes for this visualiser')

                    for node in range(layer_mask.shape[1]):

                        layer_nodes.append(str(n))

                        c.node(str(n))
                        c.attr(label=the_label)
                        c.attr(color='white')
                        c.attr(rank='same')
                        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")
                        n+=1

                    epoch_nodes.append(layer_nodes)

                    # Start adding edges. Masked nodes have invisible connections
                    if len(epoch_nodes) >= 3:
                        for current_node_mask_idx, current_node in enumerate(epoch_nodes[len(epoch_nodes)-1]):
                            for previous_node_mask_idx, previous_node in enumerate(epoch_nodes[len(epoch_nodes)-3]):

                                if layer_mask[previous_node_mask_idx][current_node_mask_idx] == 0:
                                    g.edge(str(previous_node), str(current_node), style="invis")
                                else:
                                    g.edge(str(previous_node), str(current_node))


        g.render()

        
# Render gif and save to file        
def render_gif(image_path):

    fp_in = image_path + "graph_*.png"
    fp_out =  image_path + "image.gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)   
    
    
if __name__ == "__main__":
  
    
    log_path = 'logs/2020-06-14-21-42-45/'
    masks = get_masks(log_path)

    visualise(masks, 'images/')
    render_gif('images/')