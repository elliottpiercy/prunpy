import glob
import numpy as np
import utils.visualiser.helper

from PIL import Image
from graphviz import Digraph
        

'''
Main network visualisation functions.
'''
    
# Create images and render gif using internal render function
def render_gif(log_path, validate_shapes=True):
    
    
    # Render gif and save to file        
    def render(log_path):

        save_path = log_path + 'images/'

        fp_in = save_path + "network/graph_*.png"
        fp_out =  save_path + "network.gif"

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0) 
        
        print('Gif rendered. Please see ' + fp_out + ' for the network gif.')
        

    
    log_path = utils.visualiser.helper._validate_log_path(log_path)
    
    masks = utils.visualiser.helper.get_masks(log_path)
    save_path = log_path + 'images/network/'


    if validate_shapes:
        utils.visualiser.helper._validate_shapes(masks, max_nodes = 64)

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

    # Finally render network gif
    render(log_path)
  
    