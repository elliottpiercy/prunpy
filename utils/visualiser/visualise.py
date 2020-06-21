import os
import numpy as np
import seaborn as sns
import utils.visualiser.helper
import matplotlib.pyplot as plt

from graphviz import Digraph
        

'''
Main network visualisation functions.
'''
    
# Create images and render network gif showing network connections while training
def render_network_gif(log_path, validate_shapes=True):
    
    
    log_path = utils.visualiser.helper._validate_log_path(log_path)
    
    masks = utils.visualiser.helper.get_masks(log_path)
    save_path = log_path + 'images/network/'


    # Check there arent too many nodes to render (max nodes for a layer =  64)
    if validate_shapes:
        utils.visualiser.helper._validate_shapes(masks, max_nodes = 64)

    for epoch in masks:

        filename = save_path + str(epoch).zfill(4)


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
    fp_in_extension = 'images/network/*.png'
    fp_out_extension = 'images/network.gif'
    utils.visualiser.helper._render(log_path, fp_in_extension, fp_out_extension)
  
    
# Create images and render distribution gif showing weight distribution for specific layer while training
def render_distribution_gif(log_path, layer_to_vis):
    
    weights = utils.visualiser.helper._get_weights(log_path)    
    
    save_path = log_path + 'images/distributions/' + 'layer_' + str(layer_to_vis) + '/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    for epoch_idx, epoch in enumerate(weights):

        layer_weights = weights[epoch][str(layer_to_vis)]
        filename = save_path + str(epoch).zfill(4) + '.png'
        title = 'Layer ' + str(layer_to_vis) + ' epoch ' + str(epoch_idx)

        # Visualise first plot and get bounds in exception
        try:

            sns_plot = sns.distplot(np.array(layer_weights).reshape(-1), color="b")
            plt.xlim(right=xlim[1]) #xmax is your value
            plt.xlim(left=xlim[0]) #xmin is your value
            plt.ylim(top=ylim[1]) #ymax is your value
            plt.ylim(bottom=ylim[0]) #ylim is your value

        except:
            sns_plot = sns.distplot(np.array(layer_weights).reshape(-1))
            xlim = sns_plot.get_xlim()
            ylim = sns_plot.get_ylim()

        
        plt.xlabel('Weights')
        plt.ylabel('Density')
        plt.title(title)
        plt.savefig(filename)
        plt.close()
    
    fp_in_extension = 'images/distributions/layer_' + str(layer_to_vis) + '/*.png'
    fp_out_extension = 'images/layer_' + str(layer_to_vis) + '.gif'
    utils.visualiser.helper._render(log_path, fp_in_extension, fp_out_extension)