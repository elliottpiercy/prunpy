from copy import deepcopy
import utils.tf_utils.fully_connected


def initialise(network_config, pruning_config):
    
    if network_config['network_type'] == 'fully_connected':
        return utils.tf_utils.fully_connected.model(deepcopy(network_config), deepcopy(pruning_config))
                      