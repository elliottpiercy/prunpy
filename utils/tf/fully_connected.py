import numpy as np
import tensorflow as tf


# Base fully connected model
def model(network_config):
    
    if 'seed' not in list(network_config.keys()):
        network_config['seed'] = np.random.randint(0, 99999)
            
    layers = [tf.keras.layers.Flatten(input_shape=network_config['input_shape'])]  
    
    for layer_units in network_config['layer_shapes']:
        layers.append(tf.keras.layers.Dense(layer_units, 
                                             activation=network_config['activation'],
                                             kernel_initializer = tf.keras.initializers.GlorotNormal(network_config['seed'])))
        
        if 'dropout_rate' in list(network_config.keys()):
            layers.append(tf.keras.layers.Dropout(network_config['dropout_rate']))
            
            
    layers.append(tf.keras.layers.Dense(network_config['n_classes'], 
                                             activation=network_config['activation'],
                                             kernel_initializer = tf.keras.initializers.GlorotNormal(network_config['seed'])))
    

    #     layers.append(tf.keras.layers.Softmax())
    model = tf.keras.models.Sequential(layers)
    return model

