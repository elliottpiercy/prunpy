import numpy as np
import tensorflow as tf


# Base cnn
def model(network_config):
    
    if 'seed' not in list(network_config.keys()):
        network_config['seed'] = np.random.randint(0, 99999)
            
    layers = []

    for layer_ID, filter_size in enumerate(network_config['filters']):
        
        
        layers.append(tf.keras.layers.Conv2D(filter_size, (3, 3), activation=network_config['conv_activation'], 
                                             input_shape=network_config['input_shape'],
                                             name='conv_%s'%(str(layer_ID))))
    
    
        if network_config['pool_type'] == 'max':
            layers.append(tf.keras.layers.MaxPooling2D(network_config['pool_shape'],
                                                       name='maxpool_%s'%(str(layer_ID))))
            
        elif network_config['pool_type'] == 'mean':
            layers.append(tf.keras.layers.AveragePooling2D(network_config['pool_shape'],
                                                           name='meanpool_%s'%(str(layer_ID))))
            
            
    layers.append(tf.keras.layers.Flatten())
            
    for layer_ID, fc_layer in enumerate(network_config['fc_layer_shapes']):
        
        layers.append(tf.keras.layers.Dense(fc_layer, 
                                            activation=network_config['fc_activation'],
                                            kernel_initializer = tf.keras.initializers.GlorotNormal(network_config['seed']),
                                            name='dense_%s'%(str(layer_ID))))
        
    layers.append(tf.keras.layers.Dense(network_config['n_classes'], 
                                        activation=network_config['fc_activation'],
                                        kernel_initializer = tf.keras.initializers.GlorotNormal(network_config['seed']),
                                        name='output'))
        
    model = tf.keras.models.Sequential(layers)
    
    return model



if __name__ == "__main__":

    network_config = {'network_type': 'cnn',
                      'input_shape': (28, 28, 1),

                      'filters': [32, 32],
                      'conv_activation': 'relu',

                      'fc_layer_shapes': [32, 32],
                      'fc_activation': 'relu',

                      'n_classes': 10,

                      'pool_type': 'mean', 
                      'pool_shape': (3, 3),
                      'activation': 'relu'
                 }

    cnn_model = model(network_config)

