import os
import json
import numpy as np
import tensorflow as tf

from datetime import datetime


# Save weights tf callback
class save_weights(tf.keras.callbacks.Callback):
    
    def __init__(self, save_path, save_rate):
        
        if save_rate == 'epoch':
            save_rate = 1
            
        self.save_path = save_path
        self.save_rate = save_rate
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        
        if epoch % self.save_rate == 0:
            
            layer_idx = 0
            layer_dict = {}
            for layer in self.model.get_weights():
                
                if len(layer.shape) == 2:
                    layer_dict[layer_idx] = layer.tolist()
                    layer_idx += 1
                
                
            save_path = self.save_path + 'weights-' + str(epoch).zfill(4) + '.json'
            with open(save_path, 'w') as outfile:
                json.dump(layer_dict, outfile)

                
            
# Save biases tf callback            
class save_biases(tf.keras.callbacks.Callback):
    
    def __init__(self, save_path, save_rate):
        
        if save_rate == 'epoch':
            save_rate = 1
            
        self.save_path = save_path
        self.save_rate = save_rate
        
    
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.save_rate == 0:
        
            layer_idx = 0
            layer_dict = {}
            for layer in self.model.get_weights():
                
                if len(layer.shape) == 1:
                    layer_dict[layer_idx] = layer.tolist()
                    layer_idx += 1
                
                
            save_path = self.save_path + 'biases-' + str(epoch).zfill(4) + '.json'
            with open(save_path, 'w') as outfile:
                json.dump(layer_dict, outfile)

        

        
# Save masks tf callback            
class save_masks(tf.keras.callbacks.Callback):
    
    def __init__(self, save_path, save_rate):
        
        if save_rate == 'epoch':
            save_rate = 1
            
        self.save_path = save_path
        self.save_rate = save_rate
        
        
    
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.save_rate == 0:
            
            print('Saving: ' + str(epoch))
            
            layer_dict = {}
            for layer_idx, layer in enumerate(self.model.masks):
                layer_dict[layer_idx] = layer.tolist()

            save_path = self.save_path + 'mask-' + str(epoch).zfill(4) + '.json'
            with open(save_path, 'w') as outfile:
                json.dump(layer_dict, outfile)
            
            
            
            
# Save pruning sparsity
class save_sparsity(tf.keras.callbacks.Callback):
    
    
    def __init__(self, save_path, save_rate):
        
        if save_rate == 'epoch':
            save_rate = 1
            
        self.save_path = save_path
        self.save_rate = save_rate
        
        
    def on_epoch_end(self, epoch, logs=None):
        
            save_path = self.save_path + 'sparsity-' + str(epoch).zfill(4) + '.json'
            with open(save_path, 'w') as outfile:
                json.dump({'sparsity': self.model.sparsity}, outfile)        
            
            
            
            
# Save model loss and accuracy to file. The history json gets overwritten each epoch
class save_history(tf.keras.callbacks.Callback):
    
    
    def __init__(self, save_path):
        self.save_path = save_path
        
        
    def on_epoch_begin(self, epoch, logs=None):
        
            save_path = self.save_path + 'history.json'
            with open(save_path, 'w') as outfile:
                json.dump(self.model.history.history, outfile)

    
    

# Write model to file
def save_model(filepath, save_frequency):
    
    filepath += 'model-{epoch:04d}.hdf5'
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                                        save_best_only=False, save_weights_only=False,
                                                        save_frequency=save_frequency)
    
    return save_callback



# Restore model
def load_model(filepath):
    return tf.keras.models.load_model(filepath, custom_objects=None, compile=True)



# Create directory structure for saving logs/parameters/models
def create_directory(network_config, pruning_config):
    
    base_dir = 'logs/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '/'
    os.mkdir(base_dir)
    
    with open(base_dir + 'network_config.json', 'w') as outfile:
        json.dump(network_config, outfile)
        
    with open(base_dir + 'pruning_config.json', 'w') as outfile:
        json.dump(pruning_config, outfile)

    model_dir = base_dir + 'models/'
    parameter_dir = base_dir + 'parameters/'
    image_dir = base_dir + 'images/'
    network_image_dir = base_dir + 'images/network/'
    distribution_image_dir = base_dir + 'images/distributions/'
    
    
    os.mkdir(model_dir)
    os.mkdir(parameter_dir)
    os.mkdir(image_dir)
    os.mkdir(network_image_dir)
    os.mkdir(distribution_image_dir)
    
    
    print('Output saved to directory: ' + base_dir)
    return base_dir, model_dir, parameter_dir


