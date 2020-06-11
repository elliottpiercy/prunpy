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

            layer_dict = {}
            for layer_idx, layer in enumerate(self.model.get_weights()):
                layer_dict[layer_idx] = layer.tolist()
                
                
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
        raise NotImplementedError('Save biases callback is not yet implemented')

        

        
# Save masks tf callback            
class save_masks(tf.keras.callbacks.Callback):
    
    def __init__(self, save_path, save_rate, epoch_threshold):
        
        if save_rate == 'epoch':
            save_rate = 1
            
        self.save_path = save_path
        self.save_rate = save_rate
        self.epoch_threshold = epoch_threshold
        
    
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.save_rate == 0:

            save_path = self.save_path + 'mask-' + str(epoch).zfill(4) + '.json'
            if epoch < self.epoch_threshold:
            
                mask_dict = {'masks': None}
                with open(save_path, 'w') as outfile:
                    json.dump(mask_dict, outfile)
            
            else:
                
                layer_dict = {}
                for layer_idx, layer in enumerate(self.model.masks):
                    layer_dict[layer_idx] = layer.tolist()

                with open(save_path, 'w') as outfile:
                    json.dump(layer_dict, outfile)

            
            
            
            

                

# Write model to file
def save_callback(filepath, save_frequency):
    
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
    
    os.mkdir(model_dir)
    os.mkdir(parameter_dir)
    
    
    print('Output saved to directory: ' + base_dir)
    
    return base_dir, model_dir, parameter_dir


