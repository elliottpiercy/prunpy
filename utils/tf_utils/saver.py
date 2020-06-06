import json
import numpy as np
import tensorflow as tf


class save_weights(tf.keras.callbacks.Callback):
    
    def __init__(self, base_save_path):
        
        if base_save_path[-1] != '/':
            base_save_path += '/'
            
        self.base_save_path = base_save_path
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        save_path = self.base_save_path + 'weights-{epoch:02d}.json'
        
        with open(save_path, 'w') as outfile:
            json.dump(self.model.weights, outfile)

            
class save_biases(tf.keras.callbacks.Callback):
    
    def __init__(self, base_save_path):
        
        if base_save_path[-1] != '/':
            base_save_path += '/'
            
        self.base_save_path = base_save_path
    
    
    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError('Save biases callback is not yet implemented')

        



# Write model to file
def save_callback(filepath, save_frequency):
    
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                                        save_best_only=False, save_weights_only=False,
                                                        save_frequency=save_frequency)
    return save_callback


    
# Restore model
def load_model(filepath):
    return tf.keras.models.load_model(filepath, custom_objects=None, compile=True)


