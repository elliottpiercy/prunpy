import numpy as np
import tensorflow as tf

# Write weights to file
def save_weights(weights, epoch):

    for weight_ID in weights.keys():
        save_path = 'parameters/weights/' + weight_ID + '_' + str(epoch) + '.txt'
        np.savetxt(save_path, weights[weight_ID].eval())


# Write biases to file
def save_biases(biases, epoch):

    for bias_ID in biases.keys():
        save_path = 'parameters/biases/' + bias_ID + '_' + str(epoch) + '.txt'
        np.savetxt(save_path, biases[bias_ID].eval())


# Write model to file
def save_callback(filepath, save_frequency):
    
    print(save_frequency)
    save_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                                        save_best_only=False, save_weights_only=False,
                                                        save_frequency=save_frequency)
    return save_callback

    
# Restore model
def load_model(filepath):
    return tf.keras.models.load_model(filepath, custom_objects=None, compile=True)


