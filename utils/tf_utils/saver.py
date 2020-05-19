import numpy as np


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
def save_model(saver, sess, epoch):
    save_path = saver.save(sess, "./tmp/model.ckpt", global_step=epoch)


    
# Restore model
def load_model(saver, path):
    return saver.restore(saver, path)