import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split


class convert():
    
    def __init__(self, config):
        
        if config['validation'] == None:
            config['validation'] = 0
       
        if config['train'] + config['validation'] + config['test'] != 1:
            raise ValueError("Data split must sum to 1.")
            
            
        self.data = config['x']
        self.labels = config['y']
        
        self.train = self.train(config['x'][:int(len(config['x'])*config['train'])], 
                                config['y'][:int(len(config['y'])*config['train'])])
        
        
        if config['validation'] == 0:
            self.validation = None
        else: 
            self.validation = self.validation(config['x'][int(len(config['x'])*config['train']):int(len(config['x'])*config['train'])+int(len(config['x'])*config['validation'])],
                                              config['y'][int(len(config['y'])*config['train']):int(len(config['y'])*config['train'])+int(len(config['y'])*config['validation'])])

        
        
        self.test = self.test(config['x'][int(len(config['x'])*config['train'])+int(len(config['x'])*config['validation']):],
                              config['y'][int(len(config['y'])*config['train'])+int(len(config['y'])*config['validation']):])
        
        
    
    # Class for training set variables batch method
    class train():
    
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.num_examples = x.shape[0]
            self.batch_count = 0


        def _reset_batch_count(self):
            self.batch_count = 0
            
            
        def next_batch(self, batch_size):
            
            x_next_batch = self.x[self.batch_count:self.batch_count + batch_size]
            y_next_batch = self.y[self.batch_count:self.batch_count + batch_size]
            self.batch_count += batch_size
            return x_next_batch, y_next_batch
        
        
        def get_number_of_batches(self, batch_size):
            return int(self.num_examples/batch_size)
        
        
        
    # Class for validation set variables
    class validation():
    
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.num_examples = x.shape[0]

        
    # Class for test set variables 
    class test():
            
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.num_examples = x.shape[0]

            
if __name__ == "__main__":
    
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0



    config = {'x': x_train,
             'y': y_train,
             'train': 0.1,
             'validation': None,
             'test': 0.9}


    dset = convert(config)