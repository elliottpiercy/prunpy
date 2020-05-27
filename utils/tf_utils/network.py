import numpy as np
import tensorflow as tf
import utils.tf_utils.saver
import utils.prune_utils.initalise
import utils.tf_utils.build_network
import utils.tf_utils.fully_connected

from copy import deepcopy



class model():
    
    def __init__(self, network_config, pruning_config):
            
        
        if 0 in network_config['layer_shapes']:
            raise ValueError('Each layer must have at least 1 unit')
            
        if 'pretrained_path' in list(network_config.keys()):
            self.pretrained_path = network_config['pretrained_path']
        else: 
            self.pretrained_path = None
        

        self.loss = None
        self.network = None
        self.optimiser = None
        self.network_config = network_config
        
        
        self.pruning_schedule = utils.prune_utils.initalise.scheduler(pruning_config)
        self.initialise_model()
        self.compile_model()
        
            
        
    # Intialise layers
    def initialise_model(self):
        
        if self.network_config['network_type'] == 'fully_connected':
            self.network =  utils.tf_utils.fully_connected.model(self.network_config)
            
            
    # Load pretrained weights        
    def load_weights(self, pretrained_model):
        self.network.load_weights(pretrained_model)
        self.network.compile()
        
        
           
    # Compile model with optimiser and loss function    
    def compile_model(self):
        
        opt = utils.tf_utils.build_network.optimiser(self.network_config['optimiser']).get_optimiser()
        self.optimiser = opt
        
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#         loss_fn = utils.tf_utils.build_network.loss(self.network_config['loss']).get_loss_fn()
        self.loss = loss_fn
        
        self.network.compile(optimizer=opt,
                             loss=loss_fn,
                             metrics=['accuracy'])
        
        
    # Train model
    def train(self, train_dataset):
        
        filepath="tmp/model-{epoch:02d}.hdf5"
        saver_callback = utils.tf_utils.saver.save_callback(filepath, self.network_config['save_rate'])
        
        history = self.network.fit(train_dataset,
                                   epochs=self.network_config['epochs'],
#                                    batch_size = self.network_config['batch_size'],
                                   callbacks=[self.pruning_schedule, saver_callback])
        
        
        
    # Evalaute new data (test data)
    def evaluate(self, dataset):
        test_loss, test_acc = self.network.evaluate(dataset.test.x, dataset.test.y)
        return test_loss, test_acc
        
        
    # Predict output for new data (test_data)
    def predict(self, dataset):
        return self.network.predict(dataset.test.x)
        
    
        
        

