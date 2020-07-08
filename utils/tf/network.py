import numpy as np
import tensorflow as tf
import utils.prune.initalise
import utils.tf.saver
import utils.tf.build_network
import utils.tf.fully_connected

from copy import deepcopy

'''
Main network initialisation script. Training procedure is universal to all networks.
Models are pulled along with loss functions/activations etc to create and train nets
'''

# Base tensorflow model creation and training procedure
class model():
    
    def __init__(self, network_config, pruning_config):
            
        
        if 0 in network_config['layer_shapes']:
            raise ValueError('Each layer must have at least 1 unit')
            

        self.loss = None
        self.network = None
        self.optimiser = None
        self.network_config = network_config
        self.pruning_config = pruning_config
        
        
        self.pruning_schedule = utils.prune.initalise.scheduler(pruning_config)
        self.initialise_model()
        
        if 'pretrained_path' in list(network_config.keys()):
            self.pretrained_path = network_config['pretrained_path']
            self.load_weights(network_config['pretrained_path'])
            
        else: 
            self.pretrained_path = None
        
        
        self.compile_model()
            
        
    # Intialise layers
    def initialise_model(self):
        
        if self.network_config['network_type'] == 'fully_connected':
            self.network =  utils.tf.fully_connected.model(self.network_config)
            
            
    # Load pretrained weights        
    def load_weights(self, pretrained_model):
        self.network.load_weights(pretrained_model)
        self.network.compile()
        
        
           
    # Compile model with optimiser and loss function    
    def compile_model(self):
        
        opt = utils.tf.build_network.optimiser(self.network_config['optimiser']).get_optimiser()
        self.optimiser = opt
        
        loss_fn = utils.tf.build_network.loss(self.network_config['loss']).get_loss_fn()
        self.loss = loss_fn
        
        self.network.compile(optimizer=opt,
                             loss=loss_fn,
                             metrics=['accuracy'])
        
        
    # Train model
    def train(self, train_dataset):
        
        
        base_dir, model_dir, parameter_dir = utils.tf.saver.create_directory(self.network_config, self.pruning_config)
        
        model_saver_callback = utils.tf.saver.save_callback(model_dir, self.network_config['save_rate'])
        weight_saver_callback = utils.tf.saver.save_weights(parameter_dir,  self.network_config['save_rate'])
        bias_saver_callback = utils.tf.saver.save_biases(parameter_dir,  self.network_config['save_rate'])
        
        mask_saver_callback = utils.tf.saver.save_masks(parameter_dir,  
                                                        self.network_config['save_rate'], 
                                                        self.pruning_config['epoch_threshold'])
       
        
        history = self.network.fit(train_dataset,
                                   epochs=self.network_config['epochs'],
                                   callbacks=[self.pruning_schedule,
                                              model_saver_callback,
                                              weight_saver_callback,
                                              bias_saver_callback,
                                              mask_saver_callback])
        
        return history
        
        
        
    # Evalaute new data (test data)
    def evaluate(self, dataset):
        test_loss, test_acc = self.network.evaluate(dataset.test.x, dataset.test.y)
        return test_loss, test_acc
        
        
        
    # Predict output for new data (test_data)
    def predict(self, dataset):
        return self.network.predict(dataset.test.x)
        
    
        
        

