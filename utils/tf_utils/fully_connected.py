import numpy as np
import tensorflow as tf
import utils.tf_utils.saver
import utils.prune_utils.initalise


class model():
    
    def __init__(self, network_config, pruning_config):
            
        
        if 0 in network_config['layer_shapes']:
            raise ValueError('Each layer must have at least 1 unit')
            
        self.input_shape = network_config['input_shape']
        self.layer_shapes = network_config['layer_shapes']
        self.n_classes = network_config['n_classes']
        self.optimiser = network_config['optimiser']
        self.activation = network_config['activation']
        
        
        if 'pretrained_path' in list(network_config.keys()):
            self.pretrained_path = network_config['pretrained_path']
        else: 
            self.pretrained_path = None
        
        
        self.weights = {}
        self.masks = {}
        self.biases = {}
        self.layers = {}
        
        
        self.network = None
        self.x_placeholder = None
        self.y_placeholder = None
        self.num_examples = None
        
        
        self.seed = 1
        self.pruning_schedule = utils.prune_utils.initalise.scheduler(pruning_config)
        
        
        
    # Intialise weights and biases with constant seed
    def initialise_weights(self):
        self.layer_shapes.insert(0, self.input_shape)
        
        for idx, layer in enumerate(self.layer_shapes[:-1]):
            
            weight_ID = 'w' + str(idx)
            bias_ID = 'b' + str(idx)

            self.weights[weight_ID] = tf.Variable(tf.random_normal([layer, self.layer_shapes[idx+1]], seed = self.seed))
            self.biases[bias_ID] = tf.Variable(tf.random_normal([self.layer_shapes[idx+1]], seed = self.seed))
            self.masks[weight_ID] = tf.ones([layer, self.layer_shapes[idx+1]])

            
        weight_ID = 'w' + str(len(self.layer_shapes)-1)
        bias_ID = 'b' + str(len(self.layer_shapes)-1)

        self.weights[weight_ID] = tf.Variable(tf.random_normal([self.layer_shapes[-1], self.n_classes], seed = self.seed))
        self.biases[bias_ID] = tf.Variable(tf.random_normal([self.n_classes], seed = self.seed))
        self.masks[weight_ID] = tf.ones([self.layer_shapes[-1], self.n_classes])
        
        
            
    # Intialise placeholders
    def initialise_placeholders(self):
        self.x_placeholder = tf.placeholder("float", [None, self.input_shape]) 
        self.y_placeholder = tf.placeholder("float", [None, self.n_classes]) 
        
        
        
    # Intialise layers
    def initialise(self):
        
        tf.reset_default_graph()
        
        self.initialise_placeholders()
        self.initialise_weights()
    
        previous_layer = self.x_placeholder
        for idx, (weight_ID, bias_ID) in enumerate(zip(list(self.weights.keys()), list(self.biases.keys()))):
            
            layer_ID = 'layer_' + str(idx)
            with tf.variable_scope(layer_ID): 
            
                
                masked_weights = self.weights[weight_ID] *  self.masks[weight_ID]
                layer = tf.add(tf.matmul(previous_layer, masked_weights), self.biases[bias_ID])
                layer = tf.nn.relu(layer)

                self.layers[layer_ID] = layer
                previous_layer = layer
            
        return previous_layer

           
        
    def train(self, dataset, learning_rate, epochs, batch_size, save_rate):
        
        self.num_examples = dataset.train.num_examples
        
        network = self.initialise()
        self.network = network
        
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=network, labels=self.y_placeholder))

#         # Define loss and optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
#         # Initializing the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(init)
            
            
            if self.pretrained_path != None:
#                 saver.restore(, path)
                utils.tf_utils.saver.load_model(saver, sess, self.pretrained_path)

            # Training cycle
            for epoch in range(epochs):
                
                dataset.train._reset_batch_count()
                
                utils.tf_utils.saver.save_weights(self.weights, epoch)
                utils.tf_utils.saver.save_biases(self.biases, epoch)
                    
                if epoch% save_rate == 0:
                    utils.tf_utils.saver.save_model(saver, sess, epoch)
                
                avg_cost = 0
                self.masks = self.pruning_schedule.apply(self, sess, epoch)
                
                
                for batch in range(dataset.train.get_number_of_batches(batch_size)):
                    
                    batch_x, batch_y = dataset.train.next_batch(batch_size)
                    
                    batch_x = batch_x.reshape((-1, self.input_shape))
                    batch_y = batch_y.reshape((batch_size, self.n_classes))
                    
                    
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.x_placeholder: batch_x,
                                                                    self.y_placeholder: batch_y})
                    

#                     # Compute average loss
                    avg_cost += c /  int(self.num_examples/ batch_size)
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        


