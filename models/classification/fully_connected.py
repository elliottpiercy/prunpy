
class fully_connected():
    
    def __init__(self, input_shape, layer_shapes, n_classes, optimiser, threshold, activation):
        self.input_shape = input_shape
        self.layer_shapes = layer_shapes
        self.n_classes = n_classes
        self.optimiser = optimiser
        self.thresold = threshold
        self.activation = activation
        
        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.network = None
        self.x_placeholder = None
        self.y_placeholder = None
        self.num_examples = None
        
        self.seed = 1
        
        
    def initialise_weights(self):
        self.layer_shapes.insert(0, self.input_shape)
        
        for idx, layer in enumerate(self.layer_shapes[:-1]):
            
            weight_ID = 'w' + str(idx)
            bias_ID = 'b' + str(idx)

            self.weights[weight_ID] = tf.Variable(tf.random_normal([layer, self.layer_shapes[idx+1]], seed = self.seed))
            self.biases[bias_ID] = tf.Variable(tf.random_normal([self.layer_shapes[idx+1]], seed = self.seed))
                
                
        # Final layer
        weight_ID = 'w' + str(len(self.layer_shapes)-1)
        bias_ID = 'b' + str(len(self.layer_shapes)-1)

        self.weights[weight_ID] = tf.Variable(tf.random_normal([self.layer_shapes[-1], self.n_classes], seed = self.seed))
        self.biases[bias_ID] = tf.Variable(tf.random_normal([self.n_classes], seed = self.seed))
        
            
            
    def initialise_placeholders(self):
        self.x_placeholder = tf.placeholder("float", [None, self.input_shape]) 
        self.y_placeholder = tf.placeholder("float", [None, self.n_classes]) 
        
        
        
    def initialise(self):
        
        self.initialise_placeholders()
        self.initialise_weights()
    
        previous_layer = self.x_placeholder
        for idx, (weight_ID, bias_ID) in enumerate(zip(list(self.weights.keys()), list(self.biases.keys()))):
            layer_ID = 'layer_' + str(idx)
            
            layer = tf.add(tf.matmul(previous_layer, self.weights[weight_ID]), self.biases[bias_ID])
            layer = tf.nn.relu(layer)
            
            self.layers[layer_ID] = layer
            previous_layer = layer
            
        return previous_layer
    
    
    def save_weights(self, epoch):
        
        for weights in self.weights.keys():
            save_path = 'parameters/weights/' + weights + '_' + str(epoch) + '.txt'
            np.savetxt(save_path, self.weights[weights].eval())
        
    def save_biases(self, epoch):
        
        for bias in self.biases.keys():
            save_path = 'parameters/biases/' + bias + '_' + str(epoch) + '.txt'
            np.savetxt(save_path, self.biases[bias].eval())
                    
      
    

    # Update weight vector by sparsifying it
    def get_mask(self, lbound, ubound):
        
        # Return locations where value is between bounds
        def between(data, lbound, ubound):
            return np.where(np.logical_and(data >= lbound, data <= ubound))[0]
        
        masks = {}
        for weight in self.weights.keys():
            
            
            weights = self.weights[weight].eval()
            locations = between(weights, lbound, ubound)
            mask = np.ones(weights.shape)
            mask[locations] = False
            masks[weight] = mask
            
        return masks
#             self.weights[weight] = tf.boolean_mask(self.weights[weight], mask, axis=None, name='boolean_mask')
        
    
        
        
    def train(self, train_data, test_data, learning_rate, epochs, batch_size):
        x_train, y_train = train_data
        x_test, y_test = test_data
        self.num_examples = x_train.shape[0]
        
        network = self.initialise()
        self.network = network
        
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=network, labels=self.y_placeholder))
        
#         # Define loss and optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
#         # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(epochs):
                
                
                self.save_weights(epoch)
                self.save_biases(epoch)
                    
#                 self.update_weights(lbound=-self.threshold, ubound=self.threshold)
                
                avg_cost = 0
                max_idx = int(self.num_examples/ batch_size) * batch_size
                for batch in range(0, len(x_train[:max_idx]), batch_size):
                    
             
                    
                    batch_x = x_train[batch:batch + batch_size].reshape((-1, self.input_shape))
                    batch_y = y_train[batch:batch + batch_size].reshape((batch_size, self.n_classes))
                    
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.x_placeholder: batch_x,
                                                                    self.y_placeholder: batch_y})
                    
                    
#                     # Compute average loss
                    avg_cost += c /  int(x_train.shape[0]/ batch_size)
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        

if __name__ == "__main__":

    fc = fully_connected(784, [10, 10, 10], 10, 'RMSProp', 0.5, 'relu')
    fc.train((x_train, y_train), (x_test, y_test), 0.001, 10, 128)