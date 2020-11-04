import tensorflow as tf


# Optimiser class
class optimiser():
    
    def __init__(self, config):
        self.config = config

    # Print an example list of parameters for an optimiser
    def get_optimiser_parameters(self):
        print(tf.keras.optimizers.get(self.config['name']).get_config())


    def get_optimiser(self):

        opt = tf.keras.optimizers.get(self.config['name'])
        opt = opt.from_config(self.config)
        return opt



# Loss class
class loss():
    
    def __init__(self, config):
        self.config = config
        
    
    def get_custom_loss(self):
        raise NotImplementedError('Custom loss function is not currently implemented or the tf loss function does not exist. Insert your custom loss function here if you wish.')
        
    
     # Print an example list of parameters for an optimiser
    def get_loss_parameters(self):
        print(tf.keras.losses.get(self.config['name']).get_config())
        
    
    # Try to get tf loss function
    def get_loss_fn(self):
        
        try:
            return tf.keras.losses.get(self.config['name']).from_config(self.config)
        except:
            return self.get_custom_loss()
            
           
    



    
    
    