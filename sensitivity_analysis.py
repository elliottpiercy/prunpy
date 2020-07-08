import utils.tf.network
import utils.dataset.examples
import matplotlib.pyplot as plt


dataset_config = {'dataset': 'mnist',
                  'batch_size': 128,
                  'one_hot_encode': True,
                  'shuffle': True,
                  'normalise': True}


pruning_config = {'schedule_type': 'magnitude_percentage',
                  'threshold': None,
                  'epoch_threshold': 0}



optimiser = {'name': 'RMSProp',
             'learning_rate': 0.001
            }


loss = {'name':'CategoricalCrossentropy',
        'from_logits': True
       }


network_config = {'network_type': 'fully_connected',
                  'input_shape': (28, 28),
                  'seed': 5,
                  'layer_shapes': [128],
                  'n_classes': 10,
                  'activation': 'relu',
                  'epochs': 10,
                  'dropout_rate': 0.2,
                  'save_rate': 'epoch',
                  'batch_size': 128,
                  'pretrained_path': 'pretrained/mnist.hdf5', 
                  'optimiser': optimiser,
                  'loss': loss
                 }




# Simple sensitivity analysis
def analysis(sparsity_list):
    
    
    results = {}
    results['sparsity'] = []
    results['accuracy']  = []
    for sparsity in sparsity_list:
        
        pruning_config['threshold'] = sparsity

        train_dataset, test_dataset = utils.dataset.examples.get_dataset(dataset_config)

        network = utils.tf.network.model(network_config, pruning_config)
        history = network.train(train_dataset)
        
        results['sparsity'].append(sparsity)
        results['accuracy'].append(np.max(history.history['accuracy']))
        
    return results
    

    
results = analysis([0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9])

plt.plot(results['sparsity'], results['accuracy'])
plt.show()
