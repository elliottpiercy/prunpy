import numpy as np
import tensorflow as tf

# Cast to correct datatypes
def preprocess(x, y):
      return tf.cast(x, tf.float32), tf.cast(y, tf.int64)


# Create tf dataset
def create_dataset(x, y, n_classes, dataset_config):
    
    if dataset_config['one_hot_encode']:
        y = tf.one_hot(y, depth=n_classes)
        
    if dataset_config['shuffle']:
        return tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess).shuffle(len(y)).batch(dataset_config['batch_size'])
    else:
        return tf.data.Dataset.from_tensor_slices((x, y)).map(preprocess).batch(dataset_config['batch_size'])
        


# Wrapper for multiple toy datasets pulled from tensorflow
def get_dataset(dataset_config):
    
    if dataset_config['dataset'] == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_config['dataset'] == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_config['dataset'] == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_config['dataset'] == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        
    # Normalise data
    if dataset_config['normalise']:
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        
    n_classes = len(np.unique(y_train))
    train_dataset = create_dataset(x_train, y_train, n_classes, dataset_config)
    test_dataset = create_dataset(x_test, y_test, n_classes, dataset_config)
        
    return train_dataset, test_dataset


if __name__ == "__main__":

    train_dataset, test_dataset = get_dataset('mnist', True, True, True)
