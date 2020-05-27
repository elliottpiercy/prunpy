import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder



# Wrapper for multiple toy datasets pulled from tensorflow
def get_dataset(dataset, normalise = True, one_hot_encode = True):
    
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
    # Normalise data
    if normalise:
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
    # One hot encode labels
    if one_hot_encode:
        enc = OneHotEncoder()
        enc.fit(y_train.reshape((-1, 1)))
        y_train = enc.transform(y_train.reshape((-1, 1))).toarray()
        y_test = enc.transform(y_test.reshape((-1, 1))).toarray()
        
        
    return (x_train, y_train), (x_test, y_test)

