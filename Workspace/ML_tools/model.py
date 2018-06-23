# Setup data processing dependencies
import time
import pickle
import sklearn
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn.model_selection import GridSearchCV

# Setup neural network model
def set_nn(input_features, hidden_layers=2, hidden_units=8, hidden_activation='relu', output_units=4, output_activation='softmax'):
    """
    Setup a neural network model using the Tensorflow Keras python library.

    Arguments:
    -----------
    input_features : integer, number of input features
    hidden_layers : integer, number of hidden layer(s) in the neural network
    hidden_units : integer, number of neural network nodes/units in the hidden layer(s)
    hidden_activation : selection of 'relu', 'tanh' etc, type of activation function for hidden layer(s) units
    output_units : integer, number of neural network nodes/units in the output layer
    output_activation : selection of 'relu, 'tanh' etc, type of activation function for the output layer
    """

    task_name = 'Neural network model setup'
    task_info = {}
    model_info = {}

    start_time = time.time()

    # Cleaning inputs
    hidden_activation = hidden_activation.lower()
    output_activation = output_activation.lower()

    # Build the neural network model
    nn_model = tf.keras.Sequential()
    nn_model.add(tf.keras.layers.Dense(hidden_units, activation=hidden_activation, input_shape=(input_features,)))
    if hidden_layers >= 2:
        for layer in range(hidden_layers - 1):
            nn_model.add(tf.keras.layers.Dense(hidden_units, activation=hidden_activation))
    nn_model.add(tf.keras.layers.Dense(output_units, activation=output_activation))

    end_time = time.time()
    time_taken = end_time - start_time

    print('Status: Neural network setup successful!\n')
    print('Model summary:')
    print(nn_model.summary())

    # Store the task information
    task_info = {
        'task_name': task_name,
        'input_features': input_features,
        'hidden_layers': hidden_layers,
        'hidden_units': hidden_units,
        'hidden_activation': hidden_activation,
        'output_units': output_units,
        'output_activation': output_activation,
        'time_taken': time_taken
    }

    model_info = {
        'input_features': input_features,
        'hidden_layers': hidden_layers,
        'hidden_units': hidden_units,
        'hidden_activation': hidden_activation,
        'output_units': output_units,
        'output_activation': output_activation
    }

    return nn_model, task_info, model_info


def set_conv_nn(input_shape=(128, 128), conv_layers=2, kernel_size=(5, 5), conv_activation='relu', max_pool_size=(2, 2), dense_layers=1, dense_units=1000, dense_activation='relu', drop_out_val=0.25, num_classes=2):
    """
    Setup a convolution neural network model using the Tensorflow Keras python library.

    Arguments:
    -----------
    input_shape : tuple, shape of the input features (pixel resolution)
    conv_layers : int, number of convolution layers
    kernel_size : tuple, the size of the moving window over convolution layer
    conv_activation : selection of 'relu', 'tanh' etc, type of activation function for the convolution layer
    max_pool_size : tuple, the size of the max pooling area
    dense_layers : int, number of connected layers
    dense_units : int, number of units in the connected layer
    dense_activation : selection of 'relu', 'tanh' etc, type of activation function for the convolution layer
    drop_out_val : float, regularisation variable for setting the specified fraction of input units to 0
    num_classes : int, number of distinct output classes as prediction

    Returns:
    -----------
    nn_model : keras.model, trained neural network model
    task_info : dict, records of useful information when running this function
    task_info : dict, records of useful information about the trained model
    """

    task_name = 'Convolution neural network model setup'
    task_info = {}
    model_info = {}

    start_time = time.time()

    # Cleaning inputs
    conv_activation = conv_activation.lower()
    dense_activation = dense_activation.lower()

    # Build the neural network model
    conv_nn_model = tf.keras.Sequential()
    conv_nn_model.add(tf.keras.layers.Conv2D(32, kernel_size=kernel_size, strides=(1, 1), activation=conv_activation, input_shape=input_shape))
    conv_nn_model.add(tf.keras.layers.MaxPooling2D(pool_size=max_pool_size))
    if conv_layers >= 2:
        for layer in range(conv_layers - 1):
            conv_nn_model.add(tf.keras.layers.Conv2D(32*np.exp2(layer), kernel_size=kernel_size, activation=conv_activation))
            conv_nn_model.add(tf.keras.layers.MaxPooling2D(pool_size=max_pool_size))
            if layer >= 3:
                break
    if drop_out_val > 0:
        conv_nn_model(tf.keras.layers.Dropout(drop_out_val))
    conv_nn_model.add(tf.keras.layers.Flatten())
    conv_nn_model.add(tf.keras.layers.Dense(dense_units, activation=dense_activation))
    if dense_layers >= 2:
        for layer in range(dense_layers - 1):
            conv_nn_model.add(tf.keras.layers.Dense(dense_units, activation=dense_activation))
    conv_nn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    end_time=time.time()
    time_taken=end_time - start_time

    print('Status: Convolution neural network setup successful!\n')
    print('Model summary:')
    print(conv_nn_model.summary())

    # Store the task information
    task_info={
        'task_name': task_name,
        'conv_layers': conv_layers,
        'kernel_size': kernel_size,
        'conv_activation': conv_activation,
        'max_pool_size': max_pool_size,
        'dense_layers': dense_layers,
        'dense_units': dense_units,
        'dense_activation': dense_activation,
        'drop_out_value': drop_out_val,
        'time_taken': time_taken
    }

    model_info={
        'conv_layers': conv_layers,
        'kernel_size': kernel_size,
        'conv_activation': conv_activation,
        'max_pool_size': max_pool_size,
        'dense_layers': dense_layers,
        'dense_units': dense_units,
        'dense_activation': dense_activation,
        'drop_out_value': drop_out_val,
    }

    return conv_nn_model, task_info, model_info


# Training neural network model
def train_nn_keras(X_train, y_train, X_dev, y_dev, nn_model, nn_model_info, learning_rate=0.01, num_epochs=200, loss_type='sparse_categorical_crossentropy', optimizer_type='adam', metrics=['accuracy']):
    """
    Train a neural network model using the Tensorflow Keras python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    X_dev : input development samples
    y_dev : target development samples
    model : a set neural network model to be optimized
    learning_rate : float, rate of model optimization
    num_epoch : number of max optimizer iterations
    optimizer : selection of 'adam', 'rmsprop' etc, type of optimizer used in model training
    loss : selection of 'categorical_crossentropy' etc, type of loss function used in model training

    Returns:
    -----------
    nn_model : keras.model, trained neural network model
    task_info : dict, records of useful information when running this function
    """

    task_name='Neural network model training'
    task_info={}

    start_time=time.time()

    # Cleaning inputs
    loss_type=loss_type.lower()
    optimizer_type=optimizer_type.lower()
    optimizer=set_optimizer_keras(optimizer_type, learning_rate)

    nn_model.compile(optimizer=optimizer, loss=loss_type, metrics=metrics)
    train_history=nn_model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_dev, y_dev))

    end_time=time.time()
    time_taken=end_time - start_time

    print('Status: Neural network successfully trained!')

    # Store the task information
    task_info={
        'task_name': task_name,
        'learning_rate': learning_rate,
        'loss': loss,
        'optimizer': optimizer,
        'train_history': train_history,
        'time_taken': time_taken
    }

    return nn_model, task_info

# Training neural network model
def train_nn(train_dataset, nn_model, nn_model_info, learning_rate=0.01, num_epochs=200, loss_type='sparse_softmax_crossentropy', optimizer_type='adam'):
    """
    Train a neural network model using the Tensorflow python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    model : a set neural network model to be optimized
    learning_rate : float, rate of model optimization
    num_epoch : number of max optimizer iterations
    optimizer : selection of 'adam', 'rmsprop' etc, type of optimizer used in model training
    loss : selection of 'categorical_crossentropy' etc, type of loss function used in model training

    Returns:
    -----------
    nn_model : keras.model, trained neural network model
    task_info : dict, records of useful information when running this function
    train_loss_results : np.array, records of loss values during training
    train_accuracy_results : np.array, records of accuracy values during training
    """
    
    task_name='Neural network model training'
    task_info={}

    start_time=time.time()

    # Cleaning inputs
    loss_type=loss_type.lower()
    optimizer_type=optimizer_type.lower()
    optimizer=set_optimizer(optimizer_type, learning_rate)

    # Note: Rerunning this cell uses the same model variables
    # keep results for plotting
    train_loss_results=[]
    train_accuracy_results=[]

    for epoch in range(num_epochs):
        epoch_loss_avg=tfe.metrics.Mean()
        epoch_accuracy=tfe.metrics.Accuracy()

        # Optimize the model

        for X_train, y_train in train_dataset:
            grads=grad(nn_model, X_train, y_train, loss_type)
            optimizer.apply_gradients(
                zip(grads, nn_model.variables), global_step=tf.train.get_or_create_global_step())

            # Track progress
            # add current batch loss
            epoch_loss_avg(loss(nn_model, X_train, y_train, loss))
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(nn_model(X_train), axis=1, output_type=tf.int32), y_train)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

    end_time=time.time()
    time_taken=end_time - start_time

    print('Status: Neural network successfully trained!')

    # Store the task information
    task_info={
        'task_name': task_name,
        'learning_rate': learning_rate,
        'loss': loss,
        'optimizer': optimizer,
        'time_taken': time_taken
    }

    return nn_model, task_info, train_loss_results, train_accuracy_results

# Neural network optimizer selection
def set_optimizer_keras(optimizer_type, learning_rate):
    if optimizer_type == 'adam':
        optimizer=tf.keras.optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer_type == 'gradient_descent':
        optimizer=tf.keras.optimizers.SGD(
            lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    elif optimizer_type == 'rmsprop':
        optimizer=tf.keras.optimizers.RMSprop(
            lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer_type == 'adagrad':
        optimizer=tf.keras.optimizers.Adagrad(
            lr=learning_rate, epsilon=None, decay=0.0)
    elif optimizer_type == 'adadelta':
        optimizer=tf.keras.optimizers.Adadelta(
            lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
    else:
        optimizer=tf.keras.optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    return optimizer

def set_optimizer(optimizer_type, learning_rate):
    if optimizer_type == 'adam':
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'gradient_descent':
        optimizer=tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'adagrad':
        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'adadelta':
        optimizer=tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif optimizer_type == 'momentum':
        optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate)
    else:
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer

# Neural network loss values function
def loss(model, x, y, loss_type):
    y_=model(x)
    if loss_type == 'sparse_softmax_crossentropy':
        loss_value=tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
    elif loss_type == 'softmax_crossentropy':
        loss_value=tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_)
    elif loss_type == 'cosine_distance':
        loss_value=tf.losses.cosine_distance(labels=y, predictions=y_)
    elif loss_type == 'log_loss':
        loss_value=tf.losses.log_loss(labels=y, predictions=y_)
    elif loss_type == 'mean_squared_error':
        loss_value=tf.losses.mean_squared_error(labels=y, predictions=y_)
    else:
        loss_value=tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
    return loss_value

# Neural network gradient values function
def grad(model, X_train, y_train, loss_type):
    with tf.GradientTape() as tape:
        loss_value=loss(model, X_train, y_train, loss_type)
    return tape.gradient(loss_value, model.variables)

# SVM model
def train_svm_sklearn(X_train, y_train, C=1, gamma=1):
    """
    Train a SVM model using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    C : float, SVM training parameter
    gamma : float, SVM training parameter
    """

    task_name='SVM model training'

    start_time=time.time()

    svm_model=sklearn.svm.SVC(C=C, gamma=gamma)
    svm_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    print('Status: SVM successfully trained!')

    # Store the task information
    task_info={
        'task_name': task_name,
        'C': C,
        'gamma': gamma,
        'time_taken': time_taken
    }

    return svm_model, task_info

# One-class SVM model
def train_svm_anomaly_sklearn(X_train, y_train, nu=1=0.2, kernel='rbf'):
    """
    Train an one-class SVM model (for anomaly detection) using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    nu : float, SVM training parameter, where it is upper bounded at the fraction of outliers and a lower bounded at the fraction of support vectors
    kernel : selection of 'rbf', 'linear', 'poly', 'sigmoid', type of kernel used for one-class SVM model
    """

    task_name='One-class SVM model training'

    start_time=time.time()

    svm_anomaly_model=sklearn.svm.OneClassSVM(nu=nu, kernel=kernel)
    svm_anomaly_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    print('Status: One-class SVM successfully trained!')

    # Store the task information
    task_info={
        'task_name': task_name,
        'nu' : nu,
        'kernel': kernel,
        'time_taken': time_taken
    }

    return svm_anomaly_model, task_info

# Logistic Regression model
def train_lgr_sklearn(X_train, y_train):
    """
    Train a Logistic Regression model using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    """

    task_name='Logstic Regression model training'

    start_time=time.time()

    lgr_model=sklearn.linear_model.LogisticRegressionCV()
    lgr_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    # Store the task information
    task_info={
        'task_name': task_name,
        'time_taken': time_taken
    }

    return lgr_model, task_info

# Random Forest model
def train_rforest_sklearn(X_train, y_train, max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=10):
    """
    Train a Random Forest model using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    max_depth : integer, max depth of the trees (None implies trees will keep forking until limit)
    max_features : integer or float, number of features to consider when looking for the best split
    min_samples_leaf : integer or float, min number of samples required to be at a leaf node
    min_samples_split : integer or float, min number of samples required to split an internal node
    n_estimators : integer, number of tree in the forest model
    """

    task_name='Random Forest model training'

    start_time=time.time()

    rforest_model=sklearn.ensemble.RandomForestClassifier(max_depth=max_depth, max_features=max_feature, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)

    rforest_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    # Store the task information
    task_info={
        'task_name': task_name,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'time_taken': time_taken
    }

    return rforest_model, task_info

# Random Forest model
def train_isoforest_sklearn(X_train, y_train, max_depth=None, max_features=None, n_estimators=100, outliers=0.1):
    """
    Train a Isolation Forest model (for anomaly detection) using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    max_depth : integer, max depth of the trees (None implies trees will keep forking until limit)
    max_features : integer or float, number of features to consider when looking for the best split
    n_estimators : integer, number of tree in the isolation forest model
    outlier : float, the proportion of outliers in the data
    """

    task_name='Isolation Forest model training'

    start_time=time.time()

    isoforest_model=sklearn.ensemble.IsolationForest(max_depth=max_depth, max_features=max_feature, n_estimators=n_estimators, contamination=outliers)

    isoforest_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    # Store the task information
    task_info={
        'task_name': task_name,
        'max_depth': max_depth,
        'max_features': max_features,
        'outlier_proportion': outliers,
        'time_taken': time_taken
    }

    return isoforest_model, task_info

# Adaboost (boosted) model
def train_adaboost_sklearn(X_train, y_train, n_estimators=50, learning_rate=1):
    """
    Train an Adaboost model using the Sklearn python library.

    Arguments
    -----------
    X_train : input training samples
    y_train : target training samples
    n_estimators : integer, max number of models at which boosting is terminated
    learning_rate : float, rate of model optimization
    """

    task_name='Adaboost model training'

    start_time=time.time()

    adaboost_model=sklearn.ensemble.AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    end_time=time.time()
    time_taken=end_time - start_time

    # Store the task information
    task_info={
        'task_name': task_name,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'time_taken': time_taken
    }

    return adaboost_model, task_info

# Decision Tree model
def train_dtree_sklearn(X_train, y_train, max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2):
    """
    Train a Decision Tree model using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    max_depth : integer, max depth of the trees (None implies trees will keep forking until limit)
    max_features : integer or float, number of features to consider when looking for the best split
    min_samples_leaf : integer or float, min number of samples required to be at a leaf node
    min_samples_split : integer or float, min number of samples required to split an internal node
    """

    task_name='Decision Tree model training'

    start_time=time.time()

    dtree_model=sklearn.tree.DecisionTreeClassifier()
    dtree_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    # Store the task information
    task_info={
        'task_name': task_name,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'time_taken': time_taken
    }

    return dtree_model, task_info

# Naive Bayes model
def train_nb_sklearn(X_train, y_train):
    """
    Train a Naive Bayes model using the Sklearn python library.

    Arguments:
    -----------
    X_train : input training samples
    y_train : target training samples
    """

    task_name='Naive Bayes model training'

    start_time=time.time()

    nb_model=sklearn.naive_bayes.MultinomialNB()
    nb_model.fit(X_train, y_train)

    end_time=time.time()
    time_taken=end_time - start_time

    # Store the task information
    task_info={
        'task_name': task_name,
        'time_taken': time_taken
    }

    return nb_model, task_info
