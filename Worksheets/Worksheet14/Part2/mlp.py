import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def mlp_init(features, target, seed, hidden_layers, activation, solver, max_iterations, save_path):
    np.random.seed(seed)

    # Split data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)
    print('Number of samples in training set: %d' % (len(y_train)))
    print('Number of samples in test set: %d' % (len(y_test)))

    # Standardise training data's data and fit
    scaler = StandardScaler()
    scaler.fit(x_train)

    # Apply the transformations to the data
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Initialize ANN classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation,
                        solver=solver if solver is not None else 'adam', max_iter=max_iterations)

    # Train the classifier with the training data
    mlp.fit(x_train_scaled, y_train)
    print("Training set score: %f" % mlp.score(x_train_scaled, y_train))
    print("Test set score: %f" % mlp.score(x_test_scaled, y_test))

    # Document results
    with open('./resources/results.txt', 'a' if os.path.exists('./resources/results.txt') else 'w') as file:
        file.write('Iteration results from ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        file.write('Seed: ' + str(seed) + '\n')
        file.write('Hidden Layers: ' + str(hidden_layers) + '\n')
        file.write('Activation: ' + str(activation) + '\n')
        file.write('Solver: ' + (str(solver) if solver is not None else 'adam') + '\n')
        file.write('Max Iterations: ' + str(max_iterations) + '\n')
        file.write('Training Accuracy: ' + str(mlp.score(x_train_scaled, y_train)) + '\n')
        file.write('Test Accuracy: ' + str(mlp.score(x_test_scaled, y_test)) + '\n\n')

    # Predict results from the test data
    x_test_predicted = mlp.predict(x_test_scaled)

    # Plot results
    plt.scatter(y_test, x_test_predicted)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig(save_path)
    plt.show()
