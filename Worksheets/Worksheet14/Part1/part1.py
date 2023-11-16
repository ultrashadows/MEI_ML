import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

np.random.seed(seed=2023)

# Load data
digits = load_digits()
print('We have %d samples ' % len(digits.target))

# Plot the first 64 samples to get a sense of the data
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r)
    ax.text(0, 1, str(digits.target[i]), bbox=dict(facecolor='white'))

fig.show()
fig.savefig('./resources/data')

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=2023)
print('Number of samples in training set: %d' % (len(y_train)))
print('Number of samples in test set: %d' % (len(y_test)))

# Standardise training data's data and fit
scaler = StandardScaler()
scaler.fit(X_train)

# Apply the transformations to the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize ANN classifier (3 hidden layers with 200, 200 and 150 neurons)
mlp = MLPClassifier(hidden_layer_sizes=(200, 200, 150), activation='logistic', max_iter=1000)

# Train the classifier with the training data
mlp.fit(X_train_scaled, y_train)
print("Training set score : %f" % mlp.score(X_train_scaled, y_train))
print("Test set score : %f" % mlp.score(X_test_scaled, y_test))

# Predict results from the test data
X_test_predicted = mlp.predict(X_test_scaled)

fig = plt.figure(figsize=(8, 8))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# Plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.gray_r)
    # label the image with the target value
    if X_test_predicted[i] == y_test[i]:
        ax.text(0, 1, X_test_predicted[i], color='green', bbox=dict(facecolor='white'))
    else:
        ax.text(0, 1, X_test_predicted[i], color='red', bbox=dict(facecolor='white'))
        ax.text(6, 1, y_test[i], color='green', bbox=dict(facecolor='white'))

fig.show()
fig.savefig('./resources/predict')
