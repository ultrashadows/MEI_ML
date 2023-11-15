import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(seed=2023)

# Load data
data = pd.read_csv('./resources/Diabetes.csv')

# Select desired features
selected_features = data[['bmi', 'age']]
target = data['class']

# Split data to training and testing data
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=2023)
print('Number of samples in training set: %d' % (len(y_train)))
print('Number of samples in test set: %d' % (len(y_test)))

# Standardise data , and fit only to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Apply the transformations to the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize ANN classifier
mlp = MLPClassifier(hidden_layer_sizes=(9, 9, 8), activation='tanh', solver='sgd', max_iter=1000)

# Train the classifier with the training data
mlp.fit(X_train_scaled, y_train)
print("Training set score : %f" % mlp.score(X_train_scaled, y_train))
print("Test set score : %f" % mlp.score(X_test_scaled, y_test))

# Predict results.md from the test data
X_test_predicted = mlp.predict(X_test_scaled)

# Plot results.md
plt.scatter(y_test, X_test_predicted)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.savefig('./resources/predict_diabetes-1')
plt.show()
