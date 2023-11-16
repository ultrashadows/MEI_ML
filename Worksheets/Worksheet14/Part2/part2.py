import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp import mlp_init

# Load data
data = pd.read_csv('./resources/Diabetes.csv')

# Exercise 1: ANN only using BMI and age

# Select desired features
selected_features = data[['bmi', 'age']]
target = data['class']

# Train model
mlp_init(features=selected_features, target=target, seed=2023, hidden_layers=(200, 200, 150), activation='logistic',
         solver=None, max_iterations=1000, save_path='./resources/results_1')

# Exercise 2: Retrain the model using 2 different structures

# Train model 1
mlp_init(features=selected_features, target=target, seed=2023, hidden_layers=(500, 500, 200), activation='logistic',
         solver=None, max_iterations=2000, save_path='./resources/results_2-1')

# Train model 2
mlp_init(features=selected_features, target=target, seed=2023, hidden_layers=(3, 3, 2, 2), activation='logistic',
         solver=None, max_iterations=2500, save_path='./resources/results_2-2')

# Exercise 3: Retrain the model considering all variables

# Select all data except class
selected_features = data.drop('class', axis=1)

# Train model
mlp_init(features=selected_features, target=target, seed=2023, hidden_layers=(200, 200, 150), activation='logistic',
         solver=None, max_iterations=2500, save_path='./resources/results_3')

# Exercise 4: Retrain the model using logistic regression

# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=2023)

# Standardize training data
scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize Logistic Regression classifier
logistic_regression = LogisticRegression(random_state=2023)

# Train the classifier with the training data
logistic_regression.fit(x_train_scaled, y_train)

# Predict results from the test data
y_test_predicted = logistic_regression.predict(x_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_predicted)
print("Test set accuracy (Logistic Regression): %f" % accuracy)
