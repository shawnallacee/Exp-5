# Experiment 5 : Multilayer Perceptron (MLP) Neural Network

## Aim

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> **Note:** The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP
> experiment.

## Overview

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural
network consisting of an input layer, one or more hidden layers, and an
output layer. It learns patterns in data by adjusting weights and biases
during training using backpropagation.

In this experiment, the MLP model is trained on the **Digits dataset**
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## Algorithm

1.  **Start.**
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. **Stop.**
ry slightly with changes to the train-test split,
    random seed, or model parameters.


## Program
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.datasets import load_digits  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier  
from sklearn.preprocessing import StandardScaler  

\# Load the Digits dataset and separate features and target labels  
print("--- Dataset Initialization ---")  
digits = load_digits()  
X = digits.data  
y = digits.target  

print(f"Dataset shape: {X.shape}")  
print(f"Number of classes: {len(np.unique(y))} (Digits 0-9)")    

\# Split the dataset into training and testing sets (80:20 ratio)  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42  
)  
print(f"Training set size: {X_train.shape[0]} samples")  
print(f"Test set size: {X_test.shape[0]} samples")  

\# Standardize the input features    
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)    
X_test_scaled = scaler.transform(X_test)  

\# Create an MLP classifier with a hidden layer structure (e.g., 64 hidden neurons)  
print("\n--- Model Training ---")  
mlp = MLPClassifier(  
    hidden_layer_sizes=(64,),    
    activation="relu",  
    solver="adam",  
    max_iter=500,  
    random_state=42,  
)  
mlp.fit(X_train_scaled, y_train)  
print("MLP Classifier trained successfully.")    
print(f"Number of iterations until convergence: {mlp.n_iter_}")  

\# Predict the labels for the test data    
y_pred = mlp.predict(X_test_scaled)  

\# Evaluate the model using test accuracy and classification report  
print("\n--- Evaluation Metrics ---")  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Test Accuracy: {accuracy:.4f}")  
print("\nClassification Report:\n", classification_report(y_test, y_pred))  

\# Plot the training loss curve  
print("\n--- Generating Training Loss Curve ---")  
plt.figure(figsize=(8, 5))  
plt.plot(mlp.loss_curve_, marker="o", color="teal", linewidth=2)  
plt.title("MLP Training Loss Curve")  
plt.xlabel("Iterations / Epochs")  
plt.ylabel("Loss")  
plt.grid(True, linestyle="--", alpha=0.6)  
plt.tight_layout()  
print("Training loss curve plotted successfully.")  
plt.show()        

## output
<img width="527" height="672" alt="image" src="https://github.com/user-attachments/assets/bbbe5fec-2135-4673-933c-b33dcd72c284" />
<img width="1000" height="612" alt="image" src="https://github.com/user-attachments/assets/c2d2f1ce-5d70-40e2-9e37-e3e7137dd55c" />

## Result
Thus to implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset successfully.
