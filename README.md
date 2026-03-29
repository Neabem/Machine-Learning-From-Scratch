# Machine Learning Engines From Scratch

A deep dive into the core mathematical engines that power modern artificial intelligence. 

Most machine learning tutorials start by importing `scikit-learn` or `PyTorch` and using them as black boxes. In this repository, I built the fundamental optimization algorithms and artificial neurons entirely from scratch using pure Python, NumPy, and Calculus. 

---

## 📁 Project 1: Optimization & Logistic Regression
**File:** `1_Optimization_and_Logistic_Regression.ipynb`

### 🧠 Concepts Covered & Coded:
* **Gradient Descent (1D & 2D):** Implementing the calculus (derivatives) to navigate complex cost function landscapes.
* **Optimization Variations:** Coding standard Batch Gradient Descent, Stochastic Gradient Descent (SGD), and the industry-standard Mini-Batch Gradient Descent.
* **The Artificial Neuron:** Writing the mathematics for the Sigmoid Activation Function to compress linear equations into percentage probabilities.
* **Log Loss (Binary Cross-Entropy):** Implementing the proper loss function for classification models. 
* **Logistic Regression:** Combining the Optimizer, Predictor, and Loss Function to build a custom classifier that successfully diagnoses a slice of the Wisconsin Breast Cancer dataset.

---

## 📁 Project 2: Deep Neural Network (Pure NumPy)
**File:** `Deep_Neural_Networks_From_Scratch.ipynb`

Building on the single neuron, this project constructs a fully functioning 2-layer Deep Neural Network from absolute scratch. 

### 🚀 What This Engine Does
This model is built to predict binary classifications. It takes in patient features, routes them through a hidden layer, calculates the error, and physically updates its own weights using backpropagation.

### 🛠️ Concepts Mastered
* **Forward Propagation:** Matrix dot products to route data through hidden layers.
* **Activation Functions:** Implementing the Sigmoid squishing function and its derivative.
* **Backpropagation:** Applying the Chain Rule to calculate weight gradients (`dW`, `db`) across multiple layers.
* **Gradient Descent:** Implementing a learning rate (`alpha`) to iteratively tune weights and minimize loss.
* **Matrix Dimensionality:** Handling NumPy shape alignments, transpositions (`.T`), and preventing rank-1 array bugs (`keepdims=True`).

---
**Technologies Used Across Projects:** Pure Python, NumPy, Calculus (Derivatives and Gradients).
