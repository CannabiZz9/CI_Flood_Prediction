import numpy as np
import matplotlib.pyplot as plt

# Function to load data from a file with space-separated values
def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y

# Function to normalize data
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

# Initialize parameters
def initialize_parameters(input_dim, hidden_layers, output_dim):
    parameters = {}
    layer_dims = [input_dim] + hidden_layers + [output_dim]
    
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        parameters[f"b{l}"] = np.zeros((1, layer_dims[l]))
    
    return parameters

# Initialize velocity terms
def initialize_velocity(parameters):
    velocity = {}
    L = len(parameters) // 2
    for l in range(1, L + 1):
        velocity[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        velocity[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
    return velocity

# Forward propagation
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_propagation(X, parameters, hidden_layers):
    caches = {}
    A = X
    L = len(hidden_layers) + 1
    
    for l in range(1, L):
        Z = np.dot(A, parameters[f"W{l}"]) + parameters[f"b{l}"]
        A = sigmoid(Z)
        caches[f"Z{l}"] = Z
        caches[f"A{l}"] = A
    
    ZL = np.dot(A, parameters[f"W{L}"]) + parameters[f"b{L}"]
    AL = ZL
    caches[f"Z{L}"] = ZL
    caches[f"A{L}"] = AL
    
    return AL, caches

# Loss calculation
def mse_loss(Y, AL):
    return (np.mean((Y - AL))**2)

# Calculate percentage loss
def percentage_loss(Y, AL):
    return np.mean(np.abs((Y - AL) / Y)) * 100

# Backward propagation
def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def backward_propagation(X, Y, parameters, caches, hidden_layers):
    grads = {}
    m = X.shape[0]
    L = len(hidden_layers) + 1
    
    dZL = 2 * np.pi**2 * (caches[f"A{L}"] - Y)
    grads[f"dW{L}"] = np.dot(caches[f"A{L-1}"].T, dZL) / m
    grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True) / m
    
    for l in reversed(range(1, L)):
        dA_prev = np.dot(dZL, parameters[f"W{l+1}"].T)
        dZ = dA_prev * sigmoid_derivative(caches[f"Z{l}"])
        grads[f"dW{l}"] = np.dot(caches[f"A{l-1}"].T, dZ) / m if l > 1 else np.dot(X.T, dZ) / m
        grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
        dZL = dZ
    
    return grads

# Update parameters
def update_parameters(parameters, grads, velocity, learning_rate, momentum_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        velocity[f"dW{l}"] = momentum_rate * velocity[f"dW{l}"] + (1 - momentum_rate) * grads[f"dW{l}"]
        velocity[f"db{l}"] = momentum_rate * velocity[f"db{l}"] + (1 - momentum_rate) * grads[f"db{l}"]
        
        parameters[f"W{l}"] -= learning_rate * velocity[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * velocity[f"db{l}"]
    
    return parameters, velocity

# Train the model 
def train_mlp(X, Y, hidden_layers, epochs, learning_rate, momentum_rate):
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    parameters = initialize_parameters(input_dim, hidden_layers, output_dim)
    velocity = initialize_velocity(parameters)
    
    print("Initial parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")
    
    for epoch in range(epochs):
        AL, caches = forward_propagation(X, parameters, hidden_layers)
        loss = mse_loss(Y, AL)
        percent_loss = percentage_loss(Y, AL)
        grads = backward_propagation(X, Y, parameters, caches, hidden_layers)
        parameters, velocity = update_parameters(parameters, grads, velocity, learning_rate, momentum_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE Loss: {loss}, Percent Loss: {percent_loss:.2f}%")
    
    print("Final parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")
    
    return parameters, loss, percent_loss

# Load and normalize data
file_path = 'Flood_dataset.txt'
X, Y = load_data(file_path)
print(X)
print(Y)
X, mean, std = normalize_data(X)

# Split data into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Train the model
hidden_layers = [10, 5]
epochs = 15000
learning_rate = 0.0001
momentum_rate = 0.9
parameters, final_loss, final_percent_loss = train_mlp(X_train, Y_train, hidden_layers, epochs, learning_rate, momentum_rate)

# Make predictions
AL_train, _ = forward_propagation(X_train, parameters, hidden_layers)
AL_test, _ = forward_propagation(X_test, parameters, hidden_layers)

# Plot results for training data
plt.figure(figsize=(10, 5))
plt.scatter(range(len(Y_train)), Y_train, label='Real Data (Train)')
plt.scatter(range(len(AL_train)), AL_train, label='Predicted Data (Train)')
plt.legend()
plt.title(f'Training Data - Final MSE Loss: {final_loss:.7f}, Percent Loss: {final_percent_loss:.2f}%')
plt.show()

# Plot results for testing data
plt.figure(figsize=(10, 5))
plt.scatter(range(len(Y_test)), Y_test, label='Real Data (Test)')
plt.scatter(range(len(AL_test)), AL_test, label='Predicted Data (Test)')
plt.legend()
plt.title(f'Testing Data - Final MSE Loss: {final_loss:.7f}, Percent Loss: {final_percent_loss:.2f}%')
plt.show()
