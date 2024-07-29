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
def pi_squared_loss(Y, AL):
    return np.mean((np.pi**2) * (Y - AL)**2)

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
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    
    return parameters

# Train the model
def train_mlp(X, Y, hidden_layers, epochs, learning_rate):
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    parameters = initialize_parameters(input_dim, hidden_layers, output_dim)
    
    for epoch in range(epochs):
        AL, caches = forward_propagation(X, parameters, hidden_layers)
        loss = pi_squared_loss(Y, AL)
        percent_loss = percentage_loss(Y, AL)
        grads = backward_propagation(X, Y, parameters, caches, hidden_layers)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}, Percent Loss: {percent_loss:.2f}%")
    
    return parameters, loss, percent_loss

# Perform k-fold cross-validation
def k_fold_cross_validation(X, Y, hidden_layers, epochs, learning_rate, k=10):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    avg_loss = 0
    avg_percent_loss = 0
    
    all_predictions_train = []
    all_predictions_val = []
    all_predictions_test = []
    
    for i in range(k):
        validation_indices = indices[i*fold_size:(i+1)*fold_size]
        training_indices = np.setdiff1d(indices, validation_indices)
        
        X_train, X_val = X[training_indices], X[validation_indices]
        Y_train, Y_val = Y[training_indices], Y[validation_indices]
        
        # Train model
        parameters, _, _ = train_mlp(X_train, Y_train, hidden_layers, epochs, learning_rate)
        
        # Predict on training and validation sets
        AL_train, _ = forward_propagation(X_train, parameters, hidden_layers)
        AL_val, _ = forward_propagation(X_val, parameters, hidden_layers)
        
        # Predict on test set (assuming X_test and Y_test are available globally)
        AL_test, _ = forward_propagation(X_test, parameters, hidden_layers)
        
        all_predictions_train.append((Y_train, AL_train))
        all_predictions_val.append((Y_val, AL_val))
        all_predictions_test.append((Y_test, AL_test))
        
        # Calculate loss and percent loss
        loss = pi_squared_loss(Y_val, AL_val)
        percent_loss = percentage_loss(Y_val, AL_val)
        
        avg_loss += loss
        avg_percent_loss += percent_loss
        
        print(f"Fold {i+1}, Validation Loss: {loss}, Validation Percent Loss: {percent_loss:.2f}%")
    
    avg_loss /= k
    avg_percent_loss /= k
    
    return avg_loss, avg_percent_loss, all_predictions_train, all_predictions_val, all_predictions_test

# Load and normalize data
file_path = 'Flood_dataset.txt'
X, Y = load_data(file_path)
X, mean, std = normalize_data(X)

# Split data into training, testing, and validation sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

# Hyperparameters
hidden_layers = [10, 8, 5]
epochs = 22000
learning_rate = 0.0001
k = 10

# Perform k-fold cross-validation
avg_loss, avg_percent_loss, all_predictions_train, all_predictions_val, all_predictions_test = k_fold_cross_validation(X_train, Y_train, hidden_layers, epochs, learning_rate, k)

print(f"Average Loss: {avg_loss}, Average Percent Loss: {avg_percent_loss:.2f}%")

# Plotting results for each fold
for i, (train_data, val_data, test_data) in enumerate(zip(all_predictions_train, all_predictions_val, all_predictions_test)):
    Y_train, AL_train = train_data
    Y_val, AL_val = val_data
    Y_test, AL_test = test_data
    
    # Plot training data
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(Y_train)), Y_train, label='Real Data (Train)', linestyle='-', color='blue')
    plt.plot(range(len(AL_train)), AL_train, label='Predicted Data (Train)', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Training Data Fold {i+1}')
    plt.show(block=False)  # Non-blocking show
    
    # Plot validation data
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(Y_val)), Y_val, label='Real Data (Val)', linestyle='-', color='blue')
    plt.plot(range(len(AL_val)), AL_val, label='Predicted Data (Val)', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Validation Data Fold {i+1}')
    plt.show(block=False)  # Non-blocking show
    
    # Plot test data
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(Y_test)), Y_test, label='Real Data (Test)', linestyle='-', color='blue')
    plt.plot(range(len(AL_test)), AL_test, label='Predicted Data (Test)', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Test Data Fold {i+1}')
    plt.show(block=False)  # Non-blocking show

# Wait to ensure all figures are displayed
plt.show()
