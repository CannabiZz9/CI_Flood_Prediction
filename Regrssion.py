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
def train_mlp(X, Y, hidden_layers, epochs, learning_rate, momentum_rate, X_test, Y_test):
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    parameters = initialize_parameters(input_dim, hidden_layers, output_dim)
    velocity = initialize_velocity(parameters)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        AL, caches = forward_propagation(X, parameters, hidden_layers)
        loss = mse_loss(Y, AL)
        percent_loss = percentage_loss(Y, AL)
        grads = backward_propagation(X, Y, parameters, caches, hidden_layers)
        parameters, velocity = update_parameters(parameters, grads, velocity, learning_rate, momentum_rate)
        
        # Evaluate the model
        AL_test, _ = forward_propagation(X_test, parameters, hidden_layers)
        test_loss = mse_loss(Y_test, AL_test)
        epoch_losses.append(test_loss)
    
    return parameters, epoch_losses

# Implement 10-fold cross-validation
def k_fold_cross_validation(X, Y, k=10):
    fold_size = len(X) // k
    indices = np.random.permutation(len(X))
    
    fold_losses = []
    fold_scores = []
    final_parameters = None
    final_loss = None
    final_percent_loss = None
    last_X_train, last_Y_train, last_X_test, last_Y_test = None, None, None, None
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
        
        # Train the model
        hidden_layers = [10, 5]
        epochs = 15000
        learning_rate = 0.0001
        momentum_rate = 0.9
        parameters, epoch_losses = train_mlp(X_train, Y_train, hidden_layers, epochs, learning_rate, momentum_rate, X_test, Y_test)
        
        # Evaluate the model
        AL_test, _ = forward_propagation(X_test, parameters, hidden_layers)
        test_loss = mse_loss(Y_test, AL_test)
        test_percent_loss = percentage_loss(Y_test, AL_test)
        
        fold_losses.append(test_loss)
        fold_scores.append(epoch_losses)
        
        if i == k - 1:
            final_parameters = parameters
            final_loss = test_loss
            final_percent_loss = test_percent_loss
            last_X_train, last_Y_train, last_X_test, last_Y_test = X_train, Y_train, X_test, Y_test
        
        print(f"Fold {i+1}, Test MSE Loss: {test_loss}, Test Percent Loss: {test_percent_loss:.2f}%")
    
    avg_loss = np.mean(fold_losses)
    avg_percent_loss = np.mean(test_percent_loss)
    print(f"Average Test MSE Loss: {avg_loss}, Average Test Percent Loss: {avg_percent_loss:.2f}%")
    
    return final_parameters, final_loss, final_percent_loss, last_X_train, last_Y_train, last_X_test, last_Y_test, fold_scores, fold_losses, avg_loss, avg_percent_loss

# Load and normalize data
file_path = 'Flood_dataset.txt'
X, Y = load_data(file_path)
X, mean, std = normalize_data(X)

# Perform 10-fold cross-validation
parameters, final_loss, final_percent_loss, X_train, Y_train, X_test, Y_test, fold_scores, fold_losses, avg_loss, avg_percent_loss = k_fold_cross_validation(X, Y)
 
# Plot results for training data (last fold)
AL_train, _ = forward_propagation(X_train, parameters, [10, 5])
AL_test, _ = forward_propagation(X_test, parameters, [10, 5])

plt.figure(figsize=(10, 5))
plt.plot(Y_train, label='Real Data (Train)', color='blue')
plt.plot(AL_train, label='Predicted Data (Train)', color='red')
plt.legend()
plt.title(f'Training Data - Final MSE Loss: {final_loss:.7f}, % Loss: {final_percent_loss:.2f}%, AVG Loss: {avg_loss:.2f}, AVG % Loss: {avg_percent_loss:.2f}%')
plt.show()

# Plot results for testing data (last fold)
plt.figure(figsize=(10, 5))
plt.plot(Y_test, label='Real Data (Test)', color='blue')
plt.plot(AL_test, label='Predicted Data (Test)', color='red')
plt.legend()
plt.title(f'Testing Data - Final MSE Loss: {final_loss:.7f}, % Loss: {final_percent_loss:.2f}%, AVG Loss: {avg_loss:.2f}, AVG % Loss: {avg_percent_loss:.2f}%')
plt.show()

# Plot Test MSE Loss over epochs for each fold
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(fold_scores)))
for i, epoch_losses in enumerate(fold_scores):
    plt.plot(range(len(epoch_losses)), epoch_losses, color=colors[i], label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Test MSE Loss')
plt.title('Test MSE Loss vs Epochs for Each Fold')
plt.legend()
plt.show()

# Plot final Test MSE Loss for each fold
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(fold_losses) + 1), fold_losses, color='skyblue', edgecolor='black')
plt.xlabel('Fold')
plt.ylabel('Test MSE Loss')
plt.title('Test MSE Loss for Each Fold')
plt.xticks(range(1, len(fold_losses) + 1))
plt.show()