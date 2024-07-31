import numpy as np
import matplotlib.pyplot as plt

# Read and preprocess the data
def Read_Data2(filename='cross.pat'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        a = f.readlines()
        for line in range(1, len(a), 3):
            temp1 = np.array([float(element) for element in a[line][:-1].split()])
            temp2 = np.array([float(element) for element in a[line+1].split()])
            data.append(np.append(temp1, temp2))
    data = np.array(data)
    for i in data:
        input.append(i[:-2])
        design_output.append(i[-2:])
    return input, design_output

def shuffle_Data(input, design_output):
    combined = list(zip(input, design_output))
    np.random.shuffle(combined)
    shuffled_input, shuffled_design_output = zip(*combined)
    return np.array(shuffled_input), np.array(shuffled_design_output)

def split_data(input_data, output_data, train_size=0.8):
    num_samples = len(input_data)
    split_index = int(num_samples * train_size)
    X_train = input_data[:split_index]
    X_test = input_data[split_index:]
    y_train = output_data[:split_index]
    y_test = output_data[split_index:]
    return X_train, X_test, y_train, y_test

input, design_output = Read_Data2()
sinput, sdesign_output = shuffle_Data(input, design_output)

# MLP implementation
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, momentum_rate=0.9):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.velocities_w = []
        self.velocities_b = []
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            self.velocities_w.append(np.zeros((layer_sizes[i], layer_sizes[i+1])))
            self.velocities_b.append(np.zeros((1, layer_sizes[i+1])))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        self.activations = [X]
        self.zs = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.zs.append(z)
            self.activations.append(self.sigmoid(z))
        
        return self.activations[-1]
    
    def backward(self, X, y):
        m = y.shape[0]
        output_error = self.activations[-1] - y
        delta = output_error * self.sigmoid_derivative(self.activations[-1])
        
        deltas = [delta]
        
        for i in range(len(self.weights) - 2, -1, -1):
            delta = deltas[-1].dot(self.weights[i+1].T) * self.sigmoid_derivative(self.activations[i+1])
            deltas.append(delta)
        
        deltas.reverse()
        
        for i in range(len(self.weights)):
            self.velocities_w[i] = self.momentum_rate * self.velocities_w[i] + (1 - self.momentum_rate) * self.activations[i].T.dot(deltas[i]) / m
            self.velocities_b[i] = self.momentum_rate * self.velocities_b[i] + (1 - self.momentum_rate) * np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * self.velocities_w[i]
            self.biases[i] -= self.learning_rate * self.velocities_b[i]
    
    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)
    
    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

# Configuration
layer_sizes = [2, 5, 2]  # Input layer, two hidden layers, output layer
learning_rate = 0.25
epochs = 15000
momentum_rate = 0.9

# Create the MLP
mlp = MLP(layer_sizes=layer_sizes, learning_rate=learning_rate, momentum_rate=momentum_rate)

# Confusion matrix
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

# Plot confusion matrix
def plot_confusion_matrix(cm, fold_idx, accuracy):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {fold_idx}\nAccuracy: {accuracy * 100:.2f}%')
    plt.show()

# 10-fold cross-validation
def k_fold_cross_validation(input_data, output_data, k=10):
    fold_size = len(input_data) // k
    accuracies = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test_fold = input_data[start:end]
        y_test_fold = output_data[start:end]
        
        X_train_fold = np.concatenate((input_data[:start], input_data[end:]), axis=0)
        y_train_fold = np.concatenate((output_data[:start], output_data[end:]), axis=0)
        
        # Train the model on the training fold
        mlp = MLP(layer_sizes=layer_sizes, learning_rate=learning_rate, momentum_rate=momentum_rate)
        mlp.train(X_train_fold, y_train_fold, epochs=epochs)
        
        # Evaluate the model on the testing fold
        accuracy = mlp.accuracy(X_test_fold, y_test_fold)
        accuracies.append(accuracy)
        
        # Generate confusion matrix
        y_test_labels = np.argmax(y_test_fold, axis=1)
        y_test_predictions = mlp.predict(X_test_fold)
        cm = confusion_matrix(y_test_labels, y_test_predictions, 2)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, i + 1, accuracy)
    
    return np.array(accuracies)

# Perform cross-validation
accuracies = k_fold_cross_validation(sinput, sdesign_output, k=10)
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f"Mean accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard deviation of accuracy: {std_accuracy * 100:.2f}%")

# Final training on the entire dataset
mlp.train(sinput, sdesign_output, epochs=epochs)

# Evaluate the MLP on the entire dataset
final_accuracy = mlp.accuracy(sinput, sdesign_output)
print(f"Final accuracy on the entire dataset: {final_accuracy * 100:.2f}%")

# Plot fold accuracies with additional lines for mean and final accuracy
plt.figure()
plt.plot(range(1, 11), accuracies * 100, marker='o', label='Fold Accuracy')
plt.axhline(y=mean_accuracy * 100, color='r', linestyle='--', label='Mean Accuracy')
plt.axhline(y=final_accuracy * 100, color='g', linestyle='--', label='Final Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Each Fold with Mean and Final Accuracy')
plt.legend()
plt.show()
