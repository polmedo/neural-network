import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(weighted_sum)
    

def train(neuron, X, y, epochs=1000, learning_rate=0.1):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in zip(X, y):
            # Forward pass: compute prediction
            prediction = neuron.forward(inputs)
            loss = (prediction - target) ** 2
            total_loss += loss
        
            # Backward pass: compute gradients
            error = prediction - target
            d_output = error * prediction * (1 - prediction)

            # Update weights and bias
            neuron.weights -= learning_rate * d_output * inputs
            neuron.bias -= learning_rate * d_output

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(y)
            print(f'Epoch {epoch + 1}: loss={avg_loss:.3f}')
            print(f'Weights: {neuron.weights[0]:.4f}, {neuron.weights[1]:.4f}; Bias: {neuron.bias:.4f}')

    

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_OR = np.array([0, 1, 1, 1])

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([0, 1, 1, 0])

# Create a neuron with 2 inputs for the AND gate
neuron = Neuron(num_inputs=2)

# Train for 1000 epochs with a learning rate of 0.1
train(neuron, X_XOR, y_XOR, epochs=10000, learning_rate=0.4)

print("\nTesting the trained neuron:")
for inputs in X:
    prediction = neuron.forward(inputs)
    print(f"{inputs} => {prediction:.2f}")