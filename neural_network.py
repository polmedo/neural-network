import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases randomly
        self.hidden_weights = np.random.rand(2, 2)  # 2x2 matrix (input to hidden)
        self.hidden_bias = np.random.rand(2)
        self.output_weights = np.random.rand(2)      # 2x1 vector (hidden to output)
        self.output_bias = np.random.rand()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        # Hidden layer computation
        self.hidden_sum = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        self.hidden_activation = self.sigmoid(self.hidden_sum)

        # Output layer computation
        self.output_sum = np.dot(self.hidden_activation, self.output_weights) + self.output_bias
        return self.sigmoid(self.output_sum)

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target in zip(X, y):
                # Forward pass
                prediction = self.forward(inputs)
                loss = (prediction - target) ** 2
                total_loss += loss

                # Backward pass (backpropagation)
                # Output layer error
                error = prediction - target
                d_output = error * self.sigmoid_derivative(prediction)

                # Hidden layer error
                error_hidden = d_output * self.output_weights
                d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_activation)

                # Update weights and biases
                self.output_weights -= learning_rate * d_output * self.hidden_activation
                self.output_bias -= learning_rate * d_output

                self.hidden_weights -= learning_rate * np.outer(inputs, d_hidden)
                self.hidden_bias -= learning_rate * d_hidden

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss / len(y):.4f}")

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initialize and train the network
nn = NeuralNetwork()
nn.train(X, y, epochs=100000, learning_rate=0.1)

print("\nTesting XOR gate:")
for inputs in X:
    prediction = nn.forward(inputs)
    print(f"{inputs} => {prediction:.2f}")