import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, units, activation):
        """
        Initializes the MLP model.

        Parameters:
        - input_dim: An integer representing the number of input features.
        - output_dim: An integer representing the number of output features.
        - hidden_units: A list of integers where each integer represents the number of units in a hidden layer.
        - activation: A string representing the activation function to be used after each layer.
        """
        super(MLP, self).__init__()
        
        layers = []
        activation_fn = self._get_activation(activation)

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_dim, units[0]))
        if activation_fn is not None:
            layers.append(activation_fn)

        # Hidden layers
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i+1]))
            if activation_fn is not None:
                layers.append(activation_fn)

        # Last hidden layer to output layer
        layers.append(nn.Linear(units[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def _get_activation(self, activation):
        """Returns the activation function based on the string input."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "silu":
            return nn.SiLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "none":
            return None
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        return self.model(x)

def main():
    input_dim = 10  # Number of input features
    output_dim = 5  # Number of output features
    hidden_units = [20, 30, 40]  # Example: 20 units -> 30 units -> 40 units in hidden layers
    activation = "silu"  # Single activation function for all layers

    model = MLP(input_dim, output_dim, hidden_units, activation)
    print(model)

if __name__ == "__main__":
    main()