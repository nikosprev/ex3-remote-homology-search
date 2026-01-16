import torch
import torch.nn as nn




class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers_number, hidden_layer_size, output_size, dropout=0.1, activation="relu"):
        super().__init__()

        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.hidden_layers = nn.ModuleList()
        in_features = input_size
        for _ in range(hidden_layers_number):
            self.hidden_layers.append(nn.Linear(in_features, hidden_layer_size))
            in_features = hidden_layer_size

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(self.dropout(layer(x)))
        x = self.fc_out(x)
        return x


def load_model(model_path: str, device: str = 'cpu'):
    """
    Loads a trained MLP model and its architecture from a checkpoint file.

    Args:
        model_path (str): Path to the saved .pth checkpoint file.
        device (str): Device to map the loaded model to ('cpu' or 'cuda').

    Returns:
        MLP: The instantiated and loaded MLP model, set to evaluation mode.
    """
    
    # Load the checkpoint dictionary, mapping to the specified device
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    arch = checkpoint.get("architecture", {})
    
    input_size = arch.get("input_size")
    if input_size is None:
        raise ValueError(f"Checkpoint at {model_path} is missing 'input_size' in the architecture dictionary.")

    hidden_layers_number = arch.get("hidden_layers_number", 3)
    hidden_layer_size = arch.get("hidden_layer_size", 64)
    output_size = arch.get("output_size")
    dropout = arch.get("dropout", 0.2)
    activation = arch.get("activation", "relu")

    print(f"\n[LOAD] Instantiating MLP model with architecture:")
    print(f"       Input Dim: {input_size}, Layers: {hidden_layers_number}, Nodes: {hidden_layer_size}, Output Classes: {output_size}")

    model = MLP(
        input_size=input_size,
        hidden_layers_number=hidden_layers_number,
        hidden_layer_size=hidden_layer_size,
        output_size=output_size,
        dropout=dropout,
        activation=activation
    )
    
    model.load_state_dict(checkpoint["state_dict"])
    
    model.to(device)
    model.eval() 

    print(f"[LOAD] Model successfully loaded and mapped to device: {device}")
    return model