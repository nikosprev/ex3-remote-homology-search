import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader




class NpzDataset(Dataset):
    def __init__(self, X, y):
        # 1. FLATTEN INPUT (for mnist)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
            
      
        # X must be Float (for weights). 
        # y must be Long (for CrossEntropyLoss).
        self.X = torch.from_numpy(X.copy()).float()
        self.y = torch.from_numpy(y.copy()).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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


# Training function
def train_mlp(data, labels, input_dim, num_classes, nodes, hidden_layers_number, batch_size, lr, epochs, device='cpu'):
    dataset = NpzDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(
        input_size=input_dim,
        hidden_layers_number=hidden_layers_number,
        hidden_layer_size=nodes,
        output_size=num_classes,
        dropout=0.2,
        activation="tanh"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[Training] Model: MLP")
    print(f"[Training] Device: {device}")
    print(f"[Training] Input Dim: {input_dim}")
    print(f"[Training] Hidden Layers: {hidden_layers_number}, Nodes per Layer: {nodes}, LR: {lr}, Batch: {batch_size}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    print("Training finished.")
    return model



import torch
import os 
from typing import Dict, Any

def save_model(model: MLP, args_dict: Dict[str, Any], input_dim: int, num_classes: int, index_path: str):
    """
    Saves the trained MLP model's state dictionary and architectural parameters 
    to a file in the checkpoint format expected by load_model.

    Args:
        model: The trained MLP instance.
        args_dict: Dictionary containing the original command-line arguments.
        input_dim: The actual input dimension used for training (e.g., 784 for MNIST).
        num_classes: The number of output classes (KaHIP partitions, 'm').
        index_path: The folder path where the index and model should be saved.
    """
    
    # 1. Define the architecture dictionary using parameters from args_dict
    arch_dict = {
        "input_size": input_dim,
        "hidden_layers_number": args_dict.get("layers", 3),
        "hidden_layer_size": args_dict.get("nodes", 64),
        "output_size": num_classes,
        "dropout": 0.1, 
        "activation": "relu",
    }   
    kahip_str = {0: "FAST", 1: "ECO", 2: "STRONG"}.get(args_dict.get("kahip_mode"), "UNKNOWN")
    nblocks = args_dict.get("m")
    model_filename = f"{index_path}/{args_dict.get('type')}_m{nblocks}_knn{args_dict.get("knn")}_{kahip_str}.pth"
    os.makedirs(index_path, exist_ok=True)
    
    checkpoint = {
        "architecture": arch_dict,
        "state_dict": model.state_dict()
    }

    # 4. Save the checkpoint
    torch.save(checkpoint, model_filename)
    print(f"\n[SAVE] Trained MLP model and architecture saved to: {model_filename}")





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
    
    # 1. Extract architectural parameters from the checkpoint
    # We use .get() to safely pull parameters and use the MLP's defaults 
    # if the parameter wasn't explicitly saved (though it should be).
    
    input_size = arch.get("input_size")
    if input_size is None:
        raise ValueError(f"Checkpoint at {model_path} is missing 'input_size' in the architecture dictionary.")

    hidden_layers_number = arch.get("hidden_layers_number", 3)
    hidden_layer_size = arch.get("hidden_layer_size", 64)
    output_size = arch.get("output_size")
    dropout = arch.get("dropout", 0.2)
    activation = arch.get("activation", "relu")


    # 2. Instantiate the MLP model with the loaded architecture parameters
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
    
    # 3. Load the trained weights
    model.load_state_dict(checkpoint["state_dict"])
    
    model.to(device)
    model.eval() 

    print(f"[LOAD] Model successfully loaded and mapped to device: {device}")
    return model