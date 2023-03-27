import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
#from dataset_handler import DatasetHandler
import json

def save_results_to_json(best_models, val_set, criterion, device, filename='results.json'):
    results = []

    for model in best_models:
        model = model.to(device)
        validation_dataloader = DataLoader(TensorDataset(torch.Tensor(val_set)), batch_size=32, shuffle=False)
        validation_loss = evaluate_model(model, validation_dataloader, criterion, device)

        model_info = {
            'parameter_size': sum(p.numel() for p in model.parameters()),
            'generalization_error': float(validation_loss),
            'hyperparameters': model.get_hyperparameters()
        }
        results.append(model_info)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, hidden_size):
        super(GPTModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_layers,
            hidden_size,
        )
        self.fc = nn.Linear(d_model, vocab_size)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

    def get_hyperparameters(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
        }

SOTAModels_ = {"GPTModel": GPTModel}

def SOTA_models(model_name, **kwargs):
    return SOTAModels_[model_name](**kwargs)

def create_shards(data, num_shards, ratio):
    shard_sizes = [int(len(data) * ratio**i) for i in range(num_shards)]
    shards = []
    for i in range(num_shards):
        indices = np.random.choice(len(data), shard_sizes[i], replace=False)
        shards.append(data[indices])
    return shards

def create_data_splits(data, num_shards, ratio, val_size):
    shards = create_shards(data, num_shards, ratio)
    val_set, _ = random_split(data, [val_size, len(data) - val_size])
    return shards, val_set

def generate_model_candidates(base_model, constraint_func):
    model_candidates = []
    # Generate model candidates by constraining model capacity and changing hyperparameters
    for hyperparameters in constraint_func(base_model):
        model = SOTA_models(base_model, **hyperparameters)
        model_candidates.append(model)
    return model_candidates

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch in dataloader:
        batch = batch[0].to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)
            output = model(batch)
            loss = criterion(output, batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def monte_carlo_grid_search(shards, val_set, base_model, constraint_func, criterion, device, num_trials=3):
    best_models = []
    model_candidates = generate_model_candidates(base_model, constraint_func)

    for shard in shards:
        best_model = None
        best_validation_loss = float('inf')
        shard_tensor = torch.Tensor(shard)

        for model_candidate in model_candidates:
            model = model_candidate.to(device)
            optimizer = optim.Adam(model.parameters(), weight_decay=0)  # Remove regularization (weight decay)
            dataloader = DataLoader(TensorDataset(shard_tensor), batch_size=32, shuffle=True)

            for trial in range(num_trials):
                train_model(model, dataloader, criterion, optimizer, device)

                validation_dataloader = DataLoader(TensorDataset(torch.Tensor(val_set)), batch_size=32, shuffle=False)
                validation_loss = evaluate_model(model, validation_dataloader, criterion, device)

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model = model_candidate

        best_models.append(best_model)

    return best_models

# Define device, loss function, and other settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

# Define your data and the DatasetHandler
data = np.random.rand(100000, 100)  # Replace this with your data
num_shards = 6
shard_ratio = 2
val_size = 10000
shards, val_set = create_data_splits(data, num_shards, shard_ratio, val_size)

# Define base_model and constraint_func
base_model = "GPTModel"  # Replace with your base model
v_space = [1000, 2000, 3000, 4000, 5000] # vocab_size
d_space = [32, 64, 128, 256, 512] # dimension_model
h_space = [2, 4, 8, 16, 32] # number of heads 
l_space = [2, 4, 6, 8, 10] # number of layers
hs_space = [32, 64, 128, 256, 512] # hidden size

constraint_func = lambda base_model: [{'vocab_size': v, 'd_model': d, 'nhead': n_h, 'num_layers': n_l, 'hidden_size': h} for v in v_space for d in d_space for n_h in h_space for n_l in l_space for h in hs_space]  # Replace with appropriate constraints


# Perform Monte Carlo grid search
best_models = monte_carlo_grid_search(shards, val_set, base_model, constraint_func, criterion, device)

# Save results
save_results_to_json(best_models, val_set, criterion, device)
