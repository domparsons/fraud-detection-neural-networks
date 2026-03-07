import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

sns.set_theme(style="white", context="notebook", font_scale=1.1)

df = pd.read_csv("./data/credit_card_transactions.csv")

num_of_non_fraudulent_transactions = len(df[df["Class"] == 0])
num_of_fraudulent_transactions = len(df[df["Class"] == 1])
percentage_of_fraudulent_transactions = num_of_fraudulent_transactions / len(df) * 100

print("Null values in dataset:", df.isnull().values.any())
print("Total transactions:", len(df))
print("Number of non-fraudulent transactions:", num_of_non_fraudulent_transactions)
print("Number of fraudulent transactions:", num_of_fraudulent_transactions)
print("Percentage of fraudulent transactions:", percentage_of_fraudulent_transactions)

print(df.head())
print(df.describe())

features = df.drop(["Class", "Time"], axis=1)
label = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.33, random_state=42, stratify=label
)

scaler = StandardScaler()
x_train[["Amount"]] = scaler.fit_transform(x_train[["Amount"]])
x_test[["Amount"]] = scaler.transform(x_test[["Amount"]])

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

X_train_tensor = torch.tensor(x_train.values).float().to(device)
X_test_tensor = torch.tensor(x_test.values).float().to(device)
y_train_tensor = torch.tensor(y_train.values).float().to(device)
y_test_tensor = torch.tensor(y_test.values).float().to(device)

# Increased from 4096 to better saturate MPS GPU
batch_size = 8192

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

# shuffle=True ensures different batch ordering each epoch, improving generalisation
# num_workers=0 is correct here since tensors are already on-device (MPS/GPU workers can't share GPU memory)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size)


# Batch processing function adapted from source [5]
def batch_processing(model, loss_function, x_batch, y_batch, optimiser=None):
    # Forward pass
    loss = loss_function(model(x_batch), y_batch.view(-1, 1))

    # If optimiser provided, we're training (not validating)
    if optimiser is not None:
        loss.backward()  # Compute gradients with backpropagation
        optimiser.step()  # Update weights using gradients
        optimiser.zero_grad()

    return loss.item(), len(x_batch)


# Training loop adapted from source [5]
# scheduler is optional — if provided, it steps on validation loss each epoch
def train(
    epochs,
    model,
    loss_function,
    optimiser,
    train_data_loader,
    validation_data_loader,
    scheduler=None,
):
    val_losses = []
    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # Process batches using vectorised operations
        for x_batch, y_batch in train_data_loader:
            # Compute loss, compute gradients, update weights, clear gradients
            batch_processing(model, loss_function, x_batch, y_batch, optimiser)

        # Validation phase
        model.eval()

        # Disable gradient computation to reduce computation
        with torch.no_grad():
            batch_results = [
                batch_processing(model, loss_function, xb, yb)
                for xb, yb in validation_data_loader
            ]
            losses = [r[0] for r in batch_results]
            nums = [r[1] for r in batch_results]

        # Compute weighted average loss across all validation batches
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)
        val_losses.append(val_loss)

        # Decay learning rate if validation loss has plateaued
        if scheduler is not None:
            scheduler.step(val_loss)

    return val_losses


# Network structure inspired by source [6]
class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_layer_size, use_sigmoid=False):
        # Calls the parent class (nn.Module) constructor registering model with PyTorch's module system so it can track parameters, gradients, and device placement
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dimension, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


def evaluate_model(model, data_loader, threshold=0.5, apply_sigmoid=True):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in data_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits) if apply_sigmoid else logits
            preds = (probs > threshold).float()

            all_predictions.extend(preds.cpu().numpy().flatten())
            all_labels.extend(yb.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": all_predictions,
    }


def print_evaluation(config_name, metrics):
    print(f"\n{'=' * 50}")
    print(config_name)
    print(f"{'=' * 50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Non-Fraud  Fraud")
    print(
        f"Actual Non-Fraud {metrics['confusion_matrix'][0][0]:>6}    {metrics['confusion_matrix'][0][1]:>6}"
    )
    print(
        f"Actual Fraud     {metrics['confusion_matrix'][1][0]:>6}    {metrics['confusion_matrix'][1][1]:>6}"
    )
    print(f"{'=' * 50}\n")


epochs = 30
hidden_layer_size = 128
torch.manual_seed(42)
model = NeuralNetwork(X_train_tensor.shape[1], hidden_layer_size, use_sigmoid=False).to(
    device
)

optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

# torch.compile() JIT-compiles the model graph for ~20-40% faster training on MPS/CUDA
# Called after optimiser creation since compile() changes the return type
model = torch.compile(model)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, patience=5, factor=0.5
)

pos_weight = torch.tensor([20.0]).to(device)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

losses = train(
    epochs,
    model,
    loss_function,
    optimiser,
    train_data_loader,
    validation_data_loader,
    scheduler,
)

metrics = evaluate_model(model, validation_data_loader, threshold=0.25)
print_evaluation("BCEWithLogitsLoss (Optimised)", metrics)
