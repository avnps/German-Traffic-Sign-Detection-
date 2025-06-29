import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import TrafficSignNet

# Load test dataset
with open("test.p", "rb") as f:
    test_data = pickle.load(f)

X_test = torch.tensor(test_data['features']).permute(0, 3, 1, 2).float() / 255.0
X_test = (X_test - 0.5) / 0.5  # Normalize to [-1, 1]
y_test = torch.tensor(test_data['labels'])

test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=64)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignNet().to(device)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=device))
model.eval()

# Evaluate
criterion = nn.CrossEntropyLoss()
total_loss, correct, total = 0, 0, 0

with torch.no_grad():
    for X_batch, y_batch in test_dl:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test Loss={total_loss/total:.4f}, Accuracy={correct/total:.4f}")
