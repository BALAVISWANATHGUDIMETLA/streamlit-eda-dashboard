import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

# Generate dummy time series-like data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 4 + 3 * X + np.random.randn(*X.shape)

# Convert to sequences
def create_sequences(X, y, seq_len=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y)

# Convert to tensors
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

# Define LSTM + Self-Attention module
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)  # final output (warm-start value)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output, attn_weights

# Initialize and train
model = LSTMWithAttention(input_size=1, hidden_size=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
log_file = open("lstm_warm_start_log.txt", "w")
log_file.write("Epoch\tLoss\n")

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs, _ = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    log_file.write(f"{epoch+1}\t{loss.item():.4f}\n")

log_file.close()
print("\nTraining complete. Log saved to 'lstm_warm_start_log.txt'")