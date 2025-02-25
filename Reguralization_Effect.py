import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Synthetic Data
n_samples = 100
input_dim = 1
output_dim = 1

X_clean = torch.linspace(0, 10, n_samples).unsqueeze(1)  # [n_samples, 1]
y_clean = 2 * X_clean + 1 + torch.randn(n_samples, 1) * 0.1 
# Noisy Data
X_test = torch.linspace(0, 10, n_samples).unsqueeze(1)
noise_level = 1.0
y_test = 2 * X_test + 1 + torch.randn(n_samples, 1) * noise_level  
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model_regularized = LinearRegression(input_dim, output_dim)
model_no_regularization = LinearRegression(input_dim, output_dim)

optimizer_regularized = optim.SGD(model_regularized.parameters(), lr=0.01, weight_decay=0.01)  # lambda -> weight decay
optimizer_no_regularization = optim.SGD(model_no_regularization.parameters(), lr=0.01)

# Loss Function (Mean Squared Error)
criterion = nn.MSELoss()

# 4. Training Loop
n_epochs = 100

def train(model, optimizer, X, y, criterion):
    model.train()
    for epoch in range(n_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

model_regularized = train(model_regularized, optimizer_regularized, X_clean, y_clean, criterion)
model_no_regularization = train(model_no_regularization, optimizer_no_regularization, X_clean, y_clean, criterion)

model_regularized.eval()
model_no_regularization.eval()

with torch.no_grad():
    y_pred_regularized = model_regularized(X_test)
    y_pred_no_regularization = model_no_regularization(X_test)

loss_regularized = criterion(y_pred_regularized, y_test)
loss_no_regularization = criterion(y_pred_no_regularization, y_test)

print(f"Loss (Regularized): {loss_regularized.item()}")
print(f"Loss (No Regularization): {loss_no_regularization.item()}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test.numpy(), y_test.numpy(), label="Data Test (noisy)")
plt.plot(X_test.numpy(), y_pred_regularized.numpy(), label="Regularized Prediction", color="red")
plt.plot(X_test.numpy(), y_pred_no_regularization.numpy(), label="No Regularization Prediction", color="green")
plt.legend()
plt.title("Regularization vs. No Regularization on Noisy Data")
plt.show()
