import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the target function f(x,y) for the union of lemniscate and ellipse
# f(x,y) = ((x^2 + y^2)^2 - 4*(x^2 - y^2)) * ((x - 0.5)**2 + 4*(y - 1/3)**2 - 2)
def f(x, y):
    return ((x**2 + y**2)**2 - 4*(x**2 - y**2)) * ((x - 0.5)**2 + 4*(y - 1/3)**2 - 2)

# Label: red zone (1) if f(x,y) <= 0, blue zone (0) otherwise
def label_zone(x, y):
    return (f(x, y) <= 0).astype(np.float32)

# Create a grid of points in the rectangle [-2,2] x [-1,1]
n = 200
x_vals = np.linspace(-2, 2, n)
y_vals = np.linspace(-1, 1, n)
xx, yy = np.meshgrid(x_vals, y_vals)
X = np.vstack([xx.ravel(), yy.ravel()]).T

y_true = label_zone(X[:, 0], X[:, 1]).reshape(-1, 1)

# Convert to PyTorch tensors and send to device
device = torch.device('cpu')
X_tensor = torch.from_numpy(X).float().to(device)
y_tensor = torch.from_numpy(y_true).float().to(device)

# Neural network parameters
p = 50   # neurons per hidden layer -> 3*p = 60 neurons total
batch_size = 256
epochs = 500
lr = 0.1

# Define a 4-layer network: 2->p->p->p->1 with ReLU activations
class Net(nn.Module):
    def __init__(self, p):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, p),
            nn.ReLU(),
            nn.Linear(p, p),
            nn.ReLU(),
            nn.Linear(p, p),
            nn.ReLU(),
            nn.Linear(p, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = Net(p).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs} - Loss: {running_loss/len(dataset):.4f}")

# Evaluate on the grid and reshape for plotting
model.eval()
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy().reshape(n, n)

# Plot classification result
plt.figure(figsize=(8, 4)) # Red where preds >= 0.5, Blue where preds < 0.5
plt.pcolormesh(xx, yy, preds >= 0.5, shading='auto', cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Classification (p={p}, total hidden neurons={3*p}, epochs={epochs})')
plt.tight_layout()
plt.savefig('classification_result2.png')