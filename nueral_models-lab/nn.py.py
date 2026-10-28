import torch
import torch.nn as nn
import torch.optim as optim

# Dataset definition for XOR
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_binary = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

print("==================================================")
print("TASK 4A & 4B: BASIC LEARNING & BACKPROPAGATION")
print("==================================================")
torch.manual_seed(42)
model_basic = nn.Sequential(
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 1)
)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model_basic.parameters(), lr=0.5)

# Record initial loss
initial_loss = criterion(model_basic(X), y_binary)
print(f"Initial Loss: {initial_loss.item():.4f}")

for step in range(2000):
    optimizer.zero_grad()
    loss = criterion(model_basic(X), y_binary)
    loss.backward()
    optimizer.step()

# Re-compute for final metrics
model_basic.zero_grad()
final_loss = criterion(model_basic(X), y_binary)
final_loss.backward()
probs = torch.sigmoid(model_basic(X))
preds = (probs > 0.5).float()

print(f"Final Loss: {final_loss.item():.4f}")
print(f"Final Probabilities:\n{probs.detach()}")
print(f"Final Predictions:\n{preds.detach()}")
print(f"Layer 1 Gradients (dL/dW1):\n{model_basic[0].weight.grad}\n")


print("==================================================")
print("TASK 4C: SYMMETRY EXPERIMENT (ZERO INITIALIZATION)")
print("==================================================")
model_sym = nn.Sequential(
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 1)
)
# Force identical/zero weights
nn.init.zeros_(model_sym[0].weight)
nn.init.zeros_(model_sym[0].bias)
nn.init.zeros_(model_sym[2].weight)
nn.init.zeros_(model_sym[2].bias)

optimizer_sym = optim.SGD(model_sym.parameters(), lr=0.5)

for step in range(10):
    optimizer_sym.zero_grad()
    loss = criterion(model_sym(X), y_binary)
    loss.backward()
    optimizer_sym.step()

print(f"Hidden Layer Weights after 10 steps:\n{model_sym[0].weight.detach()}")
print("Observation: The rows remain identical because identical units receive identical gradients.\n")


print("==================================================")
print("TASK 4D: ACTIVATION EXPERIMENT")
print("==================================================")
activations = {
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "ReLU": nn.ReLU()
}

print(f"{'Activation':<10} | {'Final Loss':<10} | {'4/4 Correct?':<12} | {'Early Grad Norm':<15}")
print("-" * 55)

for name, act in activations.items():
    torch.manual_seed(42) # Ensure same starting weights for fair comparison
    model_act = nn.Sequential(
        nn.Linear(2, 2),
        act,
        nn.Linear(2, 1)
    )
    opt = optim.SGD(model_act.parameters(), lr=0.5)
    
    # Get early gradient norm (Step 1)
    opt.zero_grad()
    loss = criterion(model_act(X), y_binary)
    loss.backward()
    early_grad_norm = torch.norm(model_act[0].weight.grad).item()
    opt.step()
    
    # Train remaining steps
    for step in range(1, 2000):
        opt.zero_grad()
        loss = criterion(model_act(X), y_binary)
        loss.backward()
        opt.step()
        
    final_probs = torch.sigmoid(model_act(X))
    correct = bool(torch.all((final_probs > 0.5).float() == y_binary).item())
    final_loss_val = criterion(model_act(X), y_binary).item()
    
    print(f"{name:<10} | {final_loss_val:<10.4f} | {str(correct):<12} | {early_grad_norm:<15.4f}")
print("\n")


print("==================================================")
print("TASK 5: THREE-CLASS EXTENSION & SOFTMAX STABILITY")
print("==================================================")
y_multi = torch.tensor([0, 1, 1, 2]) # 0: both inactive, 1: disagree, 2: both active

torch.manual_seed(42)
model_multi = nn.Sequential(
    nn.Linear(2, 2),
    nn.Tanh(),
    nn.Linear(2, 3) # 3 logits output
)
criterion_multi = nn.CrossEntropyLoss()
optimizer_multi = optim.SGD(model_multi.parameters(), lr=0.5)

for step in range(2000):
    optimizer_multi.zero_grad()
    loss = criterion_multi(model_multi(X), y_multi)
    loss.backward()
    optimizer_multi.step()

final_logits = model_multi(X)
multi_probs = torch.softmax(final_logits, dim=1)
preds_multi = torch.argmax(multi_probs, dim=1)

print(f"Final Predictions:\n{preds_multi.detach()}")
print(f"Probabilities:\n{multi_probs.detach()}")
print(f"Sum of probabilities for example 1: {multi_probs[0].sum().item():.4f}")

# Optional diagnostic: Add 100 to logits
shifted_logits = final_logits + 100.0
shifted_probs = torch.softmax(shifted_logits, dim=1)
print(f"\nProbabilities after adding 100 to logits:\n{shifted_probs.detach()}")
print("Observation: Softmax is translation-invariant. Stable implementations subtract the max logit to prevent exponential overflow.")