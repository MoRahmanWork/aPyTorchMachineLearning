import torch
from torch import nn

# Check if PyTorch is installed
print("PyTorch version:", torch.__version__)

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# If CUDA is available, print GPU details
if cuda_available:
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("Running on CPU.")


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
    y = x * 2

@torch.no_grad()
def doubler(x):
    return x * 2
z = doubler(x)


@torch.no_grad()
def tripler(x):
    return x * 3
z = tripler(x)


# factory function exception
with torch.no_grad():
    a = torch.nn.Parameter(torch.rand(10))
a.requires_grad