import torch
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
torch_device = torch.device("cpu")

class DAN(torch.nn.Module):
    def __init__(self, input_size):
        super(DAN, self).__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(in_features=input_size, out_features=2)

    def forward(self, x):
        return (self.linear(torch.tensor(x)))