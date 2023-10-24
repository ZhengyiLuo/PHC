import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=4)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=4, stride=4)
        self.fc = nn.Linear(144, out_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.fc(x.view(x.size(0), -1))
        return x


if __name__ == '__main__':
    import time
    torch.set_grad_enabled(False)
    net = SimpleCNN(128)
    t0 = time.time()
    input = torch.zeros(1, 3, 224, 224)
    out = net(input)
    print(time.time() - t0)
    print(out.shape)
