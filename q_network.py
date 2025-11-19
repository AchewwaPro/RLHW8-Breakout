import torch
from torch import nn

class QNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, lr):
        super(QNetwork, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    
    def inference(self, obs):
        q_value = self.seq(obs)
        return q_value
    
    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class CNN(nn.Module):
    def __init__(self, action_dim, lr):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def inference(self, obs):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        q_value = self.fc(x)
        return q_value
    
    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()