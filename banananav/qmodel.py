import torch
import torch.nn as nn
import torch.nn.functional as F

class BananaQModel(nn.Module):
    """Q Function Approximator with autoencode-like step."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Args:
            state_size: Dimension of each state (int)
            action_size: number of actions (int)
        """
        super(BananaQModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return self.fc5(x)

class SimpleBananaQModel(nn.Module):
    """Q Function Approximator."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Args:
            state_size: Dimension of each state (int)
            action_size: number of actions (int)
        """
        super(SimpleBananaQModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
