from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 32, 2)
        self.cnn2 = nn.Conv2d(32, 32, 2)
        self.fc1 = nn.Linear(21632, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        logits = F.log_softmax(self.out(x), dim=1)
        return logits
