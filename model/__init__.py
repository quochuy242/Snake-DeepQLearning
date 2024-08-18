import os

import torch
from torch import nn, optim
from pathlib import Path
from utils import float_tensor, unsqueeze_0


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)

    def save(self, weight_name: str, save_dir: Path = Path("weights")):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), str(save_dir / weight_name))


class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # TODO: Adjust optimizer and criterion to be able to change
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state, action, reward, next_state = map(
            float_tensor, [state, action, reward, next_state]
        )

        if len(state.shape) == 1:  # (1, x)
            state, action, reward, next_state = map(
                unsqueeze_0, [state, action, reward, next_state]
            )
            done = (done,)

        # Predict Q Values with current state
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # // Calculate Q_new = r + y * max(next_Q_values) and only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
