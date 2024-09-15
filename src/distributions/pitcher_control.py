import os

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import PitchControlDataset
from src.model.pitch_type import PitchType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PitcherControl(nn.Module):
    """
    This network is used to learn a pitcher's control, assumed to be a bivariate normal distribution.
    """

    def __init__(self):
        super(PitcherControl, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.conv_1 = nn.Conv2d(2 * len(PitchType), 64, 3)
        self.conv_2 = nn.Conv2d(64, 128, 3)

        self.linear_1 = nn.Linear(128, 128)

        # Concatenate the pitcher and pitch embeddings
        self.linear_2 = nn.Linear(134, 72)
        self.linear_3 = nn.Linear(72, 32)

        self.mu_x = nn.Linear(32, 1)
        self.mu_y = nn.Linear(32, 1)
        self.var_x = nn.Linear(32, 1)
        self.var_y = nn.Linear(32, 1)
        self.covar_xy = nn.Linear(32, 1)

    def forward(self, pitcher: Tensor, pitch: Tensor) -> Tensor:
        pitcher = self.dropout(pitcher)
        pitcher = F.relu(self.conv_1(pitcher))
        pitcher = F.relu(self.conv_2(pitcher))
        pitcher = pitcher.flatten(1)
        pitcher = F.relu(self.linear_1(pitcher))

        pitcher = torch.cat((pitcher, pitch), dim=1)
        pitcher = F.relu(self.linear_2(pitcher))
        pitcher = F.relu(self.linear_3(pitcher))

        mu_x = self.mu_x(pitcher)
        mu_y = self.mu_y(pitcher)
        var_x = F.softplus(self.var_x(pitcher))
        var_y = F.softplus(self.var_y(pitcher))
        covar_xy = self.covar_xy(pitcher)

        return torch.cat((mu_x, mu_y, var_x, var_y, covar_xy), dim=1)


def get_pitch_control_sets(data: BaseballData) -> (PitchControlDataset, PitchControlDataset):
    return PitchControlDataset.get_random_split(data, 0.3, seed=1)


def train(epochs: int = 400, batch_size: int = 5, learning_rate: float = 0.0001,
          path: str = '../../model_weights/pitcher_control.pth'):
    """Here is a standard training loop"""

    data = BaseballData()

    training_dataset, validation_dataset = get_pitch_control_sets(data)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = PitcherControl().to(device)

    print(f'Using device: {device}')

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.3)
    criterion = nn.MSELoss()

    loader_length = len(training_dataloader)

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for i, (obp, (pitcher, pitch_type), distribution) in tqdm(enumerate(training_dataloader), leave=True,
                                                                  total=loader_length,
                                                                  desc=f'Epoch {epoch + 1}'):
            pitcher, pitch_type, distribution = pitcher.to(device), pitch_type.to(device), distribution.to(device)

            optimizer.zero_grad()
            output = model.forward(pitcher, pitch_type)
            loss: Tensor = criterion(output, distribution)
            loss.backward()
            optimizer.step()
            training_loss += loss

            _, predicted = torch.max(output.data, 1)

        torch.save(model.state_dict(), path if path else 'model.pth')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (obp, (pitcher, pitch_type), distribution) in enumerate(validation_dataloader):
                pitcher, pitch_type, distribution = pitcher.to(device), pitch_type.to(device), distribution.to(device)

                output = model.forward(pitcher, pitch_type)
                total_loss += criterion(output, distribution)

            print(f'Epoch {epoch + 1}, '
                  f'training loss: {training_loss / len(training_dataloader)}, '
                  f'testing loss: {total_loss / len(validation_dataloader)}')

        scheduler.step()


if __name__ == '__main__':
    # This should converge to ~751 on the training set
    # There might be something wrong with this setup, as it doesn't seem to be learning too well
    # Might also be a lack of data
    train(batch_size=10, epochs=400, learning_rate=0.00005, path='../../model_weights/pitcher_control.pth')
