import os

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import SwingResult, PitchDataset
from src.model.pitch import PitchType, Pitch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwingOutcome(nn.Module):
    """
    This network learns the distribution of a swing based on the pitcher, batter, pitch, and relevant state.
    Note that inference requires softmax=True to get the expected probability distribution.
    """

    def __init__(self):
        super(SwingOutcome, self).__init__()

        assert len(PitchType) == 6  # Edit this line if you add more pitch types

        self.p_dropout_1 = nn.Dropout(0.2)
        self.p_conv_1 = nn.Conv2d(2 * len(PitchType), 64, 3)
        self.p_conv_2 = nn.Conv2d(64, 128, 3)
        self.p_linear = nn.Linear(128, 128)
        self.p_dropout_2 = nn.Dropout(0.25)

        self.b_dropout_1 = nn.Dropout(0.2)
        self.b_conv_1 = nn.Conv2d(2 * len(PitchType), 64, 3)
        self.b_conv_2 = nn.Conv2d(64, 128, 3)
        self.b_linear = nn.Linear(128, 128)
        self.b_dropout_2 = nn.Dropout(0.25)

        self.pitch_conv_1 = nn.Conv2d(len(PitchType), 32, 3)
        self.pitch_conv_2 = nn.Conv2d(32, 64, 3)
        self.pitch_linear = nn.Linear(64, 64)

        self.linear_1 = nn.Linear(128 + 128 + 64 + 7, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, len(SwingResult))

    def forward(self, pitcher: Tensor, batter: Tensor, pitch: Tensor, strikes: Tensor, balls: Tensor,
                num_runs: Tensor, num_outs: Tensor, batter_on_first: Tensor, second: Tensor, third: Tensor,
                softmax: bool = False) -> Tensor:
        pitcher = self.p_dropout_1(pitcher)
        pitcher = F.relu(self.p_conv_1(pitcher))
        pitcher = F.relu(self.p_conv_2(pitcher))
        pitcher = pitcher.flatten(1)
        pitcher = F.relu(self.p_linear(pitcher))
        pitcher = self.p_dropout_2(pitcher)

        batter = self.b_dropout_1(batter)
        batter = F.relu(self.b_conv_1(batter))
        batter = F.relu(self.b_conv_2(batter))
        batter = batter.flatten(1)
        batter = F.relu(self.b_linear(batter))
        batter = self.b_dropout_2(batter)

        pitch = F.relu(self.pitch_conv_1(pitch))
        pitch = F.relu(self.pitch_conv_2(pitch))
        pitch = pitch.flatten(1)
        pitch = F.relu(self.pitch_linear(pitch))

        strikes = strikes.unsqueeze(1)
        balls = balls.unsqueeze(1)
        num_runs = num_runs.unsqueeze(1)
        num_outs = num_outs.unsqueeze(1)
        batter_on_first = batter_on_first.unsqueeze(1)
        second = second.unsqueeze(1)
        third = third.unsqueeze(1)

        output = torch.cat((pitcher, batter, pitch, strikes, balls,
                            num_runs, num_outs, batter_on_first, second, third), dim=1)
        output = F.relu(self.linear_1(output))
        output = F.relu(self.linear_2(output))
        output = F.relu(self.linear_3(output))
        output = self.output(output)

        if softmax:
            output = F.softmax(output, dim=1)

        return output


def map_swing_outcome(idx: int, pitch: Pitch, bd: BaseballData):
    """We map the pitch to the relevant data and target for training. We include the index for utility purposes."""

    return (idx, (bd.pitchers[pitch.pitcher_id].data, bd.batters[pitch.batter_id].data,
                  pitch.get_one_hot_encoding(),
                  torch.tensor(pitch.game_state.strikes, dtype=torch.float32),
                  torch.tensor(pitch.game_state.balls, dtype=torch.float32),
                  torch.tensor(pitch.game_state.num_runs, dtype=torch.float32),
                  torch.tensor(pitch.game_state.num_outs, dtype=torch.float32),
                  torch.tensor(pitch.game_state.first, dtype=torch.float32),
                  torch.tensor(pitch.game_state.second, dtype=torch.float32),
                  torch.tensor(pitch.game_state.third, dtype=torch.float32)),
            SwingResult.from_pitch_result(pitch.result).get_one_hot_encoding())


def get_swing_outcome_dataset(data: BaseballData) -> [PitchDataset, PitchDataset]:
    return PitchDataset.get_split_on_attribute(
        data, 0.2,
        attribute=lambda p: data.pitchers[p.pitcher_id],  # Group by pitcher
        filter_on=lambda p: p.result.batter_swung(),
        map_to=lambda idx, pitch: map_swing_outcome(idx, pitch, data),
        seed=81
    )


def train(epochs: int = 30, batch_size: int = 128, learning_rate: float = 0.0003,
          path: str = '../../model_weights/swing_outcome.pth'):
    """Here is a standard training loop"""

    data = BaseballData()
    training_set, testing_set = get_swing_outcome_dataset(data)

    training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_set, batch_size=batch_size)

    # If your system supports Triton, Torch 2.0 has a compile method that can speed up the model
    # compiled_model = torch.compile(SwingOutcome())
    # model = compiled_model.to(device)
    model = SwingOutcome().to(device)

    print(f'Using device: {device}')

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    criterion = nn.CrossEntropyLoss()

    loader_length = len(training_dataloader)

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for i, (pitch_idx, data, target) in tqdm(enumerate(training_dataloader), leave=True,
                                                 total=loader_length, desc=f'Epoch {epoch + 1}'):
            data = [d.to(device) for d in data]
            target: Tensor = target.to(device)

            optimizer.zero_grad()
            output = model.forward(*data)
            loss: Tensor = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss

            _, predicted = torch.max(output.data, 1)

        torch.save(model.state_dict(), path if path else 'model.pth')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (pitch_idx, data, result) in enumerate(testing_dataloader):
                data = [d.to(device) for d in data]
                result: Tensor = result.to(device)

                output = model(*data)
                total_loss += criterion(output, result)

            print(f'Epoch {epoch + 1}, '
                  f'training loss: {training_loss / len(training_dataloader)}, '
                  f'testing loss: {total_loss / len(testing_dataloader)}')

        scheduler.step()


if __name__ == '__main__':
    # Should converge to ~1.341 testing loss
    train(epochs=25, learning_rate=0.0003, batch_size=128, path=f'../../model_weights/swing_outcome.pth')
