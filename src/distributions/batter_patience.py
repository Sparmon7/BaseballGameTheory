import os

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import PitchDataset
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BatterSwings(nn.Module):
    """
    This network learns the swing outcome based on the batter, pitch, and relevant state.
    It's intended to be trained with borderline pitches.
    """

    def __init__(self):
        super(BatterSwings, self).__init__()

        assert len(PitchType) == 6  # Edit this line if you add more pitch types and verify the architecture

        self.b_dropout_1 = nn.Dropout(0.2)
        self.b_conv_1 = nn.Conv2d(2 * len(PitchType), 64, 3)
        self.b_conv_2 = nn.Conv2d(64, 128, 3)
        self.b_linear = nn.Linear(128, 128)
        self.b_dropout_2 = nn.Dropout(0.25)

        self.pitch_conv_1 = nn.Conv2d(len(PitchType), 32, 3)
        self.pitch_conv_2 = nn.Conv2d(32, 64, 3)
        self.pitch_linear = nn.Linear(64, 64)

        self.linear_1 = nn.Linear(128 + 64 + 7, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 32)

        self.output = nn.Linear(32, 1)

    def forward(self, batter: Tensor, pitch: Tensor, strikes: Tensor, balls: Tensor, num_runs: Tensor,
                num_outs: Tensor, batter_on_first: Tensor, second: Tensor, third: Tensor) -> Tensor:
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

        output = torch.cat((batter, pitch, strikes, balls, num_runs,
                            num_outs, batter_on_first, second, third), dim=1)
        output = F.relu(self.linear_1(output))
        output = F.relu(self.linear_2(output))
        output = F.relu(self.linear_3(output))
        output = torch.sigmoid(self.output(output))

        return output


def batter_patience_map(bd: BaseballData, pitch_idx: int, pitch: Pitch) -> (int, (Tensor, Tensor, Tensor, Tensor), Tensor):
    """
    We map the pitch to the batter, pitch, and game state data, and the swing outcome. The pitch index
    is included for utility purposes in other contexts, since this map is also used manually outside this class.
    """

    return (pitch_idx, (bd.batters[pitch.batter_id].data, pitch.get_one_hot_encoding(),
                        torch.tensor(pitch.game_state.strikes, dtype=torch.float32),
                        torch.tensor(pitch.game_state.balls, dtype=torch.float32),
                        torch.tensor(pitch.game_state.num_runs, dtype=torch.float32),
                        torch.tensor(pitch.game_state.num_outs, dtype=torch.float32),
                        torch.tensor(int(pitch.game_state.first), dtype=torch.float32),
                        torch.tensor(int(pitch.game_state.second), dtype=torch.float32),
                        torch.tensor(int(pitch.game_state.third), dtype=torch.float32)),
            torch.tensor(int(pitch.result.batter_swung()), dtype=torch.float32))


def get_batter_patience_set(data: BaseballData) -> (PitchDataset, PitchDataset):
    return PitchDataset.get_split_on_attribute(
        data, 0.2,
        attribute=lambda p: data.batters[p.batter_id],
        filter_on=lambda p: p.is_borderline(),
        map_to=lambda idx, p: batter_patience_map(data, idx, p),
        seed=0
    )


def train(epochs: int = 50, batch_size: int = 512, learning_rate: float = 0.001,
          path: str = '../../model_weights/batter_patience.pth'):
    """Here is a standard training loop"""

    data = BaseballData()

    training_dataset, validation_dataset = get_batter_patience_set(data)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = BatterSwings().to(device)

    print(f'Using device: {device}')

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)
    criterion = nn.BCELoss()

    loader_length = len(training_dataloader)

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for pitch_idx, data, swing in tqdm(training_dataloader, leave=True, total=loader_length, desc=f'Epoch {epoch + 1}'):
            data = [d.to(device) for d in data]
            swing = swing.to(device)
            optimizer.zero_grad()
            output = model.forward(*data)
            loss: Tensor = criterion(output, swing.unsqueeze(1))

            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        torch.save(model.state_dict(), path if path else 'model.pth')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for pitch_idx, data, swing in validation_dataloader:
                data = [d.to(device) for d in data]
                swing = swing.to(device)
                output = model.forward(*data)
                loss = criterion(output, swing.unsqueeze(1))

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}, '
                  f'training loss: {training_loss / len(training_dataloader)}, '
                  f'validation loss: {total_loss / len(validation_dataloader)}')

        scheduler.step()


if __name__ == '__main__':
    # This should achieve a testing loss of ~0.564
    train(epochs=30, batch_size=512, learning_rate=0.0003)
