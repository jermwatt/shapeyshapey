from torch import nn


class Encoder(nn.Module):

    def __init__(self,
                 width: int,
                 height: int):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(width, height)
        )

    def forward(self, x):
        return self.net(x)
