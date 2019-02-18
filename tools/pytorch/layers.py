from torch import nn


class Flatten(nn.Module):
    def __init__(self, ignore_batch=True):
        super().__init__()
        self.ignb = ignore_batch

    def forward(self, input):
        if not self.ignb:
            return input.view(input.shape[0], -1)
        return input.view(-1)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*[input.shape[0], *self.shape])
        