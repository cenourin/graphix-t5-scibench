"""Port of the part of seq2seq/models/model_utils.py the RGAT layer uses: FFN, unchanged."""
import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, input_size):
        super(FFN, self).__init__()
        self.input_size = input_size
        self.feedforward = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size * 4, self.input_size)
        )
        self.layernorm = nn.LayerNorm(self.input_size)

    def forward(self, inputs):
        return self.layernorm(inputs + self.feedforward(inputs))
