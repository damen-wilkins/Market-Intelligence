from typing import Any

import torch
from torch import nn


class LSTMResidualModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "LSTM input size must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "LSTM hidden size must be greater than zero."
            )

        if num_layers <= 0:
            raise ValueError(
                "LSTM layer count must be greater than zero."
            )

        if not 0 <= dropout < 1:
            raise ValueError(
                "LSTM dropout must be between 0 and 1."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.normalization = nn.LayerNorm(
            hidden_size
        )
        self.output = nn.Linear(
            hidden_size,
            1,
        )

        self._initialize_parameters()

    def forward(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError(
                "LSTM input must have shape "
                "(batch, sequence, features)."
            )

        if sequences.shape[-1] != self.input_size:
            raise ValueError(
                "LSTM input feature count does not match "
                "the model configuration."
            )

        outputs, _ = self.lstm(sequences)
        final_state = outputs[:, -1, :]
        normalized_state = self.normalization(
            final_state
        )

        return self.output(
            normalized_state
        ).squeeze(-1)

    def get_config(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }

    def _initialize_parameters(self) -> None:
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

                hidden_size = parameter.shape[0] // 4
                parameter.data[
                    hidden_size:2 * hidden_size
                ].fill_(1.0)

        nn.init.xavier_uniform_(
            self.output.weight
        )
        nn.init.zeros_(
            self.output.bias
        )
