import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 3,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "Input size must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "Hidden size must be greater than zero."
            )

        if num_layers <= 0:
            raise ValueError(
                "Number of layers must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Dropout must be between 0 and 1."
            )

        if num_classes <= 1:
            raise ValueError(
                "Number of classes must be greater than one."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout_layer = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "Inputs must have shape "
                "[batch_size, sequence_length, input_size]."
            )

        if inputs.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"received {inputs.shape[2]}."
            )

        output, _ = self.lstm(inputs)

        final_hidden_state = output[:, -1, :]

        final_hidden_state = self.dropout_layer(
            final_hidden_state
        )

        logits = self.classifier(
            final_hidden_state
        )

        return logits