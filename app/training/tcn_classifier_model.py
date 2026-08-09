import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(
        self,
        chomp_size: int,
    ):
        super().__init__()

        if chomp_size <= 0:
            raise ValueError(
                "Chomp size must be greater than zero."
            )

        self.chomp_size = chomp_size

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs[
            :,
            :,
            :-self.chomp_size,
        ].contiguous()


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.chomp1 = Chomp1d(
            padding
        )

        self.activation1 = nn.GELU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.chomp2 = Chomp1d(
            padding
        )

        self.activation2 = nn.GELU()

        self.dropout2 = nn.Dropout(
            dropout
        )

        if input_channels != output_channels:
            self.residual_projection = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
            )
        else:
            self.residual_projection = nn.Identity()

        self.output_activation = nn.GELU()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1(
            inputs
        )

        output = self.chomp1(
            output
        )

        output = self.activation1(
            output
        )

        output = self.dropout1(
            output
        )

        output = self.conv2(
            output
        )

        output = self.chomp2(
            output
        )

        output = self.activation2(
            output
        )

        output = self.dropout2(
            output
        )

        residual = self.residual_projection(
            inputs
        )

        return self.output_activation(
            output + residual
        )


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        channel_width: int,
        num_blocks: int,
        kernel_size: int,
        dropout: float,
        num_classes: int = 3,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "Input size must be greater than zero."
            )

        if channel_width <= 0:
            raise ValueError(
                "Channel width must be greater than zero."
            )

        if num_blocks <= 0:
            raise ValueError(
                "Number of blocks must be greater than zero."
            )

        if kernel_size < 2:
            raise ValueError(
                "Kernel size must be at least two."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Dropout must be between zero and one."
            )

        if num_classes <= 1:
            raise ValueError(
                "Number of classes must be greater than one."
            )

        self.input_size = input_size
        self.channel_width = channel_width
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes

        blocks = []

        for block_index in range(
            num_blocks
        ):
            dilation = (
                2 ** block_index
            )

            block_input_channels = (
                input_size
                if block_index == 0
                else channel_width
            )

            blocks.append(
                TCNResidualBlock(
                    input_channels=block_input_channels,
                    output_channels=channel_width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(
            *blocks
        )

        self.classifier = nn.Linear(
            channel_width,
            num_classes,
        )

        self._initialize_weights()

    @property
    def receptive_field(self) -> int:
        dilation_sum = sum(
            2 ** index
            for index in range(
                self.num_blocks
            )
        )

        return (
            1
            + 2
            * (
                self.kernel_size - 1
            )
            * dilation_sum
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

        temporal_input = inputs.transpose(
            1,
            2,
        )

        temporal_output = self.network(
            temporal_input
        )

        final_state = temporal_output[
            :,
            :,
            -1,
        ]

        return self.classifier(
            final_state
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(
                module,
                nn.Conv1d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Linear,
            ):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )