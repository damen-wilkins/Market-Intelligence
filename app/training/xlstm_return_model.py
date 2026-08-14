import torch
from torch import nn
from xlstm import (
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


class XLSTMReturnModel(nn.Module):
    VALID_OUTPUT_MODES = {"point", "gaussian"}

    def __init__(
        self,
        input_size: int,
        context_length: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
        proj_factor: float,
        dropout: float,
        output_mode: str,
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("Input size must be greater than zero.")
        if context_length <= 0:
            raise ValueError("Context length must be greater than zero.")
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")
        if num_blocks <= 0:
            raise ValueError("Number of blocks must be greater than zero.")
        if num_heads <= 0:
            raise ValueError("Number of heads must be greater than zero.")
        if conv1d_kernel_size <= 0:
            raise ValueError("Convolution kernel size must be greater than zero.")
        if qkv_proj_blocksize <= 0:
            raise ValueError("QKV projection block size must be greater than zero.")
        if proj_factor <= 0:
            raise ValueError("Projection factor must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be between zero and one.")
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(
                f"Output mode must be one of {sorted(self.VALID_OUTPUT_MODES)}."
            )

        self.input_size = int(input_size)
        self.context_length = int(context_length)
        self.embedding_dim = int(embedding_dim)
        self.output_mode = output_mode

        self.input_projection = nn.Linear(input_size, embedding_dim)
        self.input_normalization = nn.LayerNorm(embedding_dim)
        self.input_activation = nn.GELU()
        stack_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,
                    qkv_proj_blocksize=qkv_proj_blocksize,
                    num_heads=num_heads,
                    proj_factor=proj_factor,
                )
            ),
            slstm_block=None,
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[],
            dropout=dropout,
            add_post_blocks_norm=True,
        )
        self.backbone = xLSTMBlockStack(stack_config)
        self.output_dropout = nn.Dropout(dropout)
        output_size = 1 if output_mode == "point" else 2
        self.head = nn.Linear(embedding_dim, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "Inputs must have shape [batch_size, sequence_length, input_size]."
            )
        if inputs.shape[1] != self.context_length:
            raise ValueError(
                f"Expected sequence length {self.context_length}, received {inputs.shape[1]}."
            )
        if inputs.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, received {inputs.shape[2]}."
            )
        output = self.input_projection(inputs)
        output = self.input_normalization(output)
        output = self.input_activation(output)
        output = self.backbone(output)
        final_state = self.output_dropout(output[:, -1, :])
        prediction = self.head(final_state)
        if self.output_mode == "gaussian":
            mean = prediction[:, :1]
            log_scale = prediction[:, 1:].clamp(min=-5.0, max=3.0)
            return torch.cat([mean, log_scale], dim=1)
        return prediction
