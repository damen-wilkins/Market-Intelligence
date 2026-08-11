import torch
from torch import nn
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


class XLSTMPriceRegressor(nn.Module):
    def __init__(
        self,
        sequence_length: int = 150,
        input_size: int = 1,
        embedding_dim: int = 64,
        output_size: int = 1,
        num_blocks: int = 4,
        mlstm_conv1d_kernel_size: int = 4,
        mlstm_qkv_proj_blocksize: int = 2,
        mlstm_num_heads: int = 2,
        slstm_conv1d_kernel_size: int = 2,
        slstm_num_heads: int = 2,
        slstm_feedforward_proj_factor: float = 1.1,
        slstm_backend: str = "vanilla",
    ):
        super().__init__()

        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be positive."
            )

        if input_size <= 0:
            raise ValueError(
                "Input size must be positive."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "Embedding dimension must be positive."
            )

        if output_size <= 0:
            raise ValueError(
                "Output size must be positive."
            )

        if num_blocks != 4:
            raise ValueError(
                "Paper replication requires exactly four xLSTM blocks."
            )

        self.sequence_length = sequence_length
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.num_blocks = num_blocks
        self.mlstm_conv1d_kernel_size = mlstm_conv1d_kernel_size
        self.mlstm_qkv_proj_blocksize = mlstm_qkv_proj_blocksize
        self.mlstm_num_heads = mlstm_num_heads
        self.slstm_conv1d_kernel_size = slstm_conv1d_kernel_size
        self.slstm_num_heads = slstm_num_heads
        self.slstm_feedforward_proj_factor = slstm_feedforward_proj_factor
        self.slstm_backend = slstm_backend

        self.input_projection = nn.Linear(
            input_size,
            embedding_dim,
        )

        stack_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=(
                        mlstm_conv1d_kernel_size
                    ),
                    qkv_proj_blocksize=(
                        mlstm_qkv_proj_blocksize
                    ),
                    num_heads=mlstm_num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=slstm_backend,
                    num_heads=slstm_num_heads,
                    conv1d_kernel_size=(
                        slstm_conv1d_kernel_size
                    ),
                    bias_init=(
                        "powerlaw_blockdependent"
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=(
                        slstm_feedforward_proj_factor
                    ),
                    act_fn="gelu",
                ),
            ),
            context_length=sequence_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[1],
        )

        self.backbone = xLSTMBlockStack(
            stack_config
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            output_size,
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

        if inputs.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"received {inputs.shape[1]}."
            )

        if inputs.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"received {inputs.shape[2]}."
            )

        output = self.input_projection(
            inputs
        )

        output = self.backbone(
            output
        )

        return self.output_projection(
            output[
                :,
                -1,
                :,
            ]
        )

    def get_config(self) -> dict:
        return {
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "embedding_dim": self.embedding_dim,
            "output_size": self.output_size,
            "num_blocks": self.num_blocks,
            "mlstm_conv1d_kernel_size": (
                self.mlstm_conv1d_kernel_size
            ),
            "mlstm_qkv_proj_blocksize": (
                self.mlstm_qkv_proj_blocksize
            ),
            "mlstm_num_heads": self.mlstm_num_heads,
            "slstm_conv1d_kernel_size": (
                self.slstm_conv1d_kernel_size
            ),
            "slstm_num_heads": self.slstm_num_heads,
            "slstm_feedforward_proj_factor": (
                self.slstm_feedforward_proj_factor
            ),
            "slstm_backend": self.slstm_backend,
        }
