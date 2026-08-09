import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ):
        super().__init__()

        if gamma < 0:
            raise ValueError(
                "Gamma must be greater than or equal to zero."
            )

        if alpha is not None:
            if alpha.ndim != 1:
                raise ValueError(
                    "Alpha must be a one-dimensional tensor."
                )

            if (alpha <= 0).any():
                raise ValueError(
                    "Alpha weights must be greater than zero."
                )

            self.register_buffer(
                "alpha",
                alpha.float(),
            )
        else:
            self.alpha = None

        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(
                "Logits must have shape [batch_size, num_classes]."
            )

        if targets.ndim != 1:
            raise ValueError(
                "Targets must have shape [batch_size]."
            )

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "Logits and targets must contain the same number of rows."
            )

        if targets.numel() == 0:
            raise ValueError(
                "Targets cannot be empty."
            )

        if targets.min() < 0 or targets.max() >= logits.shape[1]:
            raise ValueError(
                "Targets contain an invalid class index."
            )

        log_probabilities = nn.functional.log_softmax(
            logits,
            dim=1,
        )

        target_log_probabilities = log_probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        target_probabilities = target_log_probabilities.exp()

        loss = (
            -(
                1.0 - target_probabilities
            ) ** self.gamma
            * target_log_probabilities
        )

        if self.alpha is not None:
            loss = (
                self.alpha[targets]
                * loss
            )

        return loss.mean()