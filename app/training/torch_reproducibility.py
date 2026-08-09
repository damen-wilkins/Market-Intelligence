import os
import random

import numpy as np

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import torch


class TorchReproducibility:
    @staticmethod
    def configure(
        seed: int = 42,
        deterministic: bool = True,
    ) -> None:
        if seed < 0:
            raise ValueError(
                "Seed must be greater than or equal to zero."
            )

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(
                True
            )

            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(
                False
            )

            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True