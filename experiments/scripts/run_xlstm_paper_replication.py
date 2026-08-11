from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import pywt
import torch

from app.training.experiment_tracker import ExperimentTracker
from app.training.paper_wavelet_preprocessor import (
    PaperWaveletPreprocessor,
    PriceSequenceSet,
)
from app.training.xlstm_price_forecast_evaluator import (
    XLSTMPriceForecastEvaluator,
)
from app.training.xlstm_price_regression_trainer import (
    XLSTMPriceRegressionTrainer,
)
from app.training.torch_reproducibility import (
    TorchReproducibility,
)
from app.training.xlstm_price_regressor_model import (
    XLSTMPriceRegressor,
)
from database.connection import get_connection_string
from database.market_data_repository import MarketDataRepository


TICKER = "^GSPC"
START_DATE = pd.Timestamp(
    "2000-01-03"
)
END_DATE = pd.Timestamp(
    "2023-12-29"
)
EXPECTED_ROWS = 6037
EXPECTED_FIRST_CLOSE = 1455.219970703125
EXPECTED_LAST_CLOSE = 4769.830078125

TRAIN_END_DATE = "2021-01-01"
VALIDATION_END_DATE = "2022-07-01"
SEQUENCE_LENGTH = 150
RANDOM_STATE = 42

MODEL_DIRECTORY = Path(
    "models"
)
EXPERIMENT_DIRECTORY = Path(
    "experiments"
)

PAPER_REPORTED_TEST_ACCURACY = 0.7128
PAPER_REPORTED_F1 = 0.7300

MODES = (
    PaperWaveletPreprocessor.PAPER_NONCAUSAL,
    PaperWaveletPreprocessor.CAUSAL,
)

CHECKPOINT_VARIANTS = (
    XLSTMPriceRegressionTrainer.AUTHORS_BACKBONE_CHECKPOINT,
    XLSTMPriceRegressionTrainer.BEST_FULL_CHECKPOINT,
)


def main():
    print(
        "Loading S&P 500 index data for paper replication..."
    )

    data = load_replication_data()

    print(
        f"Rows: {len(data)}"
    )
    print(
        "Period:",
        data[
            "trade_date"
        ].min().strftime(
            "%Y-%m-%d"
        ),
        "->",
        data[
            "trade_date"
        ].max().strftime(
            "%Y-%m-%d"
        ),
    )

    print()
    print(
        "Running two preprocessing experiments:"
    )
    print(
        "1. paper_noncausal: full-series wavelet + global MinMax scaling"
    )
    print(
        "2. causal: prefix-only wavelet + training-only MinMax scaling"
    )
    print(
        "Each training run is evaluated with both the authors' "
        "backbone-only checkpoint behavior and a corrected full-model checkpoint."
    )

    results = {}

    for mode in MODES:
        results[
            mode
        ] = run_mode(
            mode=mode,
            data=data,
        )

        cleanup_cuda()

    comparison_path = save_comparison_experiment(
        results
    )

    print_final_comparison(
        results
    )

    print()
    print(
        "Comparison experiment:",
        comparison_path,
    )


def load_replication_data() -> pd.DataFrame:
    repository = MarketDataRepository(
        get_connection_string()
    )

    data = pd.DataFrame(
        repository.get_market_data(
            TICKER
        )
    )

    if data.empty:
        raise ValueError(
            "No ^GSPC market data was found. "
            "Run python -m scripts.download_sp500_index first."
        )

    data[
        "trade_date"
    ] = pd.to_datetime(
        data[
            "trade_date"
        ]
    )

    data = data.loc[
        (
            data[
                "trade_date"
            ]
            >= START_DATE
        )
        & (
            data[
                "trade_date"
            ]
            <= END_DATE
        ),
        [
            "trade_date",
            "close",
        ],
    ].copy()

    data = data.sort_values(
        "trade_date"
    ).reset_index(
        drop=True
    )

    if data.empty:
        raise ValueError(
            "No ^GSPC observations were found inside the replication period."
        )

    if data[
        "trade_date"
    ].min() != START_DATE:
        raise ValueError(
            "Replication data does not begin on 2000-01-03."
        )

    if data[
        "trade_date"
    ].max() != END_DATE:
        raise ValueError(
            "Replication data does not end on 2023-12-29."
        )

    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            "Replication row count does not match the authors' published dataset. "
            f"Expected {EXPECTED_ROWS}, received {len(data)}."
        )

    if data[
        "trade_date"
    ].duplicated().any():
        raise ValueError(
            "Replication data contains duplicate trade dates."
        )

    if data[
        "close"
    ].isna().any():
        raise ValueError(
            "Replication close prices contain missing values."
        )

    if not np.isclose(
        float(
            data.iloc[0][
                "close"
            ]
        ),
        EXPECTED_FIRST_CLOSE,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Replication first close does not match the authors' dataset."
        )

    if not np.isclose(
        float(
            data.iloc[-1][
                "close"
            ]
        ),
        EXPECTED_LAST_CLOSE,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Replication last close does not match the authors' dataset."
        )

    return data


def run_mode(
    mode: str,
    data: pd.DataFrame,
) -> dict:
    print()
    print(
        "=============================================="
    )
    print(
        f"RUNNING {mode.upper()}"
    )
    print(
        "=============================================="
    )

    if mode == PaperWaveletPreprocessor.CAUSAL:
        print(
            "Building causal wavelet sequences. "
            "This preprocessing pass is intentionally slower."
        )

    preprocessor = PaperWaveletPreprocessor(
        mode=mode,
        sequence_length=SEQUENCE_LENGTH,
        train_end_date=TRAIN_END_DATE,
        validation_end_date=VALIDATION_END_DATE,
        wavelet="db4",
        level=1,
        pad_width=100,
    )

    splits = preprocessor.prepare(
        data
    )

    print_split_summary(
        splits
    )

    TorchReproducibility.configure(
        seed=RANDOM_STATE,
        deterministic=False,
    )

    model = XLSTMPriceRegressor(
        sequence_length=SEQUENCE_LENGTH,
        input_size=1,
        embedding_dim=64,
        output_size=1,
        num_blocks=4,
        mlstm_conv1d_kernel_size=4,
        mlstm_qkv_proj_blocksize=2,
        mlstm_num_heads=2,
        slstm_conv1d_kernel_size=2,
        slstm_num_heads=2,
        slstm_feedforward_proj_factor=1.1,
        slstm_backend="vanilla",
    )

    trainer = XLSTMPriceRegressionTrainer(
        learning_rate=0.0001,
        batch_size=16,
        max_epochs=200,
        patience=40,
        scheduler_patience=10,
        scheduler_factor=0.5,
        gradient_clip=1.0,
        seed=RANDOM_STATE,
    )

    print()
    print(
        "Training xLSTM-TS next-close regressor..."
    )

    training_result = trainer.train(
        model=model,
        X_train=splits[
            "train"
        ].X,
        y_train=splits[
            "train"
        ].y,
        X_validation=splits[
            "validation"
        ].X,
        y_validation=splits[
            "validation"
        ].y,
    )

    checkpoint_results = {}

    for checkpoint_variant in CHECKPOINT_VARIANTS:
        evaluated_model = trainer.load_checkpoint_variant(
            model=training_result[
                "model"
            ],
            training_result=training_result,
            checkpoint_variant=checkpoint_variant,
        )

        checkpoint_results[
            checkpoint_variant
        ] = evaluate_checkpoint(
            model=evaluated_model,
            trainer=trainer,
            preprocessor=preprocessor,
            splits=splits,
        )

    prediction_path = save_test_predictions(
        mode=mode,
        split=splits[
            "test"
        ],
        checkpoint_results=checkpoint_results,
    )

    model_path = save_model_artifact(
        mode=mode,
        model=training_result[
            "model"
        ],
        preprocessor=preprocessor,
        trainer=trainer,
        training_result=training_result,
        checkpoint_results=checkpoint_results,
    )

    experiment_path = save_mode_experiment(
        mode=mode,
        preprocessor=preprocessor,
        model=training_result[
            "model"
        ],
        trainer=trainer,
        training_result=training_result,
        checkpoint_results=checkpoint_results,
    )

    print_mode_result(
        mode=mode,
        training_result=training_result,
        checkpoint_results=checkpoint_results,
        model_path=model_path,
        prediction_path=prediction_path,
        experiment_path=experiment_path,
    )

    return {
        "mode": mode,
        "training": {
            "best_epoch": training_result[
                "best_epoch"
            ],
            "best_validation_loss": training_result[
                "best_validation_loss"
            ],
            "epochs_trained": training_result[
                "epochs_trained"
            ],
        },
        "checkpoints": {
            checkpoint_variant: {
                split_name: checkpoint_results[
                    checkpoint_variant
                ][
                    split_name
                ][
                    "metrics"
                ]
                for split_name in (
                    "train",
                    "validation",
                    "test",
                )
            }
            for checkpoint_variant in CHECKPOINT_VARIANTS
        },
        "model_path": str(
            model_path
        ),
        "prediction_path": str(
            prediction_path
        ),
        "experiment_path": str(
            experiment_path
        ),
    }


def evaluate_checkpoint(
    model: XLSTMPriceRegressor,
    trainer: XLSTMPriceRegressionTrainer,
    preprocessor: PaperWaveletPreprocessor,
    splits: dict[str, PriceSequenceSet],
) -> dict:
    evaluator = XLSTMPriceForecastEvaluator()
    split_results = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_data = splits[
            split_name
        ]

        predicted_scaled = trainer.predict(
            model=model,
            X=split_data.X,
        )

        predicted_close = preprocessor.inverse_transform(
            predicted_scaled
        )

        split_results[
            split_name
        ] = {
            "metrics": evaluator.evaluate(
                actual_close=split_data.actual_close,
                predicted_close=predicted_close,
                current_close=split_data.current_close,
                prior_close=split_data.prior_close,
            ),
            "predicted_close": predicted_close,
        }

    return split_results


def save_test_predictions(
    mode: str,
    split: PriceSequenceSet,
    checkpoint_results: dict,
) -> Path:
    evaluator = XLSTMPriceForecastEvaluator()

    output = pd.DataFrame(
        {
            "target_date": pd.to_datetime(
                split.target_dates
            ),
            "prior_close": split.prior_close,
            "current_close": split.current_close,
            "actual_close": split.actual_close,
            "denoised_target_close": (
                split.denoised_target_close
            ),
        }
    )

    actual_production = (
        split.actual_close
        > split.current_close
    ).astype(
        np.int64
    )

    momentum_predicted = (
        split.current_close
        > split.prior_close
    ).astype(
        np.int64
    )

    output[
        "actual_next_day_direction"
    ] = np.where(
        actual_production == 1,
        "UP",
        "DOWN",
    )

    output[
        "momentum_baseline_direction"
    ] = np.where(
        momentum_predicted == 1,
        "UP",
        "DOWN",
    )

    for checkpoint_variant in CHECKPOINT_VARIANTS:
        predicted_close = checkpoint_results[
            checkpoint_variant
        ][
            "test"
        ][
            "predicted_close"
        ]

        _, production_predicted = (
            evaluator.production_direction_labels(
                actual_close=split.actual_close,
                predicted_close=predicted_close,
                current_close=split.current_close,
            )
        )

        paper_actual, paper_predicted = (
            evaluator.paper_direction_labels(
                actual_close=split.actual_close,
                predicted_close=predicted_close,
            )
        )

        paper_actual_column = np.full(
            len(split),
            None,
            dtype=object,
        )
        paper_predicted_column = np.full(
            len(split),
            None,
            dtype=object,
        )

        paper_actual_column[1:] = np.where(
            paper_actual == 1,
            "UP",
            "DOWN",
        )
        paper_predicted_column[1:] = np.where(
            paper_predicted == 1,
            "UP",
            "DOWN",
        )

        prefix = checkpoint_variant

        output[
            f"{prefix}_predicted_close"
        ] = predicted_close
        output[
            f"{prefix}_next_day_direction"
        ] = np.where(
            production_predicted == 1,
            "UP",
            "DOWN",
        )
        output[
            f"{prefix}_paper_actual_direction"
        ] = paper_actual_column
        output[
            f"{prefix}_paper_predicted_direction"
        ] = paper_predicted_column

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = MODEL_DIRECTORY / (
        f"xlstm_sp500_{mode}_test_predictions.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    return path


def save_model_artifact(
    mode: str,
    model: XLSTMPriceRegressor,
    preprocessor: PaperWaveletPreprocessor,
    trainer: XLSTMPriceRegressionTrainer,
    training_result: dict,
    checkpoint_results: dict,
) -> Path:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = MODEL_DIRECTORY / (
        f"xlstm_sp500_{mode}.pt"
    )

    artifact = {
        "model_type": "xlstm_price_regressor",
        "model_config": model.get_config(),
        "preprocessor_state": preprocessor.get_state(),
        "trainer_config": trainer.get_config(),
        "best_epoch": training_result[
            "best_epoch"
        ],
        "best_validation_loss": training_result[
            "best_validation_loss"
        ],
        "epochs_trained": training_result[
            "epochs_trained"
        ],
        "best_full_state_dict": training_result[
            "best_full_state_dict"
        ],
        "best_backbone_state_dict": training_result[
            "best_backbone_state_dict"
        ],
        "final_full_state_dict": training_result[
            "final_full_state_dict"
        ],
        "test_metrics": {
            checkpoint_variant: checkpoint_results[
                checkpoint_variant
            ][
                "test"
            ][
                "metrics"
            ]
            for checkpoint_variant in CHECKPOINT_VARIANTS
        },
        "metadata": build_metadata(
            mode
        ),
    }

    torch.save(
        artifact,
        path,
    )

    return path


def save_mode_experiment(
    mode: str,
    preprocessor: PaperWaveletPreprocessor,
    model: XLSTMPriceRegressor,
    trainer: XLSTMPriceRegressionTrainer,
    training_result: dict,
    checkpoint_results: dict,
) -> Path:
    metrics = {
        "training": {
            "best_epoch": training_result[
                "best_epoch"
            ],
            "best_validation_loss": training_result[
                "best_validation_loss"
            ],
            "epochs_trained": training_result[
                "epochs_trained"
            ],
        },
        "checkpoints": {
            checkpoint_variant: {
                split_name: checkpoint_results[
                    checkpoint_variant
                ][
                    split_name
                ][
                    "metrics"
                ]
                for split_name in (
                    "train",
                    "validation",
                    "test",
                )
            }
            for checkpoint_variant in CHECKPOINT_VARIANTS
        },
        "paper_reported_sp500_daily": {
            "test_accuracy": (
                PAPER_REPORTED_TEST_ACCURACY
            ),
            "f1_up": PAPER_REPORTED_F1,
        },
    }

    return ExperimentTracker(
        str(
            EXPERIMENT_DIRECTORY
        )
    ).save(
        experiment_name=(
            f"xlstm_sp500_{mode}"
        ),
        model_name=(
            f"xlstm_sp500_{mode}_v1"
        ),
        parameters={
            "preprocessor": preprocessor.get_state(),
            "model": model.get_config(),
            "trainer": trainer.get_config(),
            "metadata": build_metadata(
                mode
            ),
        },
        metrics=metrics,
        features=[
            "^GSPC Close"
        ],
    )


def save_comparison_experiment(
    results: dict,
) -> Path:
    return ExperimentTracker(
        str(
            EXPERIMENT_DIRECTORY
        )
    ).save(
        experiment_name=(
            "xlstm_sp500_paper_replication_comparison"
        ),
        model_name=(
            "xlstm_sp500_paper_replication_v1"
        ),
        parameters={
            "ticker": TICKER,
            "start_date": START_DATE.strftime(
                "%Y-%m-%d"
            ),
            "end_date": END_DATE.strftime(
                "%Y-%m-%d"
            ),
            "expected_rows": EXPECTED_ROWS,
            "train_end_date": TRAIN_END_DATE,
            "validation_end_date": VALIDATION_END_DATE,
            "sequence_length": SEQUENCE_LENGTH,
            "paper_reported_test_accuracy": (
                PAPER_REPORTED_TEST_ACCURACY
            ),
            "paper_reported_f1": PAPER_REPORTED_F1,
        },
        metrics=results,
        features=[
            "^GSPC Close"
        ],
    )


def build_metadata(
    mode: str,
) -> dict:
    return {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "ticker": TICKER,
        "forecast_horizon": "next_trading_day_close",
        "paper": (
            "An Evaluation of Deep Learning Models "
            "for Stock Market Trend Prediction"
        ),
        "paper_arxiv": "2408.12408",
        "replication_mode": mode,
        "data_period": {
            "start": START_DATE.strftime(
                "%Y-%m-%d"
            ),
            "end": END_DATE.strftime(
                "%Y-%m-%d"
            ),
            "expected_rows": EXPECTED_ROWS,
        },
        "split_boundaries": {
            "train_before": TRAIN_END_DATE,
            "validation_before": VALIDATION_END_DATE,
            "test_from": VALIDATION_END_DATE,
        },
        "runtime_versions": {
            "torch": torch.__version__,
            "xlstm": version(
                "xlstm"
            ),
            "PyWavelets": pywt.__version__,
        },
        "evaluation": {
            "paper_direction": (
                "Direction of consecutive predicted target closes "
                "versus consecutive actual target closes."
            ),
            "production_direction": (
                "Predicted next close versus the actual current close "
                "available at forecast time."
            ),
        },
        "checkpoint_variants": {
            XLSTMPriceRegressionTrainer.AUTHORS_BACKBONE_CHECKPOINT: (
                "Reproduces the authors' source-code checkpoint behavior: "
                "best-validation xLSTM backbone combined with final-epoch "
                "input/output projection layers."
            ),
            XLSTMPriceRegressionTrainer.BEST_FULL_CHECKPOINT: (
                "Corrected checkpoint behavior: restore the complete model "
                "from the best validation epoch."
            ),
        },
        "known_replication_differences": [
            (
                "The project keeps its current torch/xlstm environment "
                "instead of downgrading the full application environment "
                "to the authors' historical package versions."
            ),
            (
                "The authors pinned PyWavelets 1.6.0; this project uses "
                "PyWavelets 1.8.0 for the current Python environment, and "
                "the exact runtime version is recorded above."
            ),
            (
                "The published paper reports early-stopping patience 30, "
                "while the official source code uses 40. This replication "
                "follows the source implementation at 40 epochs."
            ),
            (
                "This replication fixes random seed 42 for repeatability; "
                "the authors' source does not document an equivalent fixed seed."
            ),
        ],
        "production_valid_preprocessing": (
            mode
            == PaperWaveletPreprocessor.CAUSAL
        ),
    }


def print_split_summary(
    splits: dict[str, PriceSequenceSet],
) -> None:
    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split = splits[
            split_name
        ]

        print(
            f"{split_name.capitalize()} sequences: {len(split)} "
            f"({pd.Timestamp(split.target_dates.min()).strftime('%Y-%m-%d')} "
            f"-> {pd.Timestamp(split.target_dates.max()).strftime('%Y-%m-%d')})"
        )


def print_mode_result(
    mode: str,
    training_result: dict,
    checkpoint_results: dict,
    model_path: Path,
    prediction_path: Path,
    experiment_path: Path,
) -> None:
    print()
    print(
        "=============================================="
    )
    print(
        f"{mode.upper()} TEST RESULT"
    )
    print(
        "=============================================="
    )

    print(
        "Best epoch:",
        training_result[
            "best_epoch"
        ],
    )

    for checkpoint_variant in CHECKPOINT_VARIANTS:
        metrics = checkpoint_results[
            checkpoint_variant
        ][
            "test"
        ][
            "metrics"
        ]

        print()
        print(
            checkpoint_variant
        )
        print(
            "  Price RMSE:",
            round(
                metrics[
                    "price"
                ][
                    "rmse"
                ],
                4,
            ),
        )
        print(
            "  Price RMSSE:",
            round(
                metrics[
                    "price"
                ][
                    "rmsse"
                ],
                4,
            ),
        )
        print(
            "  Paper-style accuracy:",
            round(
                metrics[
                    "paper_direction"
                ][
                    "accuracy"
                ],
                4,
            ),
        )
        print(
            "  Paper-style F1 (UP):",
            round(
                metrics[
                    "paper_direction"
                ][
                    "f1_up"
                ],
                4,
            ),
        )
        print(
            "  Production accuracy:",
            round(
                metrics[
                    "production_direction"
                ][
                    "accuracy"
                ],
                4,
            ),
        )
        print(
            "  Production F1 (UP):",
            round(
                metrics[
                    "production_direction"
                ][
                    "f1_up"
                ],
                4,
            ),
        )

    corrected_metrics = checkpoint_results[
        XLSTMPriceRegressionTrainer.BEST_FULL_CHECKPOINT
    ][
        "test"
    ][
        "metrics"
    ]

    print()
    print(
        "Momentum baseline accuracy:",
        round(
            corrected_metrics[
                "momentum_direction_baseline"
            ][
                "accuracy"
            ],
            4,
        ),
    )

    print()
    print(
        "Model artifact:",
        model_path,
    )
    print(
        "Test predictions:",
        prediction_path,
    )
    print(
        "Experiment:",
        experiment_path,
    )


def print_final_comparison(
    results: dict,
) -> None:
    noncausal_authors = get_test_metrics(
        results,
        PaperWaveletPreprocessor.PAPER_NONCAUSAL,
        XLSTMPriceRegressionTrainer.AUTHORS_BACKBONE_CHECKPOINT,
    )
    noncausal_corrected = get_test_metrics(
        results,
        PaperWaveletPreprocessor.PAPER_NONCAUSAL,
        XLSTMPriceRegressionTrainer.BEST_FULL_CHECKPOINT,
    )
    causal_authors = get_test_metrics(
        results,
        PaperWaveletPreprocessor.CAUSAL,
        XLSTMPriceRegressionTrainer.AUTHORS_BACKBONE_CHECKPOINT,
    )
    causal_corrected = get_test_metrics(
        results,
        PaperWaveletPreprocessor.CAUSAL,
        XLSTMPriceRegressionTrainer.BEST_FULL_CHECKPOINT,
    )

    print()
    print(
        "=================================================="
    )
    print(
        "FINAL PAPER REPLICATION COMPARISON"
    )
    print(
        "=================================================="
    )

    print(
        "Paper reported S&P 500 daily accuracy:",
        f"{PAPER_REPORTED_TEST_ACCURACY:.4f}"
    )
    print(
        "Paper reported S&P 500 daily F1:",
        f"{PAPER_REPORTED_F1:.4f}"
    )

    print()
    print(
        "Closest source-code reproduction "
        "(noncausal preprocessing + authors checkpoint):"
    )
    print_direction_summary(
        noncausal_authors
    )

    print()
    print(
        "Noncausal preprocessing + corrected checkpoint:"
    )
    print_direction_summary(
        noncausal_corrected
    )

    print()
    print(
        "Causal preprocessing + authors checkpoint:"
    )
    print_direction_summary(
        causal_authors
    )

    print()
    print(
        "Production candidate "
        "(causal preprocessing + corrected checkpoint):"
    )
    print_direction_summary(
        causal_corrected
    )
    print(
        "  Momentum baseline accuracy:",
        f"{causal_corrected['momentum_direction_baseline']['accuracy']:.4f}"
    )


def print_direction_summary(
    metrics: dict,
) -> None:
    print(
        "  Paper-style accuracy:",
        f"{metrics['paper_direction']['accuracy']:.4f}"
    )
    print(
        "  Paper-style F1 (UP):",
        f"{metrics['paper_direction']['f1_up']:.4f}"
    )
    print(
        "  Production accuracy:",
        f"{metrics['production_direction']['accuracy']:.4f}"
    )
    print(
        "  Production F1 (UP):",
        f"{metrics['production_direction']['f1_up']:.4f}"
    )
    print(
        "  Price RMSE:",
        f"{metrics['price']['rmse']:.4f}"
    )
    print(
        "  Price RMSSE:",
        f"{metrics['price']['rmsse']:.4f}"
    )


def get_test_metrics(
    results: dict,
    mode: str,
    checkpoint_variant: str,
) -> dict:
    return results[
        mode
    ][
        "checkpoints"
    ][
        checkpoint_variant
    ][
        "test"
    ]


def cleanup_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
