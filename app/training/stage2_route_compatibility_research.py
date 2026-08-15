from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.training.stage2_outer_validation_gate import moving_block_bootstrap_auc_ci


@dataclass(frozen=True)
class RouteCompatibilityMetrics:
    rows: int
    predicted_move_rows: int
    true_move_rows: int
    routed_true_move_rows: int
    routed_flat_rows: int
    route_move_purity: float
    route_flat_contamination: float
    true_move_recall: float
    stage2_auc_all_true_move: float
    stage2_auc_routed_true_move: float
    stage2_auc_routed_high_vol_true_move: float
    routed_direction_accuracy: float
    high_vol_direction_accuracy: float
    high_vol_directional_coverage: float


class Stage2RouteCompatibilityResearch:
    def __init__(
        self,
        bootstrap_resamples: int = 2000,
        bootstrap_block_length: int = 20,
        random_state: int = 42,
    ):
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.bootstrap_block_length = int(bootstrap_block_length)
        self.random_state = int(random_state)

    def evaluate_fold(
        self,
        outer_fold: int,
        routed_test: pd.DataFrame,
        stage2_true_move: pd.DataFrame,
        high_volatility_threshold: float,
        stage2_decision_threshold: float,
    ) -> dict:
        required_route = {
            'target_date',
            'direction',
            'stage1_move_probability',
            'stage1_predicted_move',
            'realized_volatility_20',
        }
        required_stage2 = {
            'target_date',
            'actual_up',
            'score',
            'future_log_return',
        }
        self._require_columns(routed_test, required_route, 'routed_test')
        self._require_columns(stage2_true_move, required_stage2, 'stage2_true_move')

        route = routed_test.sort_values('target_date').reset_index(drop=True).copy()
        move = stage2_true_move.sort_values('target_date').reset_index(drop=True).copy()
        route['actual_move'] = route['direction'].astype(str) != 'FLAT'
        route['high_volatility'] = (
            route['realized_volatility_20'].astype(float)
            > float(high_volatility_threshold)
        )

        predicted_move = route['stage1_predicted_move'].astype(bool)
        actual_move = route['actual_move'].astype(bool)
        routed_true = predicted_move & actual_move
        routed_flat = predicted_move & ~actual_move
        accepted_high = predicted_move & route['high_volatility'].astype(bool)
        accepted_high_true = accepted_high & actual_move
        accepted_high_flat = accepted_high & ~actual_move

        move_lookup = move.set_index(pd.to_datetime(move['target_date']))
        routed_true_dates = pd.DatetimeIndex(
            pd.to_datetime(route.loc[routed_true, 'target_date'])
        )
        high_true_dates = pd.DatetimeIndex(
            pd.to_datetime(route.loc[accepted_high_true, 'target_date'])
        )
        routed_stage2 = self._select_dates(move_lookup, routed_true_dates)
        high_stage2 = self._select_dates(move_lookup, high_true_dates)

        all_auc = self._auc(move)
        routed_auc = self._auc(routed_stage2)
        high_auc = self._auc(high_stage2)

        routed_correct = self._correct_count(
            routed_stage2,
            stage2_decision_threshold,
        )
        high_correct = self._correct_count(
            high_stage2,
            stage2_decision_threshold,
        )
        predicted_move_rows = int(predicted_move.sum())
        high_rows = int(accepted_high.sum())

        metrics = RouteCompatibilityMetrics(
            rows=int(len(route)),
            predicted_move_rows=predicted_move_rows,
            true_move_rows=int(actual_move.sum()),
            routed_true_move_rows=int(routed_true.sum()),
            routed_flat_rows=int(routed_flat.sum()),
            route_move_purity=(
                float(routed_true.sum() / predicted_move_rows)
                if predicted_move_rows > 0 else float('nan')
            ),
            route_flat_contamination=(
                float(routed_flat.sum() / predicted_move_rows)
                if predicted_move_rows > 0 else float('nan')
            ),
            true_move_recall=(
                float(routed_true.sum() / actual_move.sum())
                if actual_move.sum() > 0 else float('nan')
            ),
            stage2_auc_all_true_move=all_auc,
            stage2_auc_routed_true_move=routed_auc,
            stage2_auc_routed_high_vol_true_move=high_auc,
            routed_direction_accuracy=(
                float(routed_correct / predicted_move_rows)
                if predicted_move_rows > 0 else float('nan')
            ),
            high_vol_direction_accuracy=(
                float(high_correct / high_rows)
                if high_rows > 0 else float('nan')
            ),
            high_vol_directional_coverage=(
                float(high_rows / len(route))
                if len(route) > 0 else float('nan')
            ),
        )

        return {
            'outer_fold': int(outer_fold),
            **metrics.__dict__,
            'high_vol_routed_rows': high_rows,
            'high_vol_routed_true_move_rows': int(accepted_high_true.sum()),
            'high_vol_routed_flat_rows': int(accepted_high_flat.sum()),
            'stage1_threshold': float(route['stage1_threshold'].iloc[0]),
            'high_volatility_threshold': float(high_volatility_threshold),
            'test_start': pd.Timestamp(route['target_date'].min()),
            'test_end': pd.Timestamp(route['target_date'].max()),
        }

    def pooled_summary(
        self,
        route_predictions: pd.DataFrame,
        fold_results: pd.DataFrame,
        routed_stage2_predictions: pd.DataFrame,
        high_vol_stage2_predictions: pd.DataFrame,
    ) -> dict:
        route = route_predictions.sort_values('target_date').reset_index(drop=True)
        predicted_move = route['stage1_predicted_move'].astype(bool)
        actual_move = route['direction'].astype(str) != 'FLAT'
        routed_true = predicted_move & actual_move
        routed_flat = predicted_move & ~actual_move
        high_route = predicted_move & route['high_volatility'].astype(bool)
        high_true = high_route & actual_move
        high_flat = high_route & ~actual_move

        routed_rows = int(predicted_move.sum())
        high_rows = int(high_route.sum())
        routed_correct = int(routed_stage2_predictions['correct'].sum())
        high_correct = int(high_vol_stage2_predictions['correct'].sum())

        routed_bootstrap = self._bootstrap_auc(
            routed_stage2_predictions,
            self.random_state + 1000,
        )
        high_bootstrap = self._bootstrap_auc(
            high_vol_stage2_predictions,
            self.random_state + 2000,
        )

        contamination_all_folds = bool(
            (pd.to_numeric(fold_results['routed_flat_rows']) > 0).all()
        )

        return {
            'rows': int(len(route)),
            'true_move_rows': int(actual_move.sum()),
            'predicted_move_rows': routed_rows,
            'routed_true_move_rows': int(routed_true.sum()),
            'routed_flat_rows': int(routed_flat.sum()),
            'route_move_purity': (
                float(routed_true.sum() / routed_rows)
                if routed_rows > 0 else float('nan')
            ),
            'route_flat_contamination': (
                float(routed_flat.sum() / routed_rows)
                if routed_rows > 0 else float('nan')
            ),
            'true_move_recall': (
                float(routed_true.sum() / actual_move.sum())
                if actual_move.sum() > 0 else float('nan')
            ),
            'routed_direction_accuracy': (
                float(routed_correct / routed_rows)
                if routed_rows > 0 else float('nan')
            ),
            'high_vol_routed_rows': high_rows,
            'high_vol_routed_true_move_rows': int(high_true.sum()),
            'high_vol_routed_flat_rows': int(high_flat.sum()),
            'high_vol_route_move_purity': (
                float(high_true.sum() / high_rows)
                if high_rows > 0 else float('nan')
            ),
            'high_vol_route_flat_contamination': (
                float(high_flat.sum() / high_rows)
                if high_rows > 0 else float('nan')
            ),
            'high_vol_directional_coverage': (
                float(high_rows / len(route))
                if len(route) > 0 else float('nan')
            ),
            'high_vol_direction_accuracy': (
                float(high_correct / high_rows)
                if high_rows > 0 else float('nan')
            ),
            'routed_true_move_auc': self._auc(routed_stage2_predictions),
            'routed_true_move_auc_lower_95': routed_bootstrap['lower_95'],
            'routed_true_move_auc_upper_95': routed_bootstrap['upper_95'],
            'high_vol_routed_true_move_auc': self._auc(high_vol_stage2_predictions),
            'high_vol_routed_true_move_auc_lower_95': high_bootstrap['lower_95'],
            'high_vol_routed_true_move_auc_upper_95': high_bootstrap['upper_95'],
            'false_move_routing_present_in_all_folds': contamination_all_folds,
            'label_space_mismatch': bool(routed_flat.sum() > 0),
            'development_confirms_route_compatibility_problem': bool(
                routed_flat.sum() > 0 and contamination_all_folds
            ),
        }

    def _bootstrap_auc(self, frame: pd.DataFrame, seed: int) -> dict:
        if frame.empty or frame['actual_up'].nunique() < 2:
            return {
                'lower_95': float('nan'),
                'upper_95': float('nan'),
                'probability_auc_above_0_50': float('nan'),
                'valid_resamples': 0,
            }
        ordered = frame.sort_values('target_date')
        return moving_block_bootstrap_auc_ci(
            actual=ordered['actual_up'].astype(int).to_numpy(),
            score=ordered['score'].astype(float).to_numpy(),
            resamples=self.bootstrap_resamples,
            block_length=min(self.bootstrap_block_length, len(ordered)),
            random_state=seed,
        )

    @staticmethod
    def _correct_count(frame: pd.DataFrame, threshold: float) -> int:
        if frame.empty:
            return 0
        predicted = (frame['score'].astype(float).to_numpy() >= float(threshold)).astype(int)
        actual = frame['actual_up'].astype(int).to_numpy()
        return int((predicted == actual).sum())

    @staticmethod
    def _auc(frame: pd.DataFrame) -> float:
        if frame.empty or frame['actual_up'].nunique() < 2:
            return float('nan')
        return float(
            roc_auc_score(
                frame['actual_up'].astype(int).to_numpy(),
                frame['score'].astype(float).to_numpy(),
            )
        )

    @staticmethod
    def _select_dates(
        lookup: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        if len(dates) == 0:
            return pd.DataFrame(columns=lookup.columns)
        missing = dates.difference(lookup.index)
        if len(missing) > 0:
            raise ValueError(
                'Stage-2 saved OOF predictions are missing routed true-MOVE dates: '
                + ', '.join(str(value.date()) for value in missing[:5])
            )
        selected = lookup.loc[dates].copy().reset_index(drop=True)
        return selected.sort_values('target_date').reset_index(drop=True)

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(f'{name} is missing columns: ' + ', '.join(missing))
