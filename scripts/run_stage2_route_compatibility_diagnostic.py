from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from app.training.stage2_route_compatibility_research import (
    Stage2RouteCompatibilityResearch,
)
from app.training.volatility_direction_label_builder import (
    VolatilityDirectionLabelBuilder,
)
from database.direction_training_data_repository import (
    DirectionTrainingDataRepository,
)
from database.stage2_signal_data_repository import Stage2SignalDataRepository
from scripts.run_stage1_long_history_optimization import (
    evaluate_candidate,
    train_stage1_fold,
)
from scripts.run_stage1_target_optimization import (
    STAGE1_FEATURE_COLUMNS,
    TARGET_STATE_FEATURE as STAGE1_TARGET_STATE_FEATURE,
    build_common_feature_frame,
)
from scripts.run_stage2_conditioned_megasearch import (
    build_master,
    columns_for_groups,
    dataset,
    load_training_cutoff,
)
from scripts.run_stage2_return_architecture_search import (
    TREE_PROGRESS,
    add_normalized_target,
    latest_verified_winner_groups,
)
from app.training.stage2_conditioned_target_research import target_specs


TICKER = 'SPY'
TARGET_NAME = 'flat_90d_k700'
TARGET_WINDOW = 90
TARGET_MULTIPLIER = 0.700
REGIME_FEATURE = 'realized_volatility_20'
REGIME_QUANTILE = 2.0 / 3.0
OUTER_SPLITS = 3
RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LENGTH = 20
EXPERIMENT_DIRECTORY = Path('experiments')
STAGE1_PROGRESS_PATH = EXPERIMENT_DIRECTORY / 'stage1_target_optimization_v1_progress.json'
PROGRESS_PATH = EXPERIMENT_DIRECTORY / 'stage2_route_compatibility_diagnostic_v1_progress.json'
EXPERIMENT_NAME = 'stage2_route_compatibility_diagnostic_v1'


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, default=json_default)
    temporary.replace(path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def load_locked_stage1() -> dict:
    rows = load_json(STAGE1_PROGRESS_PATH, [])
    matches = [
        row for row in rows
        if str(row.get('target_name')) == TARGET_NAME
        and int(row.get('volatility_window')) == TARGET_WINDOW
        and abs(float(row.get('threshold_multiplier')) - TARGET_MULTIPLIER) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one locked Stage-1 result for {TARGET_NAME}; found {len(matches)}.'
        )
    if not isinstance(matches[0].get('parameters'), dict):
        raise ValueError('Locked Stage-1 result is missing parameters.')
    return matches[0]


def load_stage2_saved_oof() -> dict[int, dict]:
    payload = load_json(TREE_PROGRESS, {'rows': []})
    rows = [
        row for row in payload.get('rows', [])
        if row.get('architecture') == 'xgboost_binary_winner'
    ]
    by_fold = {int(row['outer_fold']): row for row in rows}
    missing = sorted(set(range(1, OUTER_SPLITS + 1)) - set(by_fold))
    if missing:
        raise RuntimeError(
            'Saved nested Stage-2 XGBoost OOF predictions are incomplete for folds: '
            + ', '.join(map(str, missing))
        )
    return by_fold


def locked_target_spec():
    matches = [
        spec for spec in target_specs()
        if spec.volatility_window == TARGET_WINDOW
        and abs(spec.threshold_multiplier - TARGET_MULTIPLIER) < 1e-12
    ]
    if len(matches) != 1:
        raise RuntimeError('Could not resolve the locked 90d x 0.700 Stage-2 target.')
    return matches[0]


def build_stage2_development() -> tuple[pd.DataFrame, tuple[str, ...], list[str], pd.Timestamp]:
    cutoff = load_training_cutoff()
    raw = Stage2SignalDataRepository().get_training_data(ticker=TICKER)
    master = build_master(raw, locked_target_spec(), cutoff)
    winner_groups = latest_verified_winner_groups()
    winner_features = columns_for_groups(winner_groups)
    data = add_normalized_target(dataset(master, winner_features))
    if REGIME_FEATURE not in data.columns:
        raise ValueError(f'{REGIME_FEATURE} is missing from the locked Stage-2 dataset.')
    return data.sort_values('target_date').reset_index(drop=True), winner_groups, winner_features, cutoff


def build_stage1_development(cutoff: pd.Timestamp) -> pd.DataFrame:
    raw = DirectionTrainingDataRepository().get_training_data(
        ticker=TICKER,
        include_breadth=False,
        include_cross_asset=False,
    ).copy()
    raw['trade_date'] = pd.to_datetime(raw['trade_date'])
    features = build_common_feature_frame(raw)
    labels = VolatilityDirectionLabelBuilder(
        volatility_window=TARGET_WINDOW,
        threshold_multiplier=TARGET_MULTIPLIER,
    ).build(raw[['trade_date', 'close']].copy())
    master = (
        features.merge(labels, on='feature_date', how='inner', validate='one_to_one')
        .sort_values('target_date')
        .reset_index(drop=True)
    )
    master[STAGE1_TARGET_STATE_FEATURE] = master['rolling_volatility'].astype(float)
    master = master.replace([np.inf, -np.inf], np.nan)
    master = master.dropna(subset=STAGE1_FEATURE_COLUMNS).copy()
    master = master.loc[pd.to_datetime(master['target_date']) <= cutoff].copy()
    keep = [
        'feature_date',
        'target_date',
        *STAGE1_FEATURE_COLUMNS,
        'future_log_return',
        'rolling_volatility',
        'threshold',
        'direction',
    ]
    return master[keep].sort_values('target_date').reset_index(drop=True)


def stage1_fold_data(
    stage1_data: pd.DataFrame,
    stage2_outer_train: pd.DataFrame,
    stage2_outer_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(stage2_outer_train['target_date'].max())
    test_dates = pd.DatetimeIndex(pd.to_datetime(stage2_outer_test['target_date']))
    train = stage1_data.loc[
        pd.to_datetime(stage1_data['target_date']) <= train_end
    ].reset_index(drop=True)
    test_lookup = stage1_data.set_index(pd.to_datetime(stage1_data['target_date']))
    missing = test_dates.difference(test_lookup.index)
    if len(missing) > 0:
        raise ValueError(
            'Stage-1 development data is missing Stage-2 outer-test dates: '
            + ', '.join(str(value.date()) for value in missing[:5])
        )
    test = test_lookup.loc[test_dates].reset_index(drop=True)
    if not pd.DatetimeIndex(pd.to_datetime(test['target_date'])).equals(test_dates):
        raise ValueError('Stage-1 outer-test dates do not align to Stage-2 outer-test dates.')
    return train, test


def reconstruct_stage2_true_move(
    saved: dict,
    stage2_outer_test: pd.DataFrame,
) -> pd.DataFrame:
    test_move = stage2_outer_test.loc[
        stage2_outer_test['direction'].astype(str) != 'FLAT'
    ].sort_values('target_date').reset_index(drop=True)
    saved_dates = pd.DatetimeIndex(pd.to_datetime(saved['target_dates']))
    expected_dates = pd.DatetimeIndex(pd.to_datetime(test_move['target_date']))
    if not saved_dates.equals(expected_dates):
        raise ValueError('Saved Stage-2 OOF dates do not align to the reconstructed outer fold.')
    actual = np.asarray(saved['actual'], dtype=np.int64)
    expected_actual = (test_move['direction'].astype(str) == 'UP').astype(int).to_numpy()
    if not np.array_equal(actual, expected_actual):
        raise ValueError('Saved Stage-2 actual labels do not match the locked target.')
    return pd.DataFrame({
        'target_date': expected_dates,
        'actual_up': actual,
        'score': np.asarray(saved['score'], dtype=np.float64),
        'future_log_return': test_move['future_log_return'].astype(float).to_numpy(),
    })


def completed_fold_map() -> dict[int, dict]:
    payload = load_json(PROGRESS_PATH, {'folds': []})
    return {int(row['outer_fold']): row for row in payload.get('folds', [])}


def main():
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    locked_stage1 = load_locked_stage1()
    saved_stage2 = load_stage2_saved_oof()
    stage2_data, winner_groups, winner_features, cutoff = build_stage2_development()
    stage1_data = build_stage1_development(cutoff)

    print('=' * 88)
    print('STAGE-2 ROUTE COMPATIBILITY DIAGNOSTIC V1')
    print('=' * 88)
    print(f'Target: {TARGET_WINDOW}d x {TARGET_MULTIPLIER:.3f}')
    print('Question: does Stage 1 route a materially different label space into Stage 2 than Stage 2 was trained on?')
    print(f'Stage-1 model: locked xLSTM parameters for {TARGET_NAME}')
    print('Stage-2 model: existing nested OOF xgboost_binary_winner predictions')
    print(f'Winner feature groups: {list(winner_groups)}')
    print(f'Winner model features: {len(winner_features)}')
    print(f'Development cutoff: {cutoff.date()}')
    print()
    print('DEVELOPMENT ONLY: Stage-1 thresholds are selected inside each outer fold from earlier inner OOF predictions.')
    print('No Stage-2 model is retrained or tuned. No new feature, target, regime, or threshold search occurs.')
    print('Outer validation and the final held-out test are NOT loaded or evaluated.')

    research = Stage2RouteCompatibilityResearch(
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_block_length=BOOTSTRAP_BLOCK_LENGTH,
        random_state=RANDOM_STATE,
    )
    splitter = TimeSeriesSplit(n_splits=OUTER_SPLITS)
    completed = completed_fold_map()
    route_parts = []
    routed_stage2_parts = []
    high_stage2_parts = []
    fold_rows = []

    for outer_fold, (train_index, test_index) in enumerate(splitter.split(stage2_data), start=1):
        stage2_outer_train = stage2_data.iloc[train_index].reset_index(drop=True)
        stage2_outer_test = stage2_data.iloc[test_index].reset_index(drop=True)
        stage1_train, stage1_test = stage1_fold_data(stage1_data, stage2_outer_train, stage2_outer_test)
        saved = saved_stage2[outer_fold]

        print()
        print(f'outer fold {outer_fold}/{OUTER_SPLITS}')
        if outer_fold in completed:
            cached = completed[outer_fold]
            print('  using cached Stage-1 route predictions')
            route = pd.DataFrame(cached['route_predictions'])
            route['target_date'] = pd.to_datetime(route['target_date'])
            route['feature_date'] = pd.to_datetime(route['feature_date'])
            stage1_threshold = float(cached['stage1_threshold'])
        else:
            print('  selecting Stage-1 threshold from outer-train inner OOF...')
            inner = evaluate_candidate(
                training_data=stage1_train,
                feature_columns=STAGE1_FEATURE_COLUMNS,
                parameters=dict(locked_stage1['parameters']),
            )
            stage1_threshold = float(inner['decision_threshold'])
            print(f'  train-only Stage-1 threshold: {stage1_threshold:.6f}')
            print('  fitting locked Stage 1 for outer-test routing...')
            stage1_result = train_stage1_fold(
                fold_train=stage1_train,
                fold_validation=stage1_test,
                feature_columns=STAGE1_FEATURE_COLUMNS,
                parameters=dict(locked_stage1['parameters']),
                seed=RANDOM_STATE + 50000 + outer_fold,
            )
            result_dates = pd.DatetimeIndex(pd.to_datetime(stage1_result['target_dates']))
            expected_dates = pd.DatetimeIndex(pd.to_datetime(stage1_test['target_date']))
            if not result_dates.equals(expected_dates):
                raise ValueError(f'Stage-1 fold {outer_fold} prediction dates do not align.')
            route = stage1_test[
                ['feature_date', 'target_date', 'future_log_return', 'direction']
            ].copy()
            route['stage1_move_probability'] = np.asarray(
                stage1_result['move_probabilities'], dtype=np.float64
            )
            route['stage1_threshold'] = stage1_threshold
            route['stage1_predicted_move'] = (
                route['stage1_move_probability'] >= stage1_threshold
            )
            regime_lookup = stage2_outer_test.set_index(pd.to_datetime(stage2_outer_test['target_date']))
            route[REGIME_FEATURE] = regime_lookup.loc[
                pd.DatetimeIndex(pd.to_datetime(route['target_date'])), REGIME_FEATURE
            ].to_numpy(dtype=np.float64)

            completed[outer_fold] = {
                'outer_fold': outer_fold,
                'stage1_threshold': stage1_threshold,
                'route_predictions': route.assign(
                    feature_date=route['feature_date'].astype(str),
                    target_date=route['target_date'].astype(str),
                ).to_dict(orient='records'),
            }
            save_json(PROGRESS_PATH, {'folds': [completed[key] for key in sorted(completed)]})

        stage2_true_move = reconstruct_stage2_true_move(saved, stage2_outer_test)
        training_move = stage2_outer_train.loc[
            stage2_outer_train['direction'].astype(str) != 'FLAT'
        ].reset_index(drop=True)
        high_threshold = float(
            pd.to_numeric(training_move[REGIME_FEATURE], errors='raise').quantile(REGIME_QUANTILE)
        )

        route['stage1_threshold'] = stage1_threshold
        route['high_volatility'] = route[REGIME_FEATURE].astype(float) > high_threshold

        fold_result = research.evaluate_fold(
            outer_fold=outer_fold,
            routed_test=route,
            stage2_true_move=stage2_true_move,
            high_volatility_threshold=high_threshold,
            stage2_decision_threshold=float(saved['decision_threshold']),
        )
        fold_rows.append(fold_result)
        route['outer_fold'] = outer_fold
        route_parts.append(route)

        route_lookup = route.set_index(pd.to_datetime(route['target_date']))
        move = stage2_true_move.copy()
        move_dates = pd.DatetimeIndex(pd.to_datetime(move['target_date']))
        move['stage1_predicted_move'] = route_lookup.loc[move_dates, 'stage1_predicted_move'].astype(bool).to_numpy()
        move['high_volatility'] = route_lookup.loc[move_dates, 'high_volatility'].astype(bool).to_numpy()
        move['predicted_up'] = (
            move['score'].astype(float).to_numpy() >= float(saved['decision_threshold'])
        ).astype(int)
        move['correct'] = move['predicted_up'].astype(int) == move['actual_up'].astype(int)
        move['outer_fold'] = outer_fold
        routed_stage2_parts.append(move.loc[move['stage1_predicted_move']].copy())
        high_stage2_parts.append(
            move.loc[move['stage1_predicted_move'] & move['high_volatility']].copy()
        )

        print(
            f"  routed rows {fold_result['predicted_move_rows']} | "
            f"MOVE purity {fold_result['route_move_purity']:.2%} | "
            f"FLAT contamination {fold_result['route_flat_contamination']:.2%} | "
            f"routed true-MOVE Stage-2 AUC {fold_result['stage2_auc_routed_true_move']:.4f}"
        )
        print(
            f"  HIGH-vol routed directional accuracy {fold_result['high_vol_direction_accuracy']:.4f} | "
            f"coverage {fold_result['high_vol_directional_coverage']:.2%}"
        )

    fold_frame = pd.DataFrame(fold_rows).sort_values('outer_fold').reset_index(drop=True)
    routes = pd.concat(route_parts, ignore_index=True).sort_values('target_date').reset_index(drop=True)
    routed_stage2 = pd.concat(routed_stage2_parts, ignore_index=True).sort_values('target_date').reset_index(drop=True)
    high_stage2 = pd.concat(high_stage2_parts, ignore_index=True).sort_values('target_date').reset_index(drop=True)
    summary = research.pooled_summary(routes, fold_frame, routed_stage2, high_stage2)

    print()
    print('DEVELOPMENT ROUTE-COMPATIBILITY RESULTS')
    print(
        fold_frame[
            [
                'outer_fold',
                'rows',
                'predicted_move_rows',
                'routed_true_move_rows',
                'routed_flat_rows',
                'route_move_purity',
                'route_flat_contamination',
                'true_move_recall',
                'stage2_auc_all_true_move',
                'stage2_auc_routed_true_move',
                'high_vol_direction_accuracy',
                'high_vol_directional_coverage',
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print('POOLED DEVELOPMENT ROUTING')
    print(f"Rows: {summary['rows']}")
    print(f"True MOVE rows: {summary['true_move_rows']}")
    print(f"Stage-1 predicted MOVE rows: {summary['predicted_move_rows']}")
    print(f"Routed true MOVE rows: {summary['routed_true_move_rows']}")
    print(f"Routed FLAT rows: {summary['routed_flat_rows']}")
    print(f"Stage-1 MOVE-route purity: {summary['route_move_purity']:.2%}")
    print(f"Stage-1 MOVE-route FLAT contamination: {summary['route_flat_contamination']:.2%}")
    print(f"True-MOVE recall into Stage 2: {summary['true_move_recall']:.2%}")
    print(f"Current routed directional-output accuracy: {summary['routed_direction_accuracy']:.4f}")
    print()
    print('HIGH-VOL SELECTIVE ROUTE')
    print(f"Accepted routed rows: {summary['high_vol_routed_rows']}")
    print(f"Accepted true MOVE rows: {summary['high_vol_routed_true_move_rows']}")
    print(f"Accepted false-routed FLAT rows: {summary['high_vol_routed_flat_rows']}")
    print(f"Accepted route MOVE purity: {summary['high_vol_route_move_purity']:.2%}")
    print(f"Accepted route FLAT contamination: {summary['high_vol_route_flat_contamination']:.2%}")
    print(f"Directional coverage: {summary['high_vol_directional_coverage']:.2%}")
    print(f"End-to-end directional-output accuracy: {summary['high_vol_direction_accuracy']:.4f}")
    print(f"Routed true-MOVE AUC: {summary['routed_true_move_auc']:.4f}")
    print(
        'Routed true-MOVE bootstrap 95% AUC CI: '
        f"[{summary['routed_true_move_auc_lower_95']:.4f}, {summary['routed_true_move_auc_upper_95']:.4f}]"
    )
    print(f"HIGH-vol routed true-MOVE AUC: {summary['high_vol_routed_true_move_auc']:.4f}")
    print(
        'HIGH-vol routed true-MOVE bootstrap 95% AUC CI: '
        f"[{summary['high_vol_routed_true_move_auc_lower_95']:.4f}, {summary['high_vol_routed_true_move_auc_upper_95']:.4f}]"
    )

    print()
    print('ARCHITECTURAL COMPATIBILITY CHECK')
    print(
        '- false_move_routing_present_in_all_folds:',
        'YES' if summary['false_move_routing_present_in_all_folds'] else 'NO',
    )
    print(
        '- label_space_mismatch (Stage 2 has no FLAT class but receives FLAT rows):',
        'YES' if summary['label_space_mismatch'] else 'NO',
    )
    print(
        '  DEVELOPMENT CONFIRMS ROUTE-COMPATIBILITY PROBLEM:',
        'YES' if summary['development_confirms_route_compatibility_problem'] else 'NO',
    )

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    fold_path = EXPERIMENT_DIRECTORY / f'{EXPERIMENT_NAME}_fold_results_{timestamp}.csv'
    route_path = EXPERIMENT_DIRECTORY / f'{EXPERIMENT_NAME}_route_predictions_{timestamp}.csv'
    routed_stage2_path = EXPERIMENT_DIRECTORY / f'{EXPERIMENT_NAME}_routed_stage2_{timestamp}.csv'
    experiment_path = EXPERIMENT_DIRECTORY / f'{EXPERIMENT_NAME}_{timestamp}.json'
    fold_frame.to_csv(fold_path, index=False)
    routes.to_csv(route_path, index=False)
    routed_stage2.to_csv(routed_stage2_path, index=False)

    payload = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment_name': EXPERIMENT_NAME,
        'target': {
            'name': TARGET_NAME,
            'volatility_window': TARGET_WINDOW,
            'threshold_multiplier': TARGET_MULTIPLIER,
        },
        'development_cutoff': cutoff,
        'stage1': {
            'source': STAGE1_PROGRESS_PATH,
            'parameters': locked_stage1['parameters'],
            'outer_fold_threshold_policy': '3-fold inner OOF threshold selection using locked parameters',
        },
        'stage2': {
            'source': TREE_PROGRESS,
            'architecture': 'xgboost_binary_winner',
            'new_training': False,
        },
        'summary': summary,
        'methodology': {
            'development_only': True,
            'new_feature_search': False,
            'new_target_search': False,
            'new_regime_search': False,
            'stage2_hyperparameter_search': False,
            'outer_validation_loaded': False,
            'held_out_test_loaded': False,
            'purpose': (
                'Determine whether the production Stage1->Stage2 route presents Stage 2 '
                'with FLAT observations that are outside its binary UP/DOWN training label space.'
            ),
        },
        'outputs': {
            'fold_results': fold_path,
            'route_predictions': route_path,
            'routed_stage2': routed_stage2_path,
            'experiment': experiment_path,
            'progress': PROGRESS_PATH,
        },
    }
    save_json(experiment_path, payload)

    print()
    print('Fold results:', fold_path)
    print('Route predictions:', route_path)
    print('Routed Stage-2 predictions:', routed_stage2_path)
    print('Experiment:', experiment_path)
    print('Progress checkpoint:', PROGRESS_PATH)
    print('Outer validation was NOT evaluated.')
    print('Held-out test set was NOT evaluated.')
    print()
    if summary['development_confirms_route_compatibility_problem']:
        print(
            'NEXT DECISION RULE: CONFIRMED. Stage 2 is trained on oracle true-MOVE rows but '
            'production routing sends it both MOVE and FLAT rows. The next architecture must '
            'be developed on Stage-1-routed OOF samples and include FLAT in its training label space. '
            'Do not tune against outer validation.'
        )
    else:
        print(
            'NEXT DECISION RULE: NOT CONFIRMED. Do not redesign Stage 2 around the consumed '
            'outer-validation routing pattern without development evidence.'
        )


if __name__ == '__main__':
    main()
