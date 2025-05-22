import traceback
import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
import os
from typing import List, Optional, Dict, Any, Union

import lightgbm as lgb
from sklearn.inspection import permutation_importance


class OptunaFeatureSelector:
    def __init__(self, config, data_preparer_instance=None):
        self.config = config
        self.data_preparer = data_preparer_instance
        self.logger = logging.getLogger("OptunaFeatureSelector")

        self.results_dir = Path(config.results_dir) / "optuna_features"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.optuna_db_path = self.results_dir / "optuna_studies.db"

        fe_config = config.get("feature_engineering", {})
        model_config = config.get("model", {})

        self.n_trials = fe_config.get("optuna_n_trials", 150)
        self.timeout_seconds = fe_config.get("optuna_timeout", 3600)
        self.base_study_name = fe_config.get("optuna_study_name", "feature_selection_study_v5_sharpe")
        self.optuna_objective_model = fe_config.get("optuna_objective_model", "RandomForest")

        self.essential_features = fe_config.get("essential_features",
                                                ['open', 'high', 'low', 'close', 'volume', 'sma_200', 'obv', 'ema_50',
                                                 'atr_14', 'rsi_14', 'adx_14', 'range_position'])
        self.ignore_features = fe_config.get("optuna_ignore_features", [])  # NEW

        self.optuna_n_additional_features_min = fe_config.get("optuna_n_additional_features_min", 8)
        max_model_features = model_config.get("max_features", 60)
        default_max_additional = max(0, max_model_features - len(self.essential_features) - 5)
        self.optuna_n_additional_features_max = fe_config.get("optuna_n_additional_features_max",
                                                              default_max_additional)
        if self.optuna_n_additional_features_max < self.optuna_n_additional_features_min:
            self.optuna_n_additional_features_max = self.optuna_n_additional_features_min

        self.current_window_importances = None
        self.horizon = model_config.get("horizon", 12)

        self.sim_transaction_cost = config.get("backtest", "fixed_cost", 0.001) + config.get("backtest",
                                                                                             "variable_cost", 0.0005)

        self.optuna_sim_trade_threshold_percentile = fe_config.get("optuna_sim_trade_threshold_percentile", 70)
        self.optuna_min_trades_for_score = fe_config.get("optuna_min_trades_for_score", 25)
        self.optuna_feature_count_penalty_factor = fe_config.get("optuna_feature_count_penalty_factor", 0.003)
        self.optuna_stability_penalty_factor = fe_config.get("optuna_stability_penalty_factor", 0.1)

        self.lgbm_n_estimators = fe_config.get("optuna_lgbm_n_estimators", 600)
        self.lgbm_learning_rate = fe_config.get("optuna_lgbm_learning_rate", 0.02)
        self.lgbm_min_child_samples = fe_config.get("optuna_lgbm_min_child_samples", 20)
        self.lgbm_early_stopping_rounds = fe_config.get("optuna_lgbm_early_stopping_rounds", 60)
        self.optuna_rf_n_estimators = fe_config.get("optuna_rf_n_estimators", 75)
        self.optuna_rf_max_depth = fe_config.get("optuna_rf_max_depth", 9)

    def _calculate_window_feature_importances(self, df_window_features: pd.DataFrame):
        self.logger.info(
            "Calculating feature importances for the current window using LightGBM and Permutation Importance...")
        if 'actual_close' not in df_window_features.columns:
            self.logger.error("'actual_close' not in DataFrame for importance calculation.")
            self.current_window_importances = pd.Series(dtype=float)
            return

        try:
            price_col = 'actual_close'
            y_target = df_window_features[price_col].pct_change(periods=self.horizon).shift(-self.horizon)

            X_features = df_window_features.copy()
            cols_to_drop = [col for col in X_features.columns if
                            col.startswith('actual_') or col == price_col or col == 'market_regime_type']
            X_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            X_features = X_features.select_dtypes(include=np.number)

            valid_indices = y_target.dropna().index
            X_features_aligned = X_features.loc[valid_indices].fillna(0).replace([np.inf, -np.inf], 0)
            y_target_aligned = y_target.loc[valid_indices].fillna(0)

            if X_features_aligned.empty or len(X_features_aligned.columns) == 0 or len(y_target_aligned) < 50:
                self.logger.warning("Insufficient valid data for importance calculation.")
                self.current_window_importances = pd.Series(dtype=float)
                return

            X_train_lgbm_fi, X_val_lgbm_fi, y_train_lgbm_fi, y_val_lgbm_fi = train_test_split(
                X_features_aligned, y_target_aligned, test_size=0.25, shuffle=False
            )

            eval_set_lgbm = None
            callbacks_lgbm = None
            if len(X_train_lgbm_fi) >= 20 and len(X_val_lgbm_fi) >= 20:
                eval_set_lgbm = [(X_val_lgbm_fi, y_val_lgbm_fi)]
                callbacks_lgbm = [lgb.early_stopping(stopping_rounds=self.lgbm_early_stopping_rounds, verbose=-1)]
            else:
                self.logger.warning(
                    "LGBM train/val split resulted in small sets. Using full data for LGBM training, permutation on training data.")
                X_train_lgbm_fi, y_train_lgbm_fi = X_features_aligned, y_target_aligned

            lgbm_model = lgb.LGBMRegressor(
                n_estimators=self.lgbm_n_estimators, max_depth=7, learning_rate=self.lgbm_learning_rate,
                n_jobs=-1, random_state=42, min_child_samples=self.lgbm_min_child_samples,
                colsample_bytree=0.8, subsample=0.8, importance_type='gain',
                min_split_gain=self.config.get("feature_engineering", "optuna_lgbm_min_split_gain", 0.0)
            )
            lgbm_model.fit(X_train_lgbm_fi, y_train_lgbm_fi, eval_set=eval_set_lgbm, eval_metric='l1',
                           callbacks=callbacks_lgbm)

            X_perm_data = X_val_lgbm_fi if eval_set_lgbm and not X_val_lgbm_fi.empty else X_train_lgbm_fi
            y_perm_data = y_val_lgbm_fi if eval_set_lgbm and not y_val_lgbm_fi.empty else y_train_lgbm_fi

            if len(X_perm_data) < 2:
                self.logger.warning("Permutation importance data is too small. Skipping.")
                self.current_window_importances = pd.Series(dtype=float)
                return

            perm_importance_result = permutation_importance(
                lgbm_model, X_perm_data, y_perm_data,
                n_repeats=5, random_state=42, n_jobs=-1, scoring='r2'
            )

            self.current_window_importances = pd.Series(
                perm_importance_result.importances_mean, index=X_perm_data.columns
            ).sort_values(ascending=False)

            self.logger.info(
                f"Top 10 permutation importances for current window: \n{self.current_window_importances.head(10)}")

        except Exception as e:
            self.logger.error(f"Error during window feature importance calculation: {e}\n{traceback.format_exc()}")
            self.current_window_importances = pd.Series(dtype=float)

    def _get_candidate_additional_features(self, df_window_features: pd.DataFrame) -> pd.Series:
        idx = self.current_window_importances.index
        mask_is_non_essential = ~idx.isin(self.essential_features)
        mask_is_not_ignored = ~idx.isin(self.ignore_features)  # NEW MASK
        mask_is_in_df = idx.isin(df_window_features.columns)

        mask_has_variance = pd.Series(False, index=idx)
        candidate_for_variance_check = idx[mask_is_non_essential & mask_is_not_ignored & mask_is_in_df]

        for feature_name_iter in candidate_for_variance_check:
            if df_window_features[feature_name_iter].nunique() > 1:
                mask_has_variance.loc[feature_name_iter] = True

        final_selection_mask = mask_is_non_essential & mask_is_not_ignored & mask_is_in_df & mask_has_variance
        return self.current_window_importances[final_selection_mask]

    def optimize_features(self, df_window_features: pd.DataFrame, window_id: Union[int, str]) -> List[str]:
        if df_window_features.empty:
            self.logger.warning(f"Window {window_id}: Empty data for Optuna optimization. Returning essentials.")
            return self.essential_features[:]

        self._calculate_window_feature_importances(df_window_features)

        if self.current_window_importances is None or self.current_window_importances.empty:
            self.logger.warning(
                f"Window {window_id}: Feature importances could not be computed. Returning essential features.")
            return self.essential_features[:]

        candidate_additional_features = self._get_candidate_additional_features(df_window_features)
        potential_add_features_count = len(candidate_additional_features)

        current_max_additional = min(self.optuna_n_additional_features_max, potential_add_features_count)
        current_min_additional = min(self.optuna_n_additional_features_min, current_max_additional)

        if current_min_additional > current_max_additional:
            current_min_additional = current_max_additional

        if current_max_additional == 0:
            self.logger.info(
                f"Window {window_id}: No additional non-essential, non-ignored features with variance to select. Returning essentials.")
            return self.essential_features[:]

        window_specific_study_name = f"{self.base_study_name}_win_{window_id}"

        study = None
        try:
            study = optuna.load_study(study_name=window_specific_study_name, storage=f"sqlite:///{self.optuna_db_path}")
            self.logger.info(f"Window {window_id}: Loaded existing Optuna study '{window_specific_study_name}'.")
            if study.best_trial and self.config.get("feature_engineering", "optuna_reuse_existing_study_results", True):
                self.logger.info(f"Reusing best trial {study.best_trial.number} with value {study.best_value:.4f}.")
            else:
                self.logger.info(f"Optimizing study '{window_specific_study_name}'.")
                study.optimize(
                    lambda trial: self._objective(trial, df_window_features, candidate_additional_features,
                                                  current_min_additional,
                                                  current_max_additional, window_id),
                    n_trials=self.n_trials, timeout=self.timeout_seconds, show_progress_bar=True
                )
        except KeyError:
            self.logger.info(f"Window {window_id}: Creating new Optuna study '{window_specific_study_name}'.")
            study = optuna.create_study(
                study_name=window_specific_study_name, direction="maximize",
                storage=f"sqlite:///{self.optuna_db_path}", load_if_exists=True
            )
            study.optimize(
                lambda trial: self._objective(trial, df_window_features, candidate_additional_features,
                                              current_min_additional, current_max_additional,
                                              window_id),
                n_trials=self.n_trials, timeout=self.timeout_seconds, show_progress_bar=True
            )
        except Exception as e:
            self.logger.error(
                f"Window {window_id}: Error with Optuna study '{window_specific_study_name}': {e}. Returning essentials.")
            return self.essential_features[:]

        if not study or not study.best_trial:
            self.logger.warning(
                f"Window {window_id}: Optuna study completed without a best trial. Returning essential features.")
            return self.essential_features[:]

        best_trial = study.best_trial
        self.logger.info(f"Window {window_id}: Optuna Best Trial {best_trial.number}: Value={best_trial.value:.4f}")

        n_best_additional_features = best_trial.params.get("n_additional_features", current_min_additional)
        best_features_set = self.essential_features[:]
        top_n_additional = candidate_additional_features.head(n_best_additional_features).index.tolist()

        for f_name in top_n_additional:
            if f_name not in best_features_set:
                best_features_set.append(f_name)

        final_best_features = sorted(list(set(best_features_set)))

        self.logger.info(
            f"Window {window_id}: Optuna selected {len(final_best_features)} features: {final_best_features}")
        self._save_best_features(final_best_features, filename=f"optimized_feature_set_window_{window_id}.joblib")
        return final_best_features

    def _objective(self, trial: optuna.Trial, df_window_features: pd.DataFrame,
                   candidate_additional_features: pd.Series,
                   min_add: int, max_add: int,
                   window_id: Union[int, str]) -> float:
        current_selected_features = self.essential_features[:]

        if max_add < min_add:
            n_additional = max_add
        elif max_add == min_add:
            n_additional = min_add
        else:
            n_additional = trial.suggest_int("n_additional_features", min_add, max_add)

        additional_to_select = candidate_additional_features.head(n_additional).index.tolist()

        for feature_name in additional_to_select:
            if feature_name not in current_selected_features:
                current_selected_features.append(feature_name)

        current_selected_features = sorted(list(set(current_selected_features)))

        if not current_selected_features or len(current_selected_features) < len(
                self.essential_features):  # Allow only essentials
            if len(current_selected_features) == len(self.essential_features) and min_add == 0 and max_add == 0:
                pass  # This is fine if no additional features are possible/allowed
            else:
                return -10.0

        try:
            X_eval = df_window_features[current_selected_features].copy()
            for col in X_eval.columns:
                if X_eval[col].isnull().all():
                    X_eval[col] = 0
                elif X_eval[col].isnull().any():
                    X_eval[col] = X_eval[col].fillna(X_eval[col].median())
            X_eval = X_eval.fillna(0).replace([np.inf, -np.inf], 0)

            if 'actual_close' not in df_window_features.columns:
                self.logger.error(f"Window {window_id}, Trial {trial.number}: 'actual_close' missing.")
                return -10.0

            price_col = 'actual_close'
            y_target_pct_change_raw = df_window_features[price_col].pct_change(periods=self.horizon).shift(
                -self.horizon)

            valid_indices = y_target_pct_change_raw.dropna().index
            X_eval_aligned = X_eval.loc[valid_indices]
            y_target_aligned = y_target_pct_change_raw.loc[valid_indices].fillna(0)

            if len(X_eval_aligned) < 100: return -9.0

            tscv = TimeSeriesSplit(n_splits=4)
            fold_scores = []

            for train_idx, test_idx in tscv.split(X_eval_aligned):
                X_train, X_test = X_eval_aligned.iloc[train_idx], X_eval_aligned.iloc[test_idx]
                y_train_raw, y_test_raw = y_target_aligned.iloc[train_idx], y_target_aligned.iloc[test_idx]

                if len(X_train) < 30 or len(X_test) < 20:
                    fold_scores.append(-8.0)
                    continue

                if self.optuna_objective_model == "RandomForest":
                    eval_model = RandomForestRegressor(n_estimators=self.optuna_rf_n_estimators,
                                                       max_depth=self.optuna_rf_max_depth,
                                                       random_state=trial.number + len(fold_scores), n_jobs=-1,
                                                       min_samples_leaf=20, min_samples_split=40)
                else:
                    eval_model = Ridge(alpha=trial.suggest_float("ridge_alpha", 0.1, 10.0, log=True),
                                       random_state=trial.number + len(fold_scores))

                eval_model.fit(X_train, y_train_raw)
                predictions_raw_pct = eval_model.predict(X_test)

                if len(predictions_raw_pct) > 5:
                    abs_fold_preds = np.abs(predictions_raw_pct)
                    sim_pred_threshold = np.percentile(abs_fold_preds,
                                                       self.optuna_sim_trade_threshold_percentile) if len(
                        abs_fold_preds) > 0 else 0.0001
                    sim_pred_threshold = max(sim_pred_threshold, 0.0001)

                    fold_trade_pnls = [
                        np.sign(pred) * actual_ret - self.sim_transaction_cost
                        for pred, actual_ret in zip(predictions_raw_pct, y_test_raw)
                        if abs(pred) > sim_pred_threshold
                    ]

                    if len(fold_trade_pnls) >= self.optuna_min_trades_for_score:
                        mean_pnl = np.mean(fold_trade_pnls)
                        std_pnl = np.std(fold_trade_pnls)
                        sharpe_like_score = mean_pnl / (std_pnl + 1e-7)

                        stability_penalty = (std_pnl / (abs(mean_pnl) + 1e-7)) * self.optuna_stability_penalty_factor

                        fold_score = sharpe_like_score - stability_penalty
                    else:
                        fold_score = -1.0
                else:
                    fold_score = -2.0

                fold_scores.append(fold_score)

            final_score = np.mean(fold_scores) if fold_scores else -10.0

            target_feature_count_ideal = (min_add + max_add) / 2.0 + len(self.essential_features)
            feature_count_diff = abs(len(current_selected_features) - target_feature_count_ideal)
            penalty = feature_count_diff * self.optuna_feature_count_penalty_factor
            final_score -= penalty

            return float(final_score) if not np.isnan(final_score) else -10.0

        except Exception as e:
            self.logger.error(
                f"Error in Optuna objective (Window {window_id}, Trial {trial.number}): {e}\n{traceback.format_exc()}")
            return -10.0

    def _save_best_features(self, features: List[str], filename: str = "optimized_feature_set.joblib"):
        path = self.results_dir / filename
        try:
            joblib.dump(features, path)
            self.logger.info(f"Saved {len(features)} features to {path}")
        except Exception as e:
            self.logger.error(f"Error saving features to {path}: {e}")

    def load_best_features(self, filename: str = "optimized_feature_set.joblib") -> Optional[List[str]]:
        path = self.results_dir / filename
        if path.exists():
            try:
                features = joblib.load(path)
                self.logger.info(f"Loaded {len(features)} optimized features from {path}")
                return features
            except Exception as e:
                self.logger.error(f"Error loading optimized features from {path}: {e}")
        self.logger.info(f"No pre-optimized feature set found at {path}.")
        return None