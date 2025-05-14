import traceback

import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import joblib
from pathlib import Path
import os
from typing import List, Optional, Dict, Any

# For growth_score evaluation, we might need a simplified model trainer
# This is a placeholder; in a real scenario, you might import parts of your main model.py
# or have a dedicated lightweight model for feature selection.
from sklearn.linear_model import Ridge  # Using a simpler model for faster Optuna trials if not using growth_score


class OptunaFeatureSelector:
    def __init__(self, config, data_preparer_instance=None):
        self.config = config
        self.data_preparer = data_preparer_instance
        self.logger = logging.getLogger("OptunaFeatureSelector")

        self.results_dir = Path(config.results_dir) / "optuna_features"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        fe_config = config.get("feature_engineering", {})
        model_config = config.get("model", {})

        self.n_trials = fe_config.get("optuna_n_trials", 20)
        self.timeout_seconds = fe_config.get("optuna_timeout", 1800)
        self.study_name = "feature_selection_study_v2"  # Changed name to avoid conflicts with old studies
        self.metric_to_optimize = fe_config.get("optuna_metric", "r2_score")  # growth_score, r2, directional_accuracy

        self.essential_features = fe_config.get("essential_features", [])
        self.max_features_target = model_config.get("max_features", 60)

        self.precomputed_importances = None  # Pandas Series: index=feature_name, value=importance
        self.horizon = model_config.get("horizon", 12)

        # For growth_score simulation in objective
        self.sim_transaction_cost = config.get("backtest", "fixed_cost", 0.001)  # Use a backtest cost
        self.sim_trade_periods_per_month = (24 * 30 * 60) / 30  # Assuming 30min candles for monthly extrapolation

    def _precompute_importances(self, df_all_features: pd.DataFrame):
        self.logger.info("Pre-computing feature importances for Optuna guidance...")
        # data_preparer might not be available if this is run standalone.
        # We need 'actual_close' for the target.
        if 'actual_close' not in df_all_features.columns:
            self.logger.error("'actual_close' not in DataFrame for importance calculation. Cannot precompute.")
            self.precomputed_importances = pd.Series(dtype=float)
            return

        try:
            price_col = 'actual_close'
            y_target = df_all_features[price_col].pct_change(periods=self.horizon).shift(-self.horizon).fillna(0)

            X_features = df_all_features.copy()
            # Drop actuals, target, and any non-numeric or problematic columns before importance calculation
            cols_to_drop = [col for col in X_features.columns if
                            col.startswith('actual_') or col == price_col or col == 'market_regime_type']
            X_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')

            # Select only numeric types for RF
            X_features = X_features.select_dtypes(include=np.number)
            X_features = X_features.fillna(0).replace([np.inf, -np.inf], 0)

            # Align X and y (y_target is already shifted, X_features needs to match its length from the start)
            y_target = y_target.iloc[self.horizon:]  # y_target starts after horizon shift
            X_features = X_features.iloc[:-self.horizon]  # X_features ends before horizon makes y_target NaN

            # Further align if lengths don't match due to other operations
            min_len = min(len(X_features), len(y_target))
            X_features = X_features.iloc[:min_len]
            y_target = y_target.iloc[:min_len]

            if X_features.empty or len(X_features.columns) == 0 or len(y_target) == 0:
                self.logger.warning("No valid features/target left for importance calculation.")
                self.precomputed_importances = pd.Series(dtype=float)
                return

            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, min_samples_leaf=10)
            rf.fit(X_features, y_target)

            self.precomputed_importances = pd.Series(rf.feature_importances_, index=X_features.columns).sort_values(
                ascending=False)
            self.logger.info(f"Top 10 precomputed importances: \n{self.precomputed_importances.head(10)}")
        except Exception as e:
            self.logger.error(f"Error during pre-computation of feature importances: {e}\n{traceback.format_exc()}")
            self.precomputed_importances = pd.Series(dtype=float)

    def optimize_features(self, df_all_features: pd.DataFrame) -> List[str]:
        if df_all_features.empty or len(df_all_features.columns) <= len(self.essential_features):
            self.logger.warning("Not enough features or data for Optuna optimization. Returning essentials.")
            return self.essential_features.copy()  # Return a copy

        self._precompute_importances(df_all_features)

        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            storage=f"sqlite:///{self.results_dir}/optuna_study.db",
            load_if_exists=True
        )

        study.optimize(lambda trial: self._objective(trial, df_all_features),
                       n_trials=self.n_trials,
                       timeout=self.timeout_seconds,
                       show_progress_bar=True)

        best_trial = study.best_trial
        self.logger.info(f"Optuna Best Trial {best_trial.number}: Value={best_trial.value:.4f}")

        # Reconstruct the best feature set from params
        # Ensure all potential features are considered for selection based on trial params
        all_potential_features = [f for f in df_all_features.columns if
                                  f not in self.essential_features and not f.startswith(
                                      'actual_') and f != 'market_regime_type']
        all_potential_features = [f for f in all_potential_features if
                                  df_all_features[f].nunique() > 1]  # Only use features with variance

        best_features_set = self.essential_features.copy()
        for feature_name in all_potential_features:
            param_name = f"feature_{feature_name.replace('_', '').replace('-', '').replace('.', '')}"  # Sanitize name
            if best_trial.params.get(param_name, 0.0) > 0.5:  # Threshold for selection probability
                if feature_name not in best_features_set:
                    best_features_set.append(feature_name)

        # Ensure max_features_target is respected, prioritizing by importance
        if len(best_features_set) > self.max_features_target:
            if self.precomputed_importances is not None and not self.precomputed_importances.empty:
                non_essentials_in_set = [f for f in best_features_set if f not in self.essential_features]

                # Filter precomputed_importances to only include those in non_essentials_in_set
                relevant_importances = self.precomputed_importances[
                    self.precomputed_importances.index.isin(non_essentials_in_set)]
                sorted_relevant_non_essentials = relevant_importances.sort_values(ascending=False)

                num_to_keep = self.max_features_target - len(self.essential_features)
                if num_to_keep < 0: num_to_keep = 0  # Should not happen if essentials < max_features

                final_best_features = self.essential_features + sorted_relevant_non_essentials.head(
                    num_to_keep).index.tolist()
            else:  # Fallback if no importances, just truncate
                self.logger.warning(
                    "Trimming selected features without importance scores due to missing precomputation.")
                non_essentials_in_set = [f for f in best_features_set if f not in self.essential_features]
                num_to_keep = self.max_features_target - len(self.essential_features)
                final_best_features = self.essential_features + non_essentials_in_set[:num_to_keep]
        else:
            final_best_features = best_features_set

        # Remove duplicates just in case
        final_best_features = sorted(list(set(final_best_features)))

        self.logger.info(f"Optuna selected {len(final_best_features)} features: {final_best_features}")
        self._save_best_features(final_best_features)
        return final_best_features

    def _objective(self, trial: optuna.Trial, df_all_features: pd.DataFrame) -> float:
        current_selected_features = self.essential_features.copy()

        potential_add_features = [f for f in df_all_features.columns if
                                  f not in self.essential_features and not f.startswith(
                                      'actual_') and f != 'market_regime_type']
        potential_add_features = [f for f in potential_add_features if
                                  df_all_features[f].nunique() > 1]  # Only suggest features with variance

        for feature_name in potential_add_features:
            param_name = f"feature_{feature_name.replace('_', '').replace('-', '').replace('.', '')}"  # Sanitize

            # Suggest a float between 0 and 1. We'll use a threshold (e.g., 0.5) to decide inclusion.
            # The default value can be guided by precomputed_importances if available.
            default_suggestion = 0.5
            if self.precomputed_importances is not None and feature_name in self.precomputed_importances:
                norm_imp = self.precomputed_importances[feature_name]
                if self.precomputed_importances.max() > 0: norm_imp /= self.precomputed_importances.max()
                default_suggestion = np.clip(0.1 + 0.8 * norm_imp, 0.0, 1.0)  # Scale importance to a suggestion prior

            selection_strength = trial.suggest_float(param_name, 0.0, 1.0,
                                                     step=0.1)  # Suggest a "strength" or probability

            if selection_strength > 0.5:  # Threshold for including the feature
                if feature_name not in current_selected_features:
                    current_selected_features.append(feature_name)

        if not current_selected_features or len(current_selected_features) < 3:  # Need at least a few features
            return -10.0

            # --- Evaluation based on self.metric_to_optimize ---
        try:
            X_eval = df_all_features[current_selected_features].copy()
            X_eval = X_eval.fillna(0).replace([np.inf, -np.inf], 0)

            if 'actual_close' not in df_all_features.columns:
                self.logger.error("Objective: 'actual_close' missing.")
                return -10.0

            price_col = 'actual_close'
            # Target: scaled percentage change for regression, raw for growth_score PnL
            y_target_pct_change_raw = df_all_features[price_col].pct_change(periods=self.horizon).shift(
                -self.horizon).fillna(0)

            # Align X and y
            X_eval = X_eval.iloc[:-self.horizon]
            y_target_pct_change_raw_aligned = y_target_pct_change_raw.iloc[:-self.horizon]

            if len(X_eval) < 100: return -9.0

            split_idx = int(0.7 * len(X_eval))  # 70/30 split for this quick eval
            X_train, X_test = X_eval.iloc[:split_idx], X_eval.iloc[split_idx:]
            y_train_raw, y_test_raw = y_target_pct_change_raw_aligned.iloc[
                                      :split_idx], y_target_pct_change_raw_aligned.iloc[split_idx:]

            if len(X_train) < 20 or len(X_test) < 20: return -8.0

            # Model for evaluation
            # Using Ridge regression for speed and simplicity in feature selection objective
            eval_model = Ridge(alpha=1.0, random_state=trial.number)
            eval_model.fit(X_train, y_train_raw)  # Train on raw pct_change
            predictions_raw_pct = eval_model.predict(X_test)

            if self.metric_to_optimize == "r2_score":
                score = r2_score(y_test_raw, predictions_raw_pct)
            elif self.metric_to_optimize == "directional_accuracy":
                true_direction = (y_test_raw > 0)
                pred_direction = (predictions_raw_pct > 0)
                score = accuracy_score(true_direction, pred_direction)
            elif self.metric_to_optimize == "growth_score":
                # Simplified growth score simulation
                trade_returns_pct_sim = []
                equity_curve_sim = [1.0]
                # Decide trades based on predicted direction and a minimal magnitude
                # This threshold is on raw predicted % change, not scaled model output
                sim_pred_threshold = 0.0005  # e.g., predict at least 0.05% move

                for k in range(len(predictions_raw_pct)):
                    pred_val = predictions_raw_pct[k]
                    actual_ret = y_test_raw.iloc[k]
                    if abs(pred_val) > sim_pred_threshold:
                        trade_pnl = np.sign(pred_val) * actual_ret - self.sim_transaction_cost
                        trade_returns_pct_sim.append(trade_pnl)
                        equity_curve_sim.append(equity_curve_sim[-1] * (1 + trade_pnl))

                if not trade_returns_pct_sim:
                    score = -1.0
                else:
                    n_sim_trades = len(trade_returns_pct_sim)
                    avg_pnl_sim = np.mean(trade_returns_pct_sim)

                    trades_per_test_period = n_sim_trades / len(X_test) if len(X_test) > 0 else 0
                    trades_per_month_sim = trades_per_test_period * self.sim_trade_periods_per_month

                    if 1 + avg_pnl_sim > 0 and trades_per_month_sim > 0:
                        monthly_growth_sim = ((1 + avg_pnl_sim) ** trades_per_month_sim) - 1
                    else:
                        monthly_growth_sim = avg_pnl_sim * trades_per_month_sim

                    peak_sim = np.maximum.accumulate(equity_curve_sim)
                    drawdown_sim = (peak_sim - equity_curve_sim) / peak_sim
                    max_dd_sim = np.max(drawdown_sim) if len(drawdown_sim) > 0 else 1.0

                    # Simplified score: reward growth, penalize drawdown
                    score = monthly_growth_sim - (max_dd_sim * 2)  # Example: DD is twice as bad as growth is good
                    if trades_per_month_sim < 5 or trades_per_month_sim > 50: score -= 0.1  # Penalize too few/many trades
            else:  # Default to R2 if metric unknown
                score = r2_score(y_test_raw, predictions_raw_pct)

            # Penalize for too many or too few features relative to target
            num_features = len(current_selected_features)
            if num_features > self.max_features_target + 10:  # Allow some leeway over max
                score -= (num_features - (self.max_features_target + 10)) * 0.01
            elif num_features < max(5, self.max_features_target // 3):  # Penalize if too few
                score -= 0.1

            return score if not np.isnan(score) else -10.0

        except Exception as e:
            self.logger.error(
                f"Error in Optuna objective evaluation (Trial {trial.number}): {e}\n{traceback.format_exc()}")
            return -10.0  # Large negative for errors

    def _save_best_features(self, features: List[str]):
        path = self.results_dir / "optimized_feature_set.joblib"
        try:
            joblib.dump(features, path)
            self.logger.info(f"Saved best {len(features)} features to {path}")
        except Exception as e:
            self.logger.error(f"Error saving best features: {e}")

    def load_best_features(self) -> Optional[List[str]]:
        path = self.results_dir / "optimized_feature_set.joblib"
        if path.exists():
            try:
                features = joblib.load(path)
                self.logger.info(f"Loaded {len(features)} optimized features from {path}")
                return features
            except Exception as e:
                self.logger.error(f"Error loading optimized features: {e}")
        self.logger.info("No pre-optimized feature set found.")
        return None