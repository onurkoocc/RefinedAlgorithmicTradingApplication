import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
from pathlib import Path
import os
from typing import List, Optional


class OptunaFeatureSelector:
    def __init__(self, config, data_preparer_instance=None):
        self.config = config
        self.data_preparer = data_preparer_instance
        self.logger = logging.getLogger("OptunaFeatureSelector")

        self.results_dir = Path(config.results_dir) / "optuna_features"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.n_trials = config.get("feature_engineering", "optuna_n_trials", 20)
        self.timeout_seconds = config.get("feature_engineering", "optuna_timeout", 1800)
        self.study_name = "feature_selection_study"
        self.metric_to_optimize = config.get("feature_engineering", "optuna_metric", "r2_score")

        self.essential_features = config.get("feature_engineering", "essential_features", [])
        self.max_features_target = config.get("model", "max_features", 60)

        self.precomputed_importances = None

    def _precompute_importances(self, df_all_features: pd.DataFrame):
        self.logger.info("Pre-computing feature importances for Optuna guidance...")
        if self.data_preparer is None:
            self.logger.warning("DataPreparer instance not provided, cannot create labels for importance calculation.")
            self.precomputed_importances = {}
            return

        try:
            if 'actual_close' not in df_all_features.columns:
                self.logger.error("'actual_close' not in DataFrame for importance calculation.")
                self.precomputed_importances = {}
                return

            price_col = 'actual_close'
            horizon = self.config.get("model", "horizon", 12)
            y_target = df_all_features[price_col].pct_change(periods=horizon).shift(-horizon).fillna(0)

            X_features = df_all_features.iloc[:-horizon].copy()
            y_target = y_target.iloc[:-horizon]

            cols_to_drop = [col for col in X_features.columns if
                            col.startswith('actual_') or col == price_col or col == 'market_regime_type']
            X_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            X_features = X_features.fillna(0).replace([np.inf, -np.inf], 0)

            if X_features.empty or len(X_features.columns) == 0:
                self.logger.warning("No valid features left for importance calculation.")
                self.precomputed_importances = {}
                return

            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_features, y_target)

            self.precomputed_importances = pd.Series(rf.feature_importances_, index=X_features.columns).sort_values(
                ascending=False)
            self.logger.info(f"Top 10 precomputed importances: \n{self.precomputed_importances.head(10)}")
        except Exception as e:
            self.logger.error(f"Error during pre-computation of feature importances: {e}")
            self.precomputed_importances = {}

    def optimize_features(self, df_all_features: pd.DataFrame) -> List[str]:
        if df_all_features.empty or len(df_all_features.columns) <= len(self.essential_features):
            self.logger.warning("Not enough features or data for Optuna optimization. Returning essentials.")
            return self.essential_features

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

        potential_features = [f for f in df_all_features.columns if
                              f not in self.essential_features and not f.startswith(
                                  'actual_') and f != 'market_regime_type']

        selected_from_trial = self.essential_features.copy()
        for feature_name in potential_features:
            param_name = f"feature_{feature_name.replace('_', '').replace('-', '')}"
            if best_trial.params.get(param_name, False):
                selected_from_trial.append(feature_name)

        if len(selected_from_trial) > self.max_features_target:
            if self.precomputed_importances is not None and not self.precomputed_importances.empty:
                non_essentials_in_trial = [f for f in selected_from_trial if f not in self.essential_features]
                sorted_non_essentials = self.precomputed_importances[
                    self.precomputed_importances.index.isin(non_essentials_in_trial)].sort_values(ascending=False)

                num_to_keep = self.max_features_target - len(self.essential_features)
                best_features = self.essential_features + sorted_non_essentials.head(num_to_keep).index.tolist()
            else:
                best_features = selected_from_trial[:self.max_features_target]
        else:
            best_features = selected_from_trial

        self.logger.info(f"Optuna selected {len(best_features)} features: {best_features}")
        self._save_best_features(best_features)
        return best_features

    def _objective(self, trial: optuna.Trial, df_all_features: pd.DataFrame) -> float:
        current_selected_features = self.essential_features.copy()

        potential_add_features = [f for f in df_all_features.columns if
                                  f not in self.essential_features and not f.startswith(
                                      'actual_') and f != 'market_regime_type']

        for feature_name in potential_add_features:
            param_name = f"feature_{feature_name.replace('_', '').replace('-', '')}"
            importance_score = 0.5
            if self.precomputed_importances is not None and feature_name in self.precomputed_importances:
                norm_imp = self.precomputed_importances[feature_name]
                if self.precomputed_importances.max() > 0: norm_imp /= self.precomputed_importances.max()
                importance_score = 0.1 + 0.8 * norm_imp

            if trial.suggest_categorical(param_name, [True, False]):
                current_selected_features.append(feature_name)

        if not current_selected_features: return -1.0

        try:
            X_eval = df_all_features[current_selected_features].copy()
            X_eval = X_eval.fillna(0).replace([np.inf, -np.inf], 0)

            if 'actual_close' not in df_all_features.columns:
                self.logger.error("Objective function: 'actual_close' missing for target generation.")
                return -1.0

            price_col = 'actual_close'
            horizon = self.config.get("model", "horizon", 12)
            y_target = df_all_features[price_col].pct_change(periods=horizon).shift(-horizon).fillna(0)

            X_eval = X_eval.iloc[:-horizon]
            y_target = y_target.iloc[:-horizon]

            if len(X_eval) < 100: return -0.9

            split_idx = int(0.7 * len(X_eval))
            X_train, X_test = X_eval.iloc[:split_idx], X_eval.iloc[split_idx:]
            y_train, y_test = y_target.iloc[:split_idx], y_target.iloc[split_idx:]

            if len(X_train) < 20 or len(X_test) < 20: return -0.8

            model = RandomForestRegressor(n_estimators=30, max_depth=7, random_state=trial.number, n_jobs=-1,
                                          min_samples_leaf=5)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            score = r2_score(y_test, predictions)

            feature_count_penalty = 0.0
            if len(current_selected_features) > self.max_features_target:
                feature_count_penalty = (len(current_selected_features) - self.max_features_target) * 0.01

            final_score = score - feature_count_penalty
            return final_score

        except Exception as e:
            self.logger.error(f"Error in Optuna objective evaluation: {e}")
            return -10.0

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
        return None