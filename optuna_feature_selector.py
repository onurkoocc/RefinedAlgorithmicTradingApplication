from datetime import datetime

import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib
from pathlib import Path
import os


class OptunaFeatureSelector:
    def __init__(self, config, data_preparer=None):
        self.config = config
        self.data_preparer = data_preparer
        self.logger = logging.getLogger("OptunaFeatureSelector")
        self.results_dir = Path(config.results_dir) / "models"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.n_trials = config.get("feature_engineering", "optuna_n_trials", 100)
        self.timeout = config.get("feature_engineering", "optuna_timeout", 3600)
        self.study_name = "feature_optimization"
        self.direction = "maximize"
        self.metric = "growth_score"
        self.n_jobs = -1

        self.essential_features = config.get("feature_engineering", "essential_features", [
            "open", "high", "low", "close", "volume",
            "ema_9", "ema_21", "ema_50", "sma_200",
            "rsi_14", "bb_width_20", "atr_14",
            "macd_histogram_12_26_9", "market_regime",
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'
        ])

        self.max_features = config.get("model", "max_features", 40)
        self.correlation_threshold = config.get("feature_engineering", "correlation_threshold", 0.9)
        self.score_threshold = 0.001
        self.best_features = None
        self.feature_importances = {}
        self.feature_groups = self._create_feature_groups()

    def _create_feature_groups(self):
        groups = {
            "ema": [f for f in self.essential_features if "ema_" in f],
            "macd": [f for f in self.essential_features if "macd" in f],
            "bb": [f for f in self.essential_features if "bb_" in f],
            "time": [f for f in self.essential_features if "sin" in f or "cos" in f],
            "momentum": ["rsi_14", "adx_14", "cmf_20"]
        }
        return groups

    def optimize_features(self, df_features, model=None, force_new=True):
        if df_features is None or len(df_features) < 300:
            self.logger.warning("Insufficient data for feature optimization")
            return self.essential_features

        self.logger.info(f"Starting feature optimization with Optuna ({self.n_trials} trials)")

        df_features = self._remove_correlated_features(df_features)
        self._precompute_feature_importances(df_features)

        # For per-iteration optimization, create a new study each time
        if force_new:
            study_name = f"feature_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            study_name = self.study_name

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)

        storage_path = f"sqlite:///{self.results_dir}/feature_study.db"
        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            storage=storage_path,
            load_if_exists=not force_new,
            pruner=pruner,
            sampler=sampler
        )

        try:
            study.optimize(
                lambda trial: self._objective(trial, df_features, model),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
                n_jobs=1
            )

            best_trial = study.best_trial
            self.logger.info(f"Best trial: {best_trial.number}, Value: {best_trial.value}")
            self.logger.info(f"Best params: {best_trial.params}")

            all_features = self._get_all_possible_features(df_features)
            selected_features = self._get_selected_features_from_params(best_trial.params, all_features)

            if len(selected_features) > self.max_features:
                non_essential = [f for f in selected_features if f not in self.essential_features]
                sorted_features = sorted(
                    [(f, self.feature_importances.get(f, 0)) for f in non_essential],
                    key=lambda x: x[1], reverse=True
                )

                remaining_slots = self.max_features - len(self.essential_features)
                additional_features = [f[0] for f in sorted_features[:remaining_slots]]
                selected_features = self.essential_features + additional_features

            self.best_features = selected_features
            self._save_best_features(selected_features)
            return selected_features

        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return self.essential_features

    def _remove_correlated_features(self, df):
        if len(df.columns) <= len(self.essential_features):
            return df

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        to_drop = [col for col in to_drop if col not in self.essential_features]

        self.logger.info(f"Removing {len(to_drop)} highly correlated features")
        return df.drop(columns=to_drop)

    def _precompute_feature_importances(self, df_features):
        target_col = 'close'
        if target_col not in df_features.columns:
            target_col = df_features.columns[0]
            self.logger.warning(f"Target column 'close' not found. Using {target_col} instead.")

        horizon = 16
        y = df_features[target_col].pct_change(horizon).shift(-horizon).fillna(0).values
        X = df_features.copy()

        cols_to_drop = [col for col in X.columns
                        if col == target_col or col.startswith('actual_') or
                        X[col].nunique() <= 1 or X[col].isna().all()]
        X = X.drop(columns=cols_to_drop)

        X = X.fillna(method='ffill').fillna(0)

        tscv = TimeSeriesSplit(n_splits=5)
        importances = np.zeros(X.shape[1])

        params = {
            'n_estimators': 50,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'regression',
            'verbose': -1,  # Suppress output
            'min_child_samples': 20,  # Avoid overfitting
            'min_split_gain': 0.01  # Minimum gain required for split
        }

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for train_idx, _ in tscv.split(X):
                model = lgb.LGBMRegressor(**params)
                model.fit(X.iloc[train_idx], y[train_idx])
                importances += model.feature_importances_

        importances /= 5

        self.feature_importances = {col: imp for col, imp in zip(X.columns, importances)}
        joblib.dump(self.feature_importances, self.results_dir / "feature_importances.joblib")

        top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:25]
        self.logger.info(f"Top 25 features by importance: {top_features}")

    def _objective(self, trial, df_features, model=None):
        all_features = self._get_all_possible_features(df_features)

        selected_features = self.essential_features.copy()

        for group_name, features in self.feature_groups.items():
            if trial.suggest_categorical(f"use_group_{group_name}", [True, False]):
                for feature in features:
                    if feature not in selected_features and feature in all_features:
                        selected_features.append(feature)

        for feature in all_features:
            if feature not in selected_features and not feature.startswith('actual_'):
                importance = self.feature_importances.get(feature, 0)

                suggest_prob = min(0.8, max(0.2, importance * 10))
                if trial.suggest_float(f"use_{feature}", 0, 1) < suggest_prob:
                    selected_features.append(feature)

        if len(selected_features) < 5:
            return -1.0

        if len(selected_features) > self.max_features:
            non_essential = [f for f in selected_features if f not in self.essential_features]
            sorted_by_importance = sorted(
                [(f, self.feature_importances.get(f, 0)) for f in non_essential],
                key=lambda x: x[1], reverse=True
            )
            n_additional = self.max_features - len(self.essential_features)
            additional_features = [f[0] for f in sorted_by_importance[:n_additional]]
            selected_features = self.essential_features + additional_features

        try:
            score = self._evaluate_feature_set(df_features[selected_features], df_features, trial)

            trial.set_user_attr("n_features", len(selected_features))
            trial.set_user_attr("features", selected_features)

            return score
        except Exception as e:
            self.logger.warning(f"Error evaluating feature set: {e}")
            return -1.0

    def _evaluate_feature_set(self, X, full_df, trial):
        if X.empty or len(X.columns) < 5:
            return -1.0

        horizon = 16
        if 'close' in full_df.columns:
            y = full_df['close'].pct_change(horizon).shift(-horizon).fillna(0).values
        else:
            y = full_df.iloc[:, 0].pct_change(horizon).shift(-horizon).fillna(0).values

        X = X.iloc[:-horizon].copy()
        y = y[:-horizon]

        if len(X) < 100:
            return -1.0

        X = X.fillna(method='ffill').fillna(0)

        tscv = TimeSeriesSplit(n_splits=3, test_size=len(X) // 5)

        # Modify LightGBM parameters to avoid split warnings
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 30, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'objective': 'regression',
            'random_state': 42,
            'min_child_samples': 20,  # Increase to avoid overfitting on small samples
            'min_split_gain': 0.01,  # Set minimum gain required for split
            'verbose': -1,  # Suppress LightGBM warnings
            'min_data_in_leaf': 10  # Ensure enough data in each leaf
        }

        direction_accuracy_scores = []
        r2_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # Use a context manager to suppress specific warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    eval_model = lgb.LGBMRegressor(**params)
                    eval_model.fit(X_train, y_train)

                    y_pred = eval_model.predict(X_test)

                    direction_match = np.sign(y_pred) == np.sign(y_test)
                    direction_accuracy = np.mean(direction_match)
                    r2 = r2_score(y_test, y_pred) if not np.isnan(y_test).any() and not np.isnan(y_pred).any() else 0

                    direction_accuracy_scores.append(direction_accuracy)
                    r2_scores.append(r2)
            except Exception as e:
                self.logger.warning(f"Error in cross-validation: {e}")
                continue

        if not direction_accuracy_scores:
            return -1.0

        avg_direction_accuracy = np.mean(direction_accuracy_scores)
        avg_r2 = np.mean(r2_scores)

        score = (avg_direction_accuracy * 0.7) + (avg_r2 * 0.3)

        feature_penalty = len(X.columns) / (self.max_features * 2) * (1 - score)
        regularized_score = score * (1 - min(0.3, feature_penalty))

        return regularized_score

    def _get_all_possible_features(self, df_features):
        cols = df_features.columns.tolist()
        return [col for col in cols if not col.startswith('actual_') and not col.startswith('timestamp')]

    def _get_selected_features_from_params(self, params, all_features):
        selected_features = self.essential_features.copy()

        for group_name, features in self.feature_groups.items():
            if params.get(f"use_group_{group_name}", False):
                for feature in features:
                    if feature not in selected_features and feature in all_features:
                        selected_features.append(feature)

        for feature in all_features:
            if feature not in selected_features and not feature.startswith('actual_'):
                feature_key = f"use_{feature}"
                if feature_key in params and params[feature_key] < 0.5:
                    selected_features.append(feature)

        return selected_features

    def _save_best_features(self, features):
        try:
            joblib.dump(features, self.results_dir / "optimized_features.joblib")
            with open(self.results_dir / "optimized_features.txt", "w") as f:
                f.write("\n".join(features))
            self.logger.info(f"Saved {len(features)} optimized features")
        except Exception as e:
            self.logger.error(f"Error saving optimized features: {e}")

    def load_best_features(self):
        feature_path = self.results_dir / "optimized_features.joblib"
        if os.path.exists(feature_path):
            try:
                self.best_features = joblib.load(feature_path)
                self.logger.info(f"Loaded {len(self.best_features)} optimized features")
                return self.best_features
            except Exception as e:
                self.logger.error(f"Error loading optimized features: {e}")
        return None