import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

        self.n_trials = 30
        self.study_name = "feature_optimization"
        self.direction = "maximize"
        self.metric = "growth_score"

        self.essential_features = config.get("feature_engineering", "essential_features", [
            "open", "high", "low", "close", "volume",
            "ema_9", "ema_21", "ema_50", "sma_200",
            "rsi_14", "bb_width_20", "atr_14",
            "macd_histogram_12_26_9", "market_regime"
        ])

        self.max_features = config.get("model", "max_features", 70)
        self.score_threshold = 0.001
        self.best_features = None
        self.feature_importances = {}

    def optimize_features(self, df_features, model=None):
        if df_features is None or len(df_features) < 300:
            self.logger.warning("Insufficient data for feature optimization")
            return self.essential_features

        self.logger.info(f"Starting feature optimization with Optuna ({self.n_trials} trials)")
        self._precompute_feature_importances(df_features)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=f"sqlite:///{self.results_dir}/feature_study.db",
            load_if_exists=True
        )

        study.optimize(
            lambda trial: self._objective(trial, df_features, model),
            n_trials=self.n_trials,
            timeout=3600,
            show_progress_bar=True
        )

        best_trial = study.best_trial
        self.logger.info(f"Best trial: {best_trial.number}, Value: {best_trial.value}")
        self.logger.info(f"Best params: {best_trial.params}")

        all_features = self._get_all_possible_features(df_features)
        selected_features = [f for i, f in enumerate(all_features)
                             if f in self.essential_features or
                             best_trial.params.get(f"use_{i}", False)]

        if len(selected_features) > self.max_features:
            sorted_features = sorted(
                [(f, self.feature_importances.get(f, 0)) for f in selected_features
                 if f not in self.essential_features],
                key=lambda x: x[1], reverse=True
            )
            n_additional = self.max_features - len(self.essential_features)
            additional_features = [f[0] for f in sorted_features[:n_additional]]
            selected_features = self.essential_features + additional_features

        self.best_features = selected_features
        self._save_best_features(selected_features)
        return selected_features

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

    def _precompute_feature_importances(self, df_features):
        target_col = 'close'

        if target_col not in df_features.columns:
            self.logger.warning(f"Target column {target_col} not found. Using first column.")
            target_col = df_features.columns[0]

        split_idx = int(0.8 * len(df_features))
        train_df = df_features.iloc[:split_idx].copy()

        horizon = 16
        y = train_df[target_col].pct_change(horizon).shift(-horizon).fillna(0).values
        X = train_df.copy()

        cols_to_drop = [col for col in X.columns
                        if col == target_col or X[col].nunique() <= 1 or X[col].isna().all()]
        X = X.drop(columns=cols_to_drop)

        try:
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X.fillna(0), y)

            self.feature_importances = {col: imp for col, imp in zip(X.columns, model.feature_importances_)}
            joblib.dump(self.feature_importances, self.results_dir / "feature_importances.joblib")

            top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:20]
            self.logger.info(f"Top 20 features by importance: {top_features}")

        except Exception as e:
            self.logger.error(f"Error precomputing feature importances: {e}")
            self.feature_importances = {}

    def _objective(self, trial, df_features, model=None):
        all_features = self._get_all_possible_features(df_features)
        selected_features = self.essential_features.copy()

        for i, feature in enumerate(all_features):
            if feature not in self.essential_features:
                importance = self.feature_importances.get(feature, 0)

                # Fixed: Removed choices_weights parameter
                use_feature = trial.suggest_categorical(f"use_{i}", [True, False])

                if use_feature:
                    selected_features.append(feature)

        if len(selected_features) < 5:
            return -1.0

        if len(selected_features) > self.max_features:
            non_essential_features = [f for f in selected_features if f not in self.essential_features]
            sorted_by_importance = sorted(
                [(f, self.feature_importances.get(f, 0)) for f in non_essential_features],
                key=lambda x: x[1], reverse=True
            )
            n_additional = self.max_features - len(self.essential_features)
            additional_features = [f[0] for f in sorted_by_importance[:n_additional]]
            selected_features = self.essential_features + additional_features

        score = self._evaluate_feature_set(df_features[selected_features], df_features, model)

        trial.set_user_attr("n_features", len(selected_features))
        trial.set_user_attr("features", selected_features)

        return score

    def _evaluate_feature_set(self, X, full_df, model=None):
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

        X = X.fillna(0)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        eval_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

        try:
            eval_model.fit(X_train, y_train)
            y_pred = eval_model.predict(X_test)

            direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            score = (direction_accuracy * 0.7) + (r2 * 0.3)
            feature_penalty = len(X.columns) / (self.max_features * 2)
            regularized_score = score * (1 - min(0.3, feature_penalty))

            return regularized_score

        except Exception as e:
            self.logger.error(f"Error in feature evaluation: {e}")
            return -1.0

    def _get_all_possible_features(self, df_features):
        cols = df_features.columns.tolist()
        return [col for col in cols if not col.startswith('actual_') and not col.startswith('timestamp')]

    def _save_best_features(self, features):
        try:
            joblib.dump(features, self.results_dir / "optimized_features.joblib")
            with open(self.results_dir / "optimized_features.txt", "w") as f:
                f.write("\n".join(features))
            self.logger.info(f"Saved {len(features)} optimized features")
        except Exception as e:
            self.logger.error(f"Error saving optimized features: {e}")