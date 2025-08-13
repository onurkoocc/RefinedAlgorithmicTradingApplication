import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
import joblib
from typing import List, Dict, Any, Optional, Tuple
import gc


class XGBoostFeatureSelector:
    def __init__(self, config, data_preparer=None):
        self.config = config
        self.data_preparer = data_preparer
        self.logger = logging.getLogger("XGBoostFeatureSelector")

        self.max_features = config.get("feature_engineering", "max_features", 48)
        self.min_features = 20
        self.importance_threshold = config.get("feature_engineering", "xgb_importance_threshold", 0.001)
        self.n_splits = config.get("feature_engineering", "xgb_cv_splits", 3)
        self.use_gain_importance = config.get("feature_engineering", "xgb_use_gain", True)

        self.essential_features = config.get("feature_engineering", "essential_features", [])

        self.results_dir = Path(config.results_dir)
        self.feature_dir = self.results_dir / "feature_selection"
        self.feature_dir.mkdir(exist_ok=True, parents=True)

        self.best_features_path = self.feature_dir / "xgb_best_features.json"
        self.importance_scores_path = self.feature_dir / "xgb_importance_scores.json"
        self.model_path = self.feature_dir / "xgb_selector_model.pkl"

        self.best_features = None
        self.importance_scores = {}
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'device': 'cuda' if config.get("model", "use_gpu", True) else 'cpu'
        }

    def optimize_features(self, df: pd.DataFrame) -> List[str]:
        try:
            self.logger.info("Starting XGBoost feature optimization")

            if len(df) < 1000:
                self.logger.warning("Insufficient data for feature selection, using essential features only")
                return self._get_available_essential_features(df)

            df_clean = self._prepare_data_for_selection(df)
            if df_clean is None or len(df_clean) < 1000:
                return self._get_available_essential_features(df)

            if len(df_clean.columns) == 0:
                self.logger.error("No valid columns after data preparation")
                return self._get_available_essential_features(df)

            X, y = self._create_features_and_target(df_clean)
            if X is None or y is None or len(X) < 1000:
                return self._get_available_essential_features(df)

            if len(X.columns) == 0:
                self.logger.error("No features available for importance calculation")
                return self._get_available_essential_features(df)

            feature_importance = self._calculate_feature_importance(X, y)

            if not feature_importance:
                self.logger.warning("Failed to calculate feature importance, using all available features")
                available_features = list(X.columns)
                essential_first = []
                for feature in self.essential_features:
                    if feature in available_features:
                        essential_first.append(feature)
                for feature in available_features:
                    if feature not in essential_first and len(essential_first) < self.max_features:
                        essential_first.append(feature)
                return essential_first[:self.max_features]

            selected_features = self._select_features_by_importance(feature_importance)

            if selected_features:
                validation_score = self._validate_feature_set(X, y, selected_features)
                self.logger.info(f"Validation score with selected features: {validation_score:.6f}")
            else:
                self.logger.warning("No features selected, using essential features")
                selected_features = self._get_available_essential_features(df)

            self.best_features = selected_features
            self.importance_scores = feature_importance

            self.save_best_features(selected_features)
            self._save_importance_scores(feature_importance)

            self.logger.info(f"Selected {len(selected_features)} features using XGBoost")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in XGBoost feature optimization: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_available_essential_features(df)

    def _prepare_data_for_selection(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            df_clean = df.copy()

            exclude_columns = ['actual_open', 'actual_high', 'actual_low', 'actual_close']
            feature_columns = [col for col in df_clean.columns if col not in exclude_columns]

            df_clean = df_clean[feature_columns]

            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns

            if len(non_numeric_cols) > 0:
                for col in non_numeric_cols:
                    if col in df_clean.columns:
                        if df_clean[col].dtype == 'object' or df_clean[col].dtype == 'bool':
                            try:
                                df_clean[col] = df_clean[col].astype(np.float32)
                            except:
                                df_clean = df_clean.drop(columns=[col])
                                self.logger.warning(f"Dropped non-numeric column: {col}")

            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isna().sum() > len(df_clean) * 0.5:
                    df_clean.drop(columns=[col], inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)

            if 'close' not in df_clean.columns:
                self.logger.error("Missing 'close' column for target calculation")
                return None

            return df_clean

        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None

    def _create_features_and_target(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        try:
            if 'close' not in df.columns:
                return None, None

            horizon = self.config.get("model", "horizon", 16)
            future_returns = df['close'].shift(-horizon) / df['close'] - 1

            future_returns = future_returns.dropna()

            if len(future_returns) < 1000:
                return None, None

            valid_indices = future_returns.index

            feature_cols = [col for col in df.columns if col != 'close']
            X = df.loc[valid_indices, feature_cols].copy()

            scale_factor = max(future_returns.abs().quantile(0.9), 0.003) * 12
            y = np.clip(future_returns / scale_factor, -1.0, 1.0)

            X = X.fillna(0)
            y = y.fillna(0)

            return X, y

        except Exception as e:
            self.logger.error(f"Error creating features and target: {e}")
            return None, None

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        try:
            self.logger.info(f"Calculating importance for {len(X.columns)} features")

            importance_scores = {}
            feature_names = X.columns.tolist()

            tscv = TimeSeriesSplit(n_splits=self.n_splits)

            importance_accumulator = {feature: [] for feature in feature_names}

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBRegressor(**self.xgb_params)

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=20
                )

                if self.use_gain_importance:
                    importance = model.get_booster().get_score(importance_type='gain')
                else:
                    importance = model.get_booster().get_score(importance_type='weight')

                for feature in feature_names:
                    score = importance.get(feature, 0)
                    importance_accumulator[feature].append(score)

                gc.collect()

            for feature in feature_names:
                scores = importance_accumulator[feature]
                if scores:
                    importance_scores[feature] = np.mean(scores)
                else:
                    importance_scores[feature] = 0.0

            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {k: v / total_importance for k, v in importance_scores.items()}

            if fold == self.n_splits - 1:
                joblib.dump(model, self.model_path)
                self.logger.info(f"Saved XGBoost model to {self.model_path}")

            return importance_scores

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}

    def _select_features_by_importance(self, importance_scores: Dict[str, float]) -> List[str]:
        try:
            selected_features = []

            available_essential = []
            for feature in self.essential_features:
                if feature in importance_scores:
                    selected_features.append(feature)
                    available_essential.append(feature)

            self.logger.info(f"Added {len(available_essential)} essential features")

            if not importance_scores:
                self.logger.warning("No importance scores available, using essential features only")
                if not selected_features:
                    price_features = ['open', 'high', 'low', 'close', 'volume']
                    for feature in price_features:
                        if feature in importance_scores:
                            selected_features.append(feature)
                return selected_features[:self.max_features]

            remaining_features = [(f, s) for f, s in importance_scores.items()
                                  if f not in selected_features and s > self.importance_threshold]

            remaining_features.sort(key=lambda x: x[1], reverse=True)

            max_additional = self.max_features - len(selected_features)

            if max_additional > 0:
                additional_features = [f for f, _ in remaining_features[:max_additional]]
                selected_features.extend(additional_features)
                self.logger.info(f"Added {len(additional_features)} high-importance features")

            if len(selected_features) < self.min_features:
                still_needed = self.min_features - len(selected_features)
                remaining = [f for f, _ in remaining_features if f not in selected_features]

                if remaining:
                    selected_features.extend(remaining[:still_needed])
                else:
                    all_features = list(importance_scores.keys())
                    for feature in all_features:
                        if feature not in selected_features and len(selected_features) < self.min_features:
                            selected_features.append(feature)

            if not selected_features:
                self.logger.error("No features selected, using default feature set")
                default_features = ['open', 'high', 'low', 'close', 'volume',
                                    'ema_9', 'ema_21', 'ema_50', 'rsi_14', 'atr_14']
                for feature in default_features:
                    if feature in importance_scores:
                        selected_features.append(feature)

            self.logger.info(f"Final feature count: {len(selected_features)}")

            return selected_features[:self.max_features]

        except Exception as e:
            self.logger.error(f"Error selecting features: {e}")
            return self.essential_features

    def _validate_feature_set(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> float:
        try:
            if not selected_features:
                return float('inf')

            available_features = [f for f in selected_features if f in X.columns]
            if not available_features:
                self.logger.error("No selected features available in dataframe")
                return float('inf')

            X_selected = X[available_features]

            tscv = TimeSeriesSplit(n_splits=2)
            scores = []

            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    tree_method='hist'
                )

                model.fit(X_train, y_train, verbose=False)
                predictions = model.predict(X_val)

                mse = mean_squared_error(y_val, predictions)
                scores.append(mse)

            return np.mean(scores)

        except Exception as e:
            self.logger.error(f"Error validating feature set: {e}")
            return float('inf')

    def _get_available_essential_features(self, df: pd.DataFrame) -> List[str]:
        available_features = []
        df_columns = set(df.columns)

        for feature in self.essential_features:
            if feature in df_columns:
                available_features.append(feature)
            elif f'm30_{feature}' in df_columns:
                available_features.append(f'm30_{feature}')

        return available_features[:self.max_features]

    def load_best_features(self) -> Optional[List[str]]:
        try:
            if self.best_features_path.exists():
                with open(self.best_features_path, 'r') as f:
                    data = json.load(f)
                    self.best_features = data.get('features', [])
                    self.logger.info(f"Loaded {len(self.best_features)} best features from {self.best_features_path}")
                    return self.best_features
        except Exception as e:
            self.logger.warning(f"Error loading best features: {e}")

        return None

    def save_best_features(self, features: List[str]) -> None:
        try:
            data = {
                'features': features,
                'timestamp': pd.Timestamp.now().isoformat(),
                'method': 'xgboost',
                'n_features': len(features)
            }

            with open(self.best_features_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Saved {len(features)} best features to {self.best_features_path}")

        except Exception as e:
            self.logger.error(f"Error saving best features: {e}")

    def _save_importance_scores(self, scores: Dict[str, float]) -> None:
        try:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            data = {
                'scores': dict(sorted_scores),
                'timestamp': pd.Timestamp.now().isoformat(),
                'top_features': [f for f, _ in sorted_scores[:20]]
            }

            with open(self.importance_scores_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Saved importance scores to {self.importance_scores_path}")

        except Exception as e:
            self.logger.error(f"Error saving importance scores: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        if self.importance_scores:
            return self.importance_scores

        try:
            if self.importance_scores_path.exists():
                with open(self.importance_scores_path, 'r') as f:
                    data = json.load(f)
                    return data.get('scores', {})
        except Exception as e:
            self.logger.warning(f"Error loading importance scores: {e}")

        return {}