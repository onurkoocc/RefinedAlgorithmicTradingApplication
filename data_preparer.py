import logging
import numpy as np
import pandas as pd
import os
import json
import pickle
import joblib
from typing import Tuple, Optional, Dict, List, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

from optuna_feature_selector import OptunaFeatureSelector


class DataPreparer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataPreparer")

        self.sequence_length = config.get("model", "sequence_length", 72)
        self.horizon = config.get("model", "horizon", 16)
        self.normalize_method = config.get("model", "normalize_method", "feature_specific")
        self.train_ratio = config.get("model", "train_ratio", 0.7)

        self.results_dir = config.results_dir
        self.scaler_path = os.path.join(self.results_dir, "models", "feature_scaler.pkl")
        self.feature_list_path = os.path.join(self.results_dir, "models", "feature_list.json")
        self.importance_path = os.path.join(self.results_dir, "models", "feature_importance.json")

        self.price_column = "close"

        self.scaler = None
        self.scaler_dict = {}
        self.feature_names = None
        self.feature_importance = {}
        self.test_sequence_length = self.sequence_length
        self.normalization_stats = {}

        self.train_features = None
        self.max_features = config.get("model", "max_features", 48)

        self.essential_features = [
            # Core price data
            'open', 'high', 'low', 'close', 'volume',

            # Volume dynamics
            'taker_buy_base_asset_volume', 'cumulative_delta', 'volume_imbalance_ratio',
            'volume_price_momentum',

            # Trend indicators
            'ema_9', 'ema_21', 'ema_50', 'sma_200',
            'adx_14', 'plus_di_14', 'minus_di_14',
            'trend_strength', 'ma_cross_velocity',

            # Momentum oscillators
            'rsi_14', 'rsi_roc_3', 'macd_histogram_12_26_9',

            # Volatility metrics
            'atr_14', 'bb_width_20', 'volatility_regime',

            # Market context
            'market_regime', 'mean_reversion_signal', 'price_impact_ratio',

            # Support/resistance
            'bb_percent_b', 'range_position', 'pullback_strength',

            # Time-based patterns
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'cycle_phase', 'cycle_position',

            # Price action patterns
            'relative_candle_size', 'candle_body_ratio', 'gap',

            # Order flow
            'spread_pct', 'close_vwap_diff',

            # Adaptive volatility features
            'vol_norm_close_change', 'vol_norm_momentum'
        ]

        # Create Optuna feature selector
        self.use_optuna_features = config.get("feature_engineering", "use_optuna_features", False)
        self.optuna_feature_selector = OptunaFeatureSelector(config, self)
        self.optimized_features = None

        self.use_adaptive_features = config.get("feature_engineering", "use_adaptive_features", False)
        self.feature_selection_method = config.get("feature_engineering", "feature_selection_method", "importance")
        self.dynamic_feature_count = config.get("feature_engineering", "dynamic_feature_count", 50)
        self.use_only_essential_features = config.get("feature_engineering", "use_only_essential_features", False)

        self.fallback_indicators = {
            "rsi_14": 50,
            "cci_20": 0,
            "willr_14": -50,
            "macd_histogram_12_26_9": 0,
            "bb_percent_b": 0.5,
            "market_regime": 0,
            "volatility_regime": 0.5,
            "taker_buy_ratio": 0.5,
            "mfi": 50,
            "hour_sin": 0,
            "hour_cos": 1,  # Midnight default
            "day_of_week_sin": 0,
            "day_of_week_cos": 1  # Monday default
        }

        self._load_scaler_and_features()
        self._load_feature_importance()
        self._load_optimized_features()

    def _load_scaler_and_features(self):
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    if isinstance(self.scaler, dict):
                        self.scaler_dict = self.scaler
                self.logger.info(f"Loaded scaler from {self.scaler_path}")

            if os.path.exists(self.feature_list_path):
                with open(self.feature_list_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.train_features = self.feature_names.copy()
                self.logger.info(f"Loaded feature list with {len(self.feature_names)} features")
        except Exception as e:
            self.logger.warning(f"Error loading scaler or feature list: {e}")

    def _load_feature_importance(self):
        try:
            if os.path.exists(self.importance_path):
                with open(self.importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                self.logger.info(f"Loaded feature importance for {len(self.feature_importance)} features")
        except Exception as e:
            self.logger.warning(f"Error loading feature importance: {e}")

    def _load_optimized_features(self):
        if self.use_optuna_features:
            self.optimized_features = self.optuna_feature_selector.load_best_features()
            if self.optimized_features:
                self.logger.info(f"Loaded {len(self.optimized_features)} optimized features")

    def _save_scaler_and_features(self):
        try:
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)

            if self.normalize_method == 'feature_specific' and self.scaler_dict:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler_dict, f)
            elif self.scaler is not None:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            self.logger.info(f"Saved scaler to {self.scaler_path}")

            if self.feature_names is not None:
                with open(self.feature_list_path, 'w') as f:
                    json.dump(self.feature_names, f)
                self.logger.info(f"Saved feature list to {self.feature_list_path}")
        except Exception as e:
            self.logger.error(f"Error saving scaler or feature list: {e}")

    def _save_feature_importance(self):
        try:
            if self.feature_importance:
                with open(self.importance_path, 'w') as f:
                    json.dump(self.feature_importance, f)
                self.logger.info(f"Saved feature importance to {self.importance_path}")
        except Exception as e:
            self.logger.error(f"Error saving feature importance: {e}")

    def prepare_data(self, df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        actual_cols = [col for col in df.columns if col.startswith('actual_')]

        if self.use_optuna_features:
            self.logger.info("Running Optuna feature optimization...")
            self.optimized_features = self.optuna_feature_selector.optimize_features(df)

        # Use Optuna-optimized features if available
        if self.use_optuna_features and self.optimized_features:
            self.train_features = self._filter_available_features(df, self.optimized_features)
            self.logger.info(f"Using {len(self.train_features)} Optuna-optimized features for current iteration")
        else:
            self.train_features = self._get_available_features(df)
            self.logger.info(f"Using {len(self.train_features)} standard features")

        missing_essential = [f for f in self.essential_features if f not in self.train_features]
        if missing_essential:
            self.logger.warning(f"Missing {len(missing_essential)} essential features: {missing_essential}")

        df_features = df[self.train_features].copy()
        df_features = self._clean_dataframe(df_features)

        try:
            min_required_rows = self.sequence_length + self.horizon + 10
            if len(df_features) < min_required_rows:
                self.logger.warning(
                    f"Not enough data for sequence creation: {len(df_features)} rows, need {min_required_rows}")
                return (np.array([]),) * 6

            df_labeled, target_returns, fwd_returns = self._create_regression_labels(df_features)
            df_labeled = self._handle_outliers(df_labeled)

            if self.use_adaptive_features:
                self._calculate_feature_importance(df_labeled, target_returns)

            self.feature_names = df_labeled.columns.tolist()

            # Build sequences first without normalization
            X_features = df_labeled.values.astype(np.float32)
            X_full, y_full, fwd_returns_full = self._build_sequences(X_features, target_returns, fwd_returns)

            if len(X_full) == 0:
                self.logger.warning("No sequences created during processing")
                return (np.array([]),) * 6

            # Split into train and validation sets
            train_size = int(self.train_ratio * len(X_full))
            X_train, X_val = X_full[:train_size], X_full[train_size:]
            y_train, y_val = y_full[:train_size], y_full[train_size:]
            fwd_returns_val = fwd_returns_full[train_size:]

            # Now normalize train and validation separately
            if self.normalize_method != "none":
                # Reshape for normalization
                X_train_reshaped = X_train.reshape(-1, X_train.shape[2])

                # Fit scaler on training data only
                if self.normalize_method == "robust":
                    self.scaler = RobustScaler(quantile_range=(10, 90))
                elif self.normalize_method == "standard":
                    self.scaler = StandardScaler()
                else:
                    # For feature-specific normalization
                    self.scaler_dict = {}
                    for i, feature in enumerate(self.feature_names):
                        col_data = X_train_reshaped[:, i].reshape(-1, 1)

                        if 'rsi' in feature.lower() or 'percent_b' in feature.lower():
                            scaler = MinMaxScaler(feature_range=(0, 1))
                        elif 'regime' in feature.lower():
                            scaler = MinMaxScaler(feature_range=(-1, 1))
                        elif 'momentum' in feature.lower() or 'macd' in feature.lower():
                            scaler = RobustScaler()
                        else:
                            scaler = StandardScaler()

                        scaler.fit(col_data)
                        self.scaler_dict[feature] = scaler

                    X_train = self._apply_feature_specific_normalization(X_train_reshaped, self.feature_names).reshape(
                        X_train.shape)

                    if len(X_val) > 0:
                        X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
                        X_val = self._apply_feature_specific_normalization(X_val_reshaped, self.feature_names).reshape(
                            X_val.shape)

                if not self.normalize_method == "feature_specific":
                    # Transform training data
                    X_train_normalized = self.scaler.fit_transform(X_train_reshaped)
                    X_train = X_train_normalized.reshape(X_train.shape)

                    # Transform validation data using the scaler fit on training data
                    if len(X_val) > 0:
                        X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
                        X_val_normalized = self.scaler.transform(X_val_reshaped)
                        X_val = X_val_normalized.reshape(X_val.shape)

                self._save_scaler_and_features()
                self._save_feature_importance()

            entry_indices = list(range(self.sequence_length - 1, len(df_labeled)))
            val_entry_indices = entry_indices[train_size:]

            if val_entry_indices:
                df_val = df_labeled.iloc[val_entry_indices].copy()

                for col in actual_cols:
                    if col in df.columns:
                        df_val[col] = df[col].iloc[val_entry_indices].values
            else:
                df_val = pd.DataFrame()

            return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

        except Exception as e:
            self.logger.error(f"Error in prepare_data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return (np.array([]),) * 6

    def _filter_available_features(self, df: pd.DataFrame, feature_list: List[str]) -> List[str]:
        """Filter the optimized feature list to include only available features"""
        available_features = []
        df_columns = set(df.columns)

        # First add all essential features that are available
        for feature in self.essential_features:
            if feature in df_columns:
                available_features.append(feature)
            elif f'm30_{feature}' in df_columns:
                available_features.append(f'm30_{feature}')

        # Then add optimized features that are available
        for feature in feature_list:
            if feature in df_columns and feature not in available_features:
                available_features.append(feature)
            elif f'm30_{feature}' in df_columns and f'm30_{feature}' not in available_features:
                available_features.append(f'm30_{feature}')

        return available_features[:self.max_features]

    def _calculate_feature_importance(self, df: pd.DataFrame, target_returns: np.ndarray) -> None:
        try:
            if len(df) < 200 or len(target_returns) < 200:
                return

            X = df.iloc[:len(target_returns)].values
            y = target_returns

            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

            if self.feature_selection_method == 'importance':
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                model.fit(X, y)
                importance = model.feature_importances_
            else:
                importance = mutual_info_regression(X, y, random_state=42)

            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)

            feature_importance = {}
            for i, feature in enumerate(df.columns):
                feature_importance[feature] = float(importance[i])

            self.feature_importance = feature_importance

        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")

    def _get_available_features(self, df: pd.DataFrame) -> List[str]:
        available_features = []

        if self.use_only_essential_features:
            available_essential_features = []
            df_columns = set(df.columns)

            for feature in self.essential_features:
                if feature in df_columns:
                    available_essential_features.append(feature)
                elif f'm30_{feature}' in df_columns:
                    available_essential_features.append(f'm30_{feature}')

            self.logger.info(f"Using only {len(available_essential_features)} essential features")
            return available_essential_features
        for feature in self.essential_features:
            if feature in df.columns:
                available_features.append(feature)
            elif f'm30_{feature}' in df.columns:
                available_features.append(f'm30_{feature}')

        for col in df.columns:
            if col not in available_features and not col.startswith('actual_'):
                available_features.append(col)

        if not self.use_adaptive_features or not self.feature_importance:
            return available_features[:self.max_features]

        scored_features = []
        for feature in available_features:
            importance = self.feature_importance.get(feature, 0.0)
            scored_features.append((feature, importance))

        scored_features.sort(key=lambda x: x[1], reverse=True)

        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        selected_features = [feat for feat in essential_cols if feat in available_features]

        for feature, _ in scored_features:
            if feature not in selected_features and len(selected_features) < self.dynamic_feature_count:
                selected_features.append(feature)

        self.logger.info(f"Selected {len(selected_features)} features using {self.feature_selection_method} method")
        return selected_features

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        actual_cols = [col for col in df.columns if col.startswith('actual_')]

        if self.feature_names:
            required_features = self.feature_names
        else:
            self.logger.warning("No training features found - using essential features only")
            required_features = self.essential_features

        available_features_in_test = set(df.columns)
        missing_features = [f for f in required_features if f not in available_features_in_test]
        if missing_features:
            self.logger.warning(f"Test data missing {len(missing_features)} required features: {missing_features}")

        df_features = pd.DataFrame(index=df.index)
        for feature in required_features:
            if feature in df.columns:
                df_features[feature] = df[feature]
            elif f'm30_{feature}' in df.columns:
                df_features[feature] = df[f'm30_{feature}']
            else:
                df_features[feature] = self._impute_missing_feature(df, feature)

        df_features = self._clean_dataframe(df_features)

        try:
            if len(df_features) < (self.sequence_length + self.horizon):
                self.logger.warning(
                    f"Not enough data for test sequence creation: {len(df_features)} rows, need {self.sequence_length + self.horizon}")
                return np.array([]), np.array([]), df_features, np.array([])

            df_labeled, target_returns, fwd_returns = self._create_regression_labels(df_features)

            df_actual = df_labeled.copy()
            for col in actual_cols:
                if col in df.columns:
                    df_actual[col] = df[col].values[:len(df_labeled)]

            df_labeled = self._handle_outliers(df_labeled)

            X_features = df_labeled.values.astype(np.float32)

            if self.normalize_method != "none":
                if self.normalize_method == 'feature_specific' and self.scaler_dict:
                    X_features = self._apply_feature_specific_normalization(X_features, df_labeled.columns)
                elif self.scaler:
                    if X_features.shape[1] != self.scaler.n_features_in_:
                        self.logger.warning(
                            f"Feature dimension mismatch: test has {X_features.shape[1]}, scaler expects {self.scaler.n_features_in_}")
                        X_features = self._ensure_feature_dimensions(X_features, df_labeled.columns.tolist())

                    try:
                        X_features = self.scaler.transform(X_features)
                    except Exception as e:
                        self.logger.error(f"Error transforming test data: {e}")
                        X_features = (X_features - np.mean(X_features, axis=0)) / np.maximum(np.std(X_features, axis=0),
                                                                                             1e-5)

            X_test, y_test, fwd_returns_test = self._build_sequences(X_features, target_returns, fwd_returns)

            return X_test, y_test, df_actual, fwd_returns_test

        except Exception as e:
            self.logger.error(f"Error in prepare_test_data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([]), np.array([]), df_features, np.array([])

    def _impute_missing_feature(self, df: pd.DataFrame, feature: str) -> pd.Series:
        if feature in self.fallback_indicators:
            return pd.Series(self.fallback_indicators[feature], index=df.index)

        # Time features - calculate them if possible
        if feature in ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']:
            try:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df_with_datetime = df.copy()
                    df_with_datetime.index = pd.to_datetime(df_with_datetime.index)
                    time_features = self.optuna_feature_selector.data_preparer.indicator_util.calculate_time_features(
                        df_with_datetime)
                    if feature in time_features.columns:
                        return time_features[feature]
            except:
                # Fall back to default values
                if feature == 'hour_sin': return pd.Series(0, index=df.index)
                if feature == 'hour_cos': return pd.Series(1, index=df.index)
                if feature == 'day_of_week_sin': return pd.Series(0, index=df.index)
                if feature == 'day_of_week_cos': return pd.Series(1, index=df.index)

        if feature in ['open', 'high', 'low'] and 'close' in df.columns:
            return df['close'].copy()

        if 'macd' in feature.lower() and 'close' in df.columns:
            return df['close'].pct_change(5).fillna(0) * 100

        if 'volume' in feature.lower() and 'volume' in df.columns:
            return df['volume'].copy()

        if 'regime' in feature.lower():
            if 'close' in df.columns and len(df) > 20:
                sma20 = df['close'].rolling(20).mean()
                return ((df['close'] - sma20) / sma20).clip(-1, 1).fillna(0)

        return pd.Series(0.0, index=df.index)

    def _apply_feature_specific_normalization(self, data: np.ndarray, feature_names: pd.Index) -> np.ndarray:
        try:
            result = np.zeros_like(data)

            for i, feature in enumerate(feature_names):
                if feature in self.scaler_dict:
                    scaler = self.scaler_dict[feature]
                    result[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
                else:
                    mean = np.mean(data[:, i])
                    std = np.std(data[:, i])
                    if std > 1e-8:
                        result[:, i] = (data[:, i] - mean) / std
                    else:
                        result[:, i] = data[:, i] - mean

            return result

        except Exception as e:
            self.logger.error(f"Error in feature-specific normalization: {e}")
            return (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 1e-5)

    def _create_regression_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        price = df[self.price_column].copy()
        future_prices = price.shift(-self.horizon)
        fwd_return = (future_prices / price - 1)
        fwd_return = fwd_return.fillna(0)

        abs_returns = fwd_return.abs()
        scale_factor = max(abs_returns.quantile(0.9), 0.003) * 12
        scaled_returns = np.clip(fwd_return / scale_factor, -1.1, 1.1)
        self.return_scale_factor = scale_factor

        valid_length = len(scaled_returns)
        df_valid = df.iloc[:valid_length].copy()

        return df_valid, scaled_returns.values, fwd_return.values

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        for col in df_clean.select_dtypes(include=np.number).columns:
            if col.startswith('actual_'):
                continue

            if col in ['open', 'high', 'low', 'close', 'volume']:
                q1 = df_clean[col].quantile(0.001)
                q3 = df_clean[col].quantile(0.999)
                df_clean[col] = df_clean[col].clip(q1, q3)
            elif 'rsi' in col.lower():
                df_clean[col] = df_clean[col].clip(0, 100)
            elif 'willr' in col.lower():
                df_clean[col] = df_clean[col].clip(-100, 0)
            elif 'bb_percent_b' in col.lower():
                df_clean[col] = df_clean[col].clip(0, 1)
            elif 'regime' in col.lower():
                df_clean[col] = df_clean[col].clip(-1, 1)
            else:
                q1 = df_clean[col].quantile(0.01)
                q3 = df_clean[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - (3 * iqr)
                upper_bound = q3 + (3 * iqr)
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        return df_clean

    def _normalize_features(self, data: np.ndarray, feature_names: pd.Index) -> Tuple[np.ndarray, object]:
        if self.normalize_method == "none":
            return data, None

        try:
            if self.normalize_method == "feature_specific":
                normalized_data = np.zeros_like(data)
                scaler_dict = {}

                for i, feature in enumerate(feature_names):
                    if 'rsi' in feature.lower() or 'percent_b' in feature.lower():
                        scaler = MinMaxScaler(feature_range=(0, 1))
                    elif 'regime' in feature.lower():
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                    elif 'momentum' in feature.lower() or 'macd' in feature.lower():
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()

                    col_data = data[:, i].reshape(-1, 1)

                    try:
                        normalized_data[:, i] = scaler.fit_transform(col_data).flatten()
                        scaler_dict[feature] = scaler
                    except Exception as e:
                        self.logger.warning(f"Error normalizing {feature}: {e}")
                        mean = np.mean(col_data)
                        std = max(np.std(col_data), 1e-8)
                        normalized_data[:, i] = ((col_data - mean) / std).flatten()

                self.scaler_dict = scaler_dict
                return normalized_data, scaler_dict

            elif self.normalize_method == "robust":
                scaler = RobustScaler(quantile_range=(10, 90))
            else:
                scaler = StandardScaler()

            normalized_data = scaler.fit_transform(data)

            if hasattr(scaler, 'center_'):
                self.normalization_stats['center'] = scaler.center_
            if hasattr(scaler, 'scale_'):
                self.normalization_stats['scale'] = scaler.scale_

            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)

            return normalized_data, scaler

        except Exception as e:
            self.logger.error(f"Normalization error: {e}")
            means = np.nanmean(data, axis=0)
            stds = np.nanstd(data, axis=0)
            stds[stds < 1e-8] = 1.0

            normalized_data = (data - means) / stds
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)

            simple_scaler = StandardScaler()
            simple_scaler.mean_ = means
            simple_scaler.scale_ = stds
            simple_scaler.var_ = stds ** 2
            simple_scaler.n_features_in_ = data.shape[1]

            return normalized_data, simple_scaler

    def _build_sequences(self, data_array: np.ndarray, target_array: np.ndarray, fwd_returns_array: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_samples = len(data_array) - self.sequence_length

        if num_samples <= 0:
            self.logger.warning(
                f"Not enough data to create sequences. Have {len(data_array)} rows, need {self.sequence_length}.")
            return np.array([]), np.array([]), np.array([])

        feature_dim = data_array.shape[1]
        X = np.zeros((num_samples, self.sequence_length, feature_dim), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.float32)
        fwd_r = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            X[i] = data_array[i:i + self.sequence_length]

            label_idx = i + self.sequence_length
            if label_idx < len(target_array):
                y[i] = target_array[label_idx]
                fwd_r[i] = fwd_returns_array[label_idx]
            else:
                y[i] = 0.0
                fwd_r[i] = 0.0

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)
        fwd_r = np.nan_to_num(fwd_r, nan=0.0)

        return X, y, fwd_r

    def _ensure_feature_dimensions(self, X_test: np.ndarray, test_feature_names: List[str]) -> np.ndarray:
        if self.scaler is None or self.feature_names is None:
            self.logger.warning("No scaler or feature names available for dimension alignment")
            return X_test

        try:
            n_train_features = self.scaler.n_features_in_
            test_to_train_idx = np.full(n_train_features, -1)

            for i, train_feature in enumerate(self.feature_names):
                if train_feature in test_feature_names:
                    test_idx = test_feature_names.index(train_feature)
                    test_to_train_idx[i] = test_idx

            aligned_data = np.zeros((X_test.shape[0], n_train_features), dtype=np.float32)

            for i in range(n_train_features):
                if test_to_train_idx[i] >= 0:
                    aligned_data[:, i] = X_test[:, test_to_train_idx[i]]
                else:
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else "unknown"

                    if 'rsi' in feature_name.lower():
                        aligned_data[:, i] = 50
                    elif 'macd' in feature_name.lower():
                        aligned_data[:, i] = 0
                    elif 'bb_percent' in feature_name.lower():
                        aligned_data[:, i] = 0.5
                    elif feature_name in self.fallback_indicators:
                        aligned_data[:, i] = self.fallback_indicators[feature_name]
                    else:
                        aligned_data[:, i] = 0

            return aligned_data

        except Exception as e:
            self.logger.error(f"Error aligning feature dimensions: {e}")
            if X_test.shape[1] > n_train_features:
                return X_test[:, :n_train_features]
            elif X_test.shape[1] < n_train_features:
                padded = np.zeros((X_test.shape[0], n_train_features), dtype=np.float32)
                padded[:, :X_test.shape[1]] = X_test
                return padded
            else:
                return X_test

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if col.startswith('actual_'):
                continue

            if col in ['open', 'high', 'low', 'close']:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            elif col == 'volume' or 'volume' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)
            elif 'rsi' in col.lower():
                df_clean[col] = df_clean[col].fillna(50)
            elif 'willr' in col.lower():
                df_clean[col] = df_clean[col].fillna(-50)
            elif 'macd' in col.lower() or 'regime' in col.lower():
                df_clean[col] = df_clean[col].fillna(0)
            elif 'mfi' in col.lower():
                df_clean[col] = df_clean[col].fillna(50)
            elif 'ratio' in col.lower():
                df_clean[col] = df_clean[col].fillna(0.5)
            else:
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=5)
                median_val = df_clean[col].median()
                if not pd.isna(median_val):
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    df_clean[col] = df_clean[col].fillna(0)

        try:
            cols_with_nans = [col for col in numeric_cols if
                              df_clean[col].isna().any() and not col.startswith('actual_')]

            if cols_with_nans and len(df_clean) > 200:
                correlation_matrix = df_clean[numeric_cols].corr().abs()

                for col in cols_with_nans:
                    correlated_features = correlation_matrix[col][
                        (correlation_matrix[col] > 0.7) &
                        (correlation_matrix[col] < 1.0)
                        ].index.tolist()

                    if correlated_features:
                        mask = df_clean[col].notna()
                        if mask.sum() > 100:
                            X = df_clean.loc[mask, correlated_features]
                            y = df_clean.loc[mask, col]

                            nan_mask = df_clean[col].isna()
                            X_pred = df_clean.loc[nan_mask, correlated_features]

                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(X, y)

                            predictions = model.predict(X_pred)
                            df_clean.loc[nan_mask, col] = predictions
        except Exception as e:
            self.logger.warning(f"Advanced imputation error: {e}")

        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].median() if not pd.isna(df_clean[col].median()) else 0)

        return df_clean

    def optimize_features(self, df_features):
        if self.use_optuna_features:
            # Force optimization regardless of whether optimized_features already exists
            self.logger.info("Running Optuna feature optimization for current iteration...")
            self.optimized_features = self.optuna_feature_selector.optimize_features(df_features)
            return self.optimized_features
        return None
