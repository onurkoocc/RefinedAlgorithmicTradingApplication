import logging
import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Tuple, Optional, Dict, List, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


class DataPreparer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataPreparer")

        # Load configuration parameters
        self.sequence_length = config.get("model", "sequence_length", 72)
        self.horizon = config.get("model", "horizon", 16)
        self.normalize_method = config.get("model", "normalize_method", "feature_specific")
        self.train_ratio = config.get("model", "train_ratio", 0.7)

        # Define paths for serializing scalers and feature lists
        self.results_dir = config.results_dir
        self.scaler_path = os.path.join(self.results_dir, "models", "feature_scaler.pkl")
        self.feature_list_path = os.path.join(self.results_dir, "models", "feature_list.json")
        self.importance_path = os.path.join(self.results_dir, "models", "feature_importance.json")

        self.price_column = "close"

        # Initialize scalers and feature tracking
        self.scaler = None
        self.scaler_dict = {}
        self.feature_names = None
        self.feature_importance = {}
        self.test_sequence_length = self.sequence_length
        self.normalization_stats = {}

        # Track the features used in training for consistent test normalization
        self.train_features = None
        self.max_features = config.get("model", "max_features", 50)

        # Define essential features - these must be consistent across training and inference
        self.essential_features = [
            "open", "high", "low", "close", "volume",
            "ema_cross", "hull_ma_16", "macd_histogram",
            "rsi_14", "stoch_k", "stoch_d", "cci_20", "willr_14",
            "atr_14", "bb_width", "bb_percent_b", "keltner_width",
            "obv", "taker_buy_ratio", "cmf", "mfi", "volume_oscillator",
            "market_regime", "volatility_regime"
        ]

        # Configure adaptive features options
        self.use_adaptive_features = config.get("feature_engineering", "use_adaptive_features", True)
        self.feature_selection_method = config.get("feature_engineering", "feature_selection_method", "importance")
        self.dynamic_feature_count = config.get("feature_engineering", "dynamic_feature_count", 35)

        # Consistent fallback values for missing features with more market-aware values
        self.fallback_indicators = {
            "rsi_14": 50,
            "stoch_k": 50,
            "stoch_d": 50,
            "cci_20": 0,
            "willr_14": -50,
            "macd_histogram": 0,
            "bb_percent_b": 0.5,
            "market_regime": 0,
            "volatility_regime": 0.5,
            "taker_buy_ratio": 0.5,
            "mfi": 50
        }

        # Try to load existing scaler and feature list
        self._load_scaler_and_features()
        self._load_feature_importance()

    def _load_scaler_and_features(self):
        """Load serialized scaler and feature list if they exist"""
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
        """Load feature importance scores if they exist"""
        try:
            if os.path.exists(self.importance_path):
                with open(self.importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                self.logger.info(f"Loaded feature importance for {len(self.feature_importance)} features")
        except Exception as e:
            self.logger.warning(f"Error loading feature importance: {e}")

    def _save_scaler_and_features(self):
        """Serialize scaler and feature list for consistent usage"""
        try:
            # Create directory if it doesn't exist
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
        """Save feature importance scores"""
        try:
            if self.feature_importance:
                with open(self.importance_path, 'w') as f:
                    json.dump(self.feature_importance, f)
                self.logger.info(f"Saved feature importance to {self.importance_path}")
        except Exception as e:
            self.logger.error(f"Error saving feature importance: {e}")

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, pd.DataFrame, np.ndarray]:
        """Prepare data for model training with consistent normalization and feature set."""
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # Identify actual price columns for later use
        actual_cols = [col for col in df.columns if col.startswith('actual_')]

        # Get consistent feature set - critical for reproducibility
        self.train_features = self._get_available_features(df)

        # Log feature usage info
        self.logger.info(f"Using {len(self.train_features)} features for model training")
        missing_essential = [f for f in self.essential_features if f not in self.train_features]
        if missing_essential:
            self.logger.warning(f"Missing {len(missing_essential)} essential features: {missing_essential}")

        # Copy only needed columns to avoid unnecessary data
        df_features = df[self.train_features].copy()

        # Apply consistent cleaning across all preparation steps
        df_features = self._clean_dataframe(df_features)

        try:
            # Ensure we have enough data for sequence creation
            min_required_rows = self.sequence_length + self.horizon + 10  # Add buffer
            if len(df_features) < min_required_rows:
                self.logger.warning(
                    f"Not enough data for sequence creation: {len(df_features)} rows, need {min_required_rows}")
                return (np.array([]),) * 6

            # Create target labels for regression model
            df_labeled, target_returns, fwd_returns = self._create_regression_labels(df_features)

            # Clean and handle outliers consistently
            df_labeled = self._handle_outliers(df_labeled)

            # Calculate feature importance for dynamic feature selection
            if self.use_adaptive_features:
                self._calculate_feature_importance(df_labeled, target_returns)

            # Store the features used in training for consistent test normalization
            self.feature_names = df_labeled.columns.tolist()

            # Apply normalization to features BEFORE sequence creation
            X_normalized = df_labeled.values.astype(np.float32)

            if self.normalize_method != "none":
                X_normalized, self.scaler = self._normalize_features(X_normalized, df_labeled.columns)
                # After successful normalization, save the scaler and feature list
                self._save_scaler_and_features()
                self._save_feature_importance()

            # Build sequences with normalized data
            X_full, y_full, fwd_returns_full = self._build_sequences(
                X_normalized, target_returns, fwd_returns
            )

            # Check if sequences were created successfully
            if len(X_full) == 0:
                self.logger.warning("No sequences created during processing")
                return (np.array([]),) * 6

            # Split into training and validation sets
            train_size = int(self.train_ratio * len(X_full))
            X_train, X_val = X_full[:train_size], X_full[train_size:]
            y_train, y_val = y_full[:train_size], y_full[train_size:]
            fwd_returns_val = fwd_returns_full[train_size:]

            # Create validation dataframe for signal analysis
            entry_indices = list(range(self.sequence_length - 1, len(df_labeled)))
            val_entry_indices = entry_indices[train_size:]

            if val_entry_indices:
                # Extract validation dataframe with actual prices
                df_val = df_labeled.iloc[val_entry_indices].copy()

                # Add actual price columns for accurate signal generation
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

    def _calculate_feature_importance(self, df: pd.DataFrame, target_returns: np.ndarray) -> None:
        """Calculate feature importance using random forest or mutual information"""
        try:
            # Skip if we don't have enough data
            if len(df) < 200 or len(target_returns) < 200:
                return

            # Align data lengths
            X = df.iloc[:len(target_returns)].values
            y = target_returns

            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

            if self.feature_selection_method == 'importance':
                # Use Random Forest to get feature importance
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                model.fit(X, y)
                importance = model.feature_importances_
            else:
                # Use mutual information as an alternative
                importance = mutual_info_regression(X, y, random_state=42)

            # Normalize importance scores
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)

            # Store importance in dictionary
            feature_importance = {}
            for i, feature in enumerate(df.columns):
                feature_importance[feature] = float(importance[i])

            self.feature_importance = feature_importance

        except Exception as e:
            self.logger.warning(f"Error calculating feature importance: {e}")

    def _get_available_features(self, df: pd.DataFrame) -> List[str]:
        """Get available features with dynamic feature selection"""
        # First collect all available features
        available_features = []

        # First ensure we have all essential features we can find
        for feature in self.essential_features:
            if feature in df.columns:
                available_features.append(feature)
            elif f'm30_{feature}' in df.columns:
                available_features.append(f'm30_{feature}')

        # Then add any other columns that might be useful
        for col in df.columns:
            if col not in available_features and not col.startswith('actual_'):
                available_features.append(col)

        if not self.use_adaptive_features or not self.feature_importance:
            # If not using adaptive features or no importance scores yet,
            # return all available features (up to max limit)
            return available_features[:self.max_features]

        # Get available features with their importance scores
        scored_features = []
        for feature in available_features:
            importance = self.feature_importance.get(feature, 0.0)
            scored_features.append((feature, importance))

        # Sort by importance and take top features
        scored_features.sort(key=lambda x: x[1], reverse=True)

        # Always include essential price columns
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        selected_features = [feat for feat in essential_cols if feat in available_features]

        # Add remaining features by importance
        for feature, _ in scored_features:
            if feature not in selected_features and len(selected_features) < self.dynamic_feature_count:
                selected_features.append(feature)

        self.logger.info(f"Selected {len(selected_features)} features using {self.feature_selection_method} method")
        return selected_features

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        """Prepare test data with the same normalization and feature set as training data."""
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # Identify actual price columns
        actual_cols = [col for col in df.columns if col.startswith('actual_')]

        # Ensure we use the same features as during training
        if self.feature_names:
            # Use exactly the features from training - this is critical
            required_features = self.feature_names
        else:
            # Fall back to essential features if no training features are available
            self.logger.warning("No training features found - using essential features only")
            required_features = self.essential_features

        # Log feature alignment details
        available_features_in_test = set(df.columns)
        missing_features = [f for f in required_features if f not in available_features_in_test]
        if missing_features:
            self.logger.warning(f"Test data missing {len(missing_features)} required features: {missing_features}")

        # Extract available features and create placeholders for missing ones
        df_features = pd.DataFrame(index=df.index)
        for feature in required_features:
            if feature in df.columns:
                df_features[feature] = df[feature]
            elif f'm30_{feature}' in df.columns:
                # Use prefixed version as fallback
                df_features[feature] = df[f'm30_{feature}']
            else:
                # Use data-driven imputation rather than static fallbacks
                df_features[feature] = self._impute_missing_feature(df, feature)

        # Apply consistent cleaning
        df_features = self._clean_dataframe(df_features)

        try:
            # Check data length
            if len(df_features) < (self.sequence_length + self.horizon):
                self.logger.warning(
                    f"Not enough data for test sequence creation: {len(df_features)} rows, need {self.sequence_length + self.horizon}")
                return np.array([]), np.array([]), df_features, np.array([])

            # Create regression labels
            df_labeled, target_returns, fwd_returns = self._create_regression_labels(df_features)

            # Add actual price columns
            df_actual = df_labeled.copy()
            for col in actual_cols:
                if col in df.columns:
                    df_actual[col] = df[col].values[:len(df_labeled)]

            # Handle outliers consistently
            df_labeled = self._handle_outliers(df_labeled)

            # Apply the same normalization as during training
            X_features = df_labeled.values.astype(np.float32)

            if self.normalize_method != "none":
                if self.normalize_method == 'feature_specific' and self.scaler_dict:
                    # Use feature-specific scalers
                    X_features = self._apply_feature_specific_normalization(X_features, df_labeled.columns)
                elif self.scaler:
                    # Use single scaler
                    if X_features.shape[1] != self.scaler.n_features_in_:
                        self.logger.warning(
                            f"Feature dimension mismatch: test has {X_features.shape[1]}, scaler expects {self.scaler.n_features_in_}")
                        X_features = self._ensure_feature_dimensions(X_features, df_labeled.columns.tolist())

                    try:
                        X_features = self.scaler.transform(X_features)
                    except Exception as e:
                        self.logger.error(f"Error transforming test data: {e}")
                        # Fallback to standardized values
                        X_features = (X_features - np.mean(X_features, axis=0)) / np.maximum(np.std(X_features, axis=0),
                                                                                             1e-5)

            # Build sequences
            X_test, y_test, fwd_returns_test = self._build_sequences(
                X_features, target_returns, fwd_returns
            )

            return X_test, y_test, df_actual, fwd_returns_test

        except Exception as e:
            self.logger.error(f"Error in prepare_test_data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([]), np.array([]), df_features, np.array([])

    def _impute_missing_feature(self, df: pd.DataFrame, feature: str) -> pd.Series:
        """Impute missing feature with data-driven methods"""
        # If we have a known fallback for this type of indicator, use it
        if feature in self.fallback_indicators:
            return pd.Series(self.fallback_indicators[feature], index=df.index)

        # For price/volume features, use close price as reasonable proxy
        if feature in ['open', 'high', 'low'] and 'close' in df.columns:
            return df['close'].copy()

        # Try to use related features for imputation
        if 'macd' in feature.lower() and 'close' in df.columns:
            # Simple proxy: percent change of close price
            return df['close'].pct_change(5).fillna(0) * 100

        if 'volume' in feature.lower() and 'volume' in df.columns:
            return df['volume'].copy()

        if 'regime' in feature.lower():
            # Create a proxy for market regime based on price action
            if 'close' in df.columns and len(df) > 20:
                sma20 = df['close'].rolling(20).mean()
                return ((df['close'] - sma20) / sma20).clip(-1, 1).fillna(0)

        # Default fallback: return zeros
        return pd.Series(0.0, index=df.index)

    def _apply_feature_specific_normalization(self, data: np.ndarray, feature_names: pd.Index) -> np.ndarray:
        """Apply feature-specific normalization to test data"""
        try:
            result = np.zeros_like(data)

            for i, feature in enumerate(feature_names):
                if feature in self.scaler_dict:
                    # Get the specific scaler for this feature
                    scaler = self.scaler_dict[feature]
                    # Apply transformation
                    result[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
                else:
                    # If no scaler exists, use standard normalization
                    mean = np.mean(data[:, i])
                    std = np.std(data[:, i])
                    if std > 1e-8:
                        result[:, i] = (data[:, i] - mean) / std
                    else:
                        result[:, i] = data[:, i] - mean

            return result

        except Exception as e:
            self.logger.error(f"Error in feature-specific normalization: {e}")
            # Fallback to simple standardization
            return (data - np.mean(data, axis=0)) / np.maximum(np.std(data, axis=0), 1e-5)

    def _create_regression_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        price = df[self.price_column].copy()

        # Forward-looking future price
        future_prices = price.shift(-self.horizon)

        # Calculate forward returns
        fwd_return = (future_prices / price - 1)
        fwd_return = fwd_return.fillna(0)

        # Use absolute returns for scaling (more robust than static values)
        abs_returns = fwd_return.abs()
        scale_factor = max(abs_returns.quantile(0.9), 0.003) * 12

        # Clip extreme values for stability
        scaled_returns = np.clip(fwd_return / scale_factor, -1.1, 1.1)

        self.return_scale_factor = scale_factor

        valid_length = len(scaled_returns)
        df_valid = df.iloc[:valid_length].copy()

        return df_valid, scaled_returns.values, fwd_return.values

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        for col in df_clean.select_dtypes(include=np.number).columns:
            # Skip actual price columns
            if col.startswith('actual_'):
                continue

            # Apply different handling based on feature type
            if col in ['open', 'high', 'low', 'close', 'volume']:
                # For price and volume, use percentile-based clipping
                q1 = df_clean[col].quantile(0.001)
                q3 = df_clean[col].quantile(0.999)
                df_clean[col] = df_clean[col].clip(q1, q3)

            # For oscillator indicators with known bounds
            elif 'rsi' in col.lower():
                df_clean[col] = df_clean[col].clip(0, 100)
            elif 'stoch' in col.lower():
                df_clean[col] = df_clean[col].clip(0, 100)
            elif 'willr' in col.lower():
                df_clean[col] = df_clean[col].clip(-100, 0)
            elif 'bb_percent_b' in col.lower():
                df_clean[col] = df_clean[col].clip(0, 1)
            elif 'regime' in col.lower():
                df_clean[col] = df_clean[col].clip(-1, 1)

            # For other indicators, use robust outlier detection
            else:
                q1 = df_clean[col].quantile(0.01)
                q3 = df_clean[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - (3 * iqr)
                upper_bound = q3 + (3 * iqr)
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        return df_clean

    def _normalize_features(self, data: np.ndarray, feature_names: pd.Index) -> Tuple[np.ndarray, object]:
        """Apply normalization to feature data with feature-specific methods"""
        if self.normalize_method == "none":
            return data, None

        try:
            if self.normalize_method == "feature_specific":
                # Apply feature-specific normalization
                normalized_data = np.zeros_like(data)
                scaler_dict = {}

                for i, feature in enumerate(feature_names):
                    # Select appropriate scaler for each feature type
                    if 'rsi' in feature.lower() or 'stoch' in feature.lower() or 'percent_b' in feature.lower():
                        # Bounded indicators [0,100] or [0,1]
                        scaler = MinMaxScaler(feature_range=(0, 1))
                    elif 'regime' in feature.lower():
                        # Already normalized features
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                    elif 'momentum' in feature.lower() or 'macd' in feature.lower():
                        # Potentially unbounded indicators
                        scaler = RobustScaler()
                    else:
                        # General indicators
                        scaler = StandardScaler()

                    # Reshape to 2D for sklearn
                    col_data = data[:, i].reshape(-1, 1)

                    # Fit and transform
                    try:
                        normalized_data[:, i] = scaler.fit_transform(col_data).flatten()
                        scaler_dict[feature] = scaler
                    except Exception as e:
                        self.logger.warning(f"Error normalizing {feature}: {e}")
                        # Fallback to simple standardization
                        mean = np.mean(col_data)
                        std = max(np.std(col_data), 1e-8)
                        normalized_data[:, i] = ((col_data - mean) / std).flatten()

                self.scaler_dict = scaler_dict
                return normalized_data, scaler_dict

            elif self.normalize_method == "robust":
                # RobustScaler is better for financial data with outliers
                scaler = RobustScaler(quantile_range=(10, 90))
            else:
                # StandardScaler for more normally distributed data
                scaler = StandardScaler()

            # Fit and transform data with single scaler
            normalized_data = scaler.fit_transform(data)

            # Store normalization statistics if available
            if hasattr(scaler, 'center_'):
                self.normalization_stats['center'] = scaler.center_
            if hasattr(scaler, 'scale_'):
                self.normalization_stats['scale'] = scaler.scale_

            # Clean up any remaining invalid values
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)

            return normalized_data, scaler

        except Exception as e:
            self.logger.error(f"Normalization error: {e}")
            # Fall back to standardized values rather than original data
            means = np.nanmean(data, axis=0)
            stds = np.nanstd(data, axis=0)
            # Replace zero stds with 1 to avoid division by zero
            stds[stds < 1e-8] = 1.0

            normalized_data = (data - means) / stds
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Create a simple scaler for consistency
            simple_scaler = StandardScaler()
            simple_scaler.mean_ = means
            simple_scaler.scale_ = stds
            simple_scaler.var_ = stds ** 2
            simple_scaler.n_features_in_ = data.shape[1]

            return normalized_data, simple_scaler

    def _build_sequences(self, data_array: np.ndarray, target_array: np.ndarray,
                         fwd_returns_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build sequences for time series model input with proper error checking."""
        # Calculate number of possible sequences
        num_samples = len(data_array) - self.sequence_length

        if num_samples <= 0:
            self.logger.warning(
                f"Not enough data to create sequences. Have {len(data_array)} rows, need {self.sequence_length}.")
            return np.array([]), np.array([]), np.array([])

        # Pre-allocate arrays
        feature_dim = data_array.shape[1]
        X = np.zeros((num_samples, self.sequence_length, feature_dim), dtype=np.float32)
        y = np.zeros(num_samples, dtype=np.float32)
        fwd_r = np.zeros(num_samples, dtype=np.float32)

        # Fill sequences
        for i in range(num_samples):
            # Extract sequence window up to t-1
            X[i] = data_array[i:i + self.sequence_length]

            # Get label from the next point after the sequence (t)
            label_idx = i + self.sequence_length
            if label_idx < len(target_array):
                y[i] = target_array[label_idx]
                fwd_r[i] = fwd_returns_array[label_idx]
            else:
                # Handle edge case
                y[i] = 0.0
                fwd_r[i] = 0.0

        # Clean any invalid values as final safety check
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)
        fwd_r = np.nan_to_num(fwd_r, nan=0.0)

        return X, y, fwd_r

    def _ensure_feature_dimensions(self, X_test: np.ndarray, test_feature_names: List[str]) -> np.ndarray:
        """Ensure test features exactly match training features in number and order."""
        if self.scaler is None or self.feature_names is None:
            self.logger.warning("No scaler or feature names available for dimension alignment")
            return X_test

        try:
            # Create ordered mapping from test features to training features
            n_train_features = self.scaler.n_features_in_
            test_to_train_idx = np.full(n_train_features, -1)  # -1 indicates no match

            # Map test features to training features where possible
            for i, train_feature in enumerate(self.feature_names):
                if train_feature in test_feature_names:
                    test_idx = test_feature_names.index(train_feature)
                    test_to_train_idx[i] = test_idx

            # Create aligned array with correct dimensions
            aligned_data = np.zeros((X_test.shape[0], n_train_features), dtype=np.float32)

            # Fill with data where features match
            for i in range(n_train_features):
                if test_to_train_idx[i] >= 0:
                    # Copy data from test array
                    aligned_data[:, i] = X_test[:, test_to_train_idx[i]]
                else:
                    # Use data-driven imputation for missing features
                    feature_name = self.feature_names[i] if i < len(self.feature_names) else "unknown"

                    # Use more intelligent imputation
                    if 'rsi' in feature_name.lower():
                        aligned_data[:, i] = 50  # Neutral RSI
                    elif 'macd' in feature_name.lower():
                        aligned_data[:, i] = 0  # Neutral MACD
                    elif 'stoch' in feature_name.lower():
                        aligned_data[:, i] = 50  # Middle stochastic
                    elif 'bb_percent' in feature_name.lower():
                        aligned_data[:, i] = 0.5  # Middle bollinger
                    elif feature_name in self.fallback_indicators:
                        aligned_data[:, i] = self.fallback_indicators[feature_name]
                    else:
                        # Default to zero for unknown features
                        aligned_data[:, i] = 0

            return aligned_data

        except Exception as e:
            self.logger.error(f"Error aligning feature dimensions: {e}")
            # In case of error, try to pad or truncate data to match expected dimensions
            if X_test.shape[1] > n_train_features:
                return X_test[:, :n_train_features]
            elif X_test.shape[1] < n_train_features:
                padded = np.zeros((X_test.shape[0], n_train_features), dtype=np.float32)
                padded[:, :X_test.shape[1]] = X_test
                return padded
            else:
                return X_test

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe with advanced imputation for NaN and inf values."""
        if df.empty:
            return df

        # Make a copy to avoid modifying original
        df_clean = df.copy()

        # Replace inf values with NaN for consistent handling
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Get numeric columns
        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        # First handle obvious NaN values with appropriate methods for each type
        for col in numeric_cols:
            # Skip actual price columns
            if col.startswith('actual_'):
                continue

            # Apply appropriate NaN handling by column type
            if col in ['open', 'high', 'low', 'close']:
                # For price columns, use forward/backward fill
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')

            elif col == 'volume' or 'volume' in col.lower():
                # For volume columns, replace NaN with 0
                df_clean[col] = df_clean[col].fillna(0)

            elif 'rsi' in col.lower():
                # For RSI, use neutral value of 50
                df_clean[col] = df_clean[col].fillna(50)

            elif 'stoch' in col.lower():
                # For stochastic oscillator, use neutral value of 50
                df_clean[col] = df_clean[col].fillna(50)

            elif 'willr' in col.lower():
                # For Williams %R, use middle value of -50
                df_clean[col] = df_clean[col].fillna(-50)

            elif 'macd' in col.lower() or 'regime' in col.lower():
                # For MACD and regime indicators, use 0
                df_clean[col] = df_clean[col].fillna(0)

            elif 'mfi' in col.lower():
                # For Money Flow Index, use neutral value of 50
                df_clean[col] = df_clean[col].fillna(50)

            elif 'ratio' in col.lower():
                # For ratio indicators, use neutral value of 0.5
                df_clean[col] = df_clean[col].fillna(0.5)

            else:
                # For other indicators:
                # 1. First try forward fill for small gaps (up to 5 points)
                df_clean[col] = df_clean[col].fillna(method='ffill', limit=5)

                # 2. Then try to use the median value if available
                median_val = df_clean[col].median()
                if not pd.isna(median_val):
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    # 3. Finally default to zero
                    df_clean[col] = df_clean[col].fillna(0)

        # For highly correlated features, use correlations to improve imputation
        try:
            # Find columns that still have NaNs
            cols_with_nans = [col for col in numeric_cols
                              if df_clean[col].isna().any() and not col.startswith('actual_')]

            if cols_with_nans and len(df_clean) > 200:
                # Create a correlation matrix for all numeric features
                correlation_matrix = df_clean[numeric_cols].corr().abs()

                for col in cols_with_nans:
                    # Find highly correlated features (>0.7)
                    correlated_features = correlation_matrix[col][
                        (correlation_matrix[col] > 0.7) &
                        (correlation_matrix[col] < 1.0)
                        ].index.tolist()

                    if correlated_features:
                        # Create a simple linear model using the correlated features
                        mask = df_clean[col].notna()
                        if mask.sum() > 100:  # Need enough data points
                            X = df_clean.loc[mask, correlated_features]
                            y = df_clean.loc[mask, col]

                            # Find rows with NaNs in this column
                            nan_mask = df_clean[col].isna()
                            X_pred = df_clean.loc[nan_mask, correlated_features]

                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(X, y)

                            # Predict and fill NaNs
                            predictions = model.predict(X_pred)
                            df_clean.loc[nan_mask, col] = predictions
        except Exception as e:
            self.logger.warning(f"Advanced imputation error: {e}")

        # Final NaN check - replace any remaining NaNs
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].median() if not pd.isna(df_clean[col].median()) else 0)

        return df_clean