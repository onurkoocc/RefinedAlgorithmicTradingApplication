import logging
import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Tuple, Optional, Dict, List, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from pathlib import Path


class DataPreparer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataPreparer")

        self.sequence_length = config.get("model", "sequence_length", 60)
        self.horizon = config.get("model", "horizon", 12)
        self.normalize_method = config.get("model", "normalize_method", "feature_specific")
        self.train_ratio = config.get("model", "train_ratio", 0.75)

        self.results_dir = Path(config.results_dir)
        self.scaler_path = self.results_dir / "models" / "feature_scaler.pkl"
        # self.feature_list_path = self.results_dir / "models" / "feature_list.json" # Redundant if features stored with scaler

        self.price_column = "actual_close"  # Use actual_close for label generation

        self.scalers = {}  # For feature_specific
        self.fitted_feature_names = None  # Features scaler was fit on
        self.return_scale_factor = 1.0  # For scaling regression target

        self.essential_features = config.get("feature_engineering", "essential_features", [])
        self.max_features_model = config.get("model", "max_features", 60)

        self._load_scaler_and_features()

    def _load_scaler_and_features(self):
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    saved_state = pickle.load(f)
                    self.scalers = saved_state.get("scalers", {})
                    self.fitted_feature_names = saved_state.get("feature_names", None)
                    self.return_scale_factor = saved_state.get("return_scale_factor", 1.0)
                self.logger.info(f"Loaded scaler and feature state from {self.scaler_path}")
            except Exception as e:
                self.logger.warning(f"Error loading scaler/features: {e}. Will refit.")
                self.scalers = {}
                self.fitted_feature_names = None

    def _save_scaler_and_features(self):
        try:
            os.makedirs(self.scaler_path.parent, exist_ok=True)
            save_state = {
                "scalers": self.scalers,
                "feature_names": self.fitted_feature_names,
                "return_scale_factor": self.return_scale_factor
            }
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(save_state, f)
            self.logger.info(f"Saved scaler and feature state to {self.scaler_path}")
        except Exception as e:
            self.logger.error(f"Error saving scaler/features: {e}")

    def prepare_data(self, df_full_features: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:

        df = df_full_features.copy()

        all_available_features = [col for col in df.columns if not col.startswith(
            'actual_') and col != self.price_column and col != 'market_regime_type']

        current_features = [f for f in self.essential_features if f in all_available_features]

        remaining_slots = self.max_features_model - len(current_features)
        if remaining_slots > 0:
            other_available = [f for f in all_available_features if f not in current_features]
            # A more sophisticated selection (e.g. based on variance or pre-computed importance) could go here
            current_features.extend(other_available[:remaining_slots])

        df_model_features = df[current_features].copy()
        df_model_features = self._handle_outliers_and_nans(df_model_features)

        if len(df_model_features) < self.sequence_length + self.horizon + 10:
            self.logger.warning("Not enough data for sequence creation.")
            return (np.array([]),) * 6

        # df_labeled is aligned with df_full_features but potentially shorter due to horizon shift
        df_labeled, target_returns_scaled, fwd_returns_raw = self._create_regression_labels(df)

        # Align df_model_features with df_labeled (which is already aligned with where labels are valid)
        df_model_features = df_model_features.loc[df_labeled.index]

        if self.normalize_method != "none":
            if not self.scalers or self.fitted_feature_names is None or \
                    set(self.fitted_feature_names) != set(df_model_features.columns):
                self.logger.info("Fitting new scalers as features changed or no scaler found.")
                self.fitted_feature_names = df_model_features.columns.tolist()
                self.scalers = self._fit_scalers(
                    df_model_features)  # Fit on the (potentially shorter) aligned df_model_features
                self._save_scaler_and_features()

            df_normalized_features = self._apply_scalers(df_model_features)
        else:
            df_normalized_features = df_model_features

        X_full, y_full, fwd_returns_seq = self._build_sequences(
            df_normalized_features.values.astype(np.float32),
            target_returns_scaled,  # Already aligned with df_normalized_features
            fwd_returns_raw  # Already aligned
        )

        if len(X_full) == 0: return (np.array([]),) * 6

        train_size = int(self.train_ratio * len(X_full))
        X_train, X_val = X_full[:train_size], X_full[train_size:]
        y_train, y_val = y_full[:train_size], y_full[train_size:]
        fwd_returns_val_seq = fwd_returns_seq[train_size:]

        # df_val should correspond to the *last observation* of each sequence in X_val.
        # The indices for these last observations are relative to df_normalized_features (and thus df_labeled).
        # val_original_indices_in_df_labeled are the integer positions in df_labeled
        val_original_indices_in_df_labeled = [train_size + i + self.sequence_length - 1 for i in range(len(X_val))]

        # Filter out any indices that might be out of bounds for df_labeled (should not happen if logic is correct)
        val_original_indices_in_df_labeled = [idx for idx in val_original_indices_in_df_labeled if
                                              idx < len(df_labeled)]

        if val_original_indices_in_df_labeled:
            # Get the actual DatetimeIndex labels from df_labeled using these integer positions
            datetime_indices_for_df_val = df_labeled.index[val_original_indices_in_df_labeled]
            # Use .loc with these DatetimeIndex labels on the original df_full_features
            df_val = df_full_features.loc[datetime_indices_for_df_val].copy()
        else:
            df_val = pd.DataFrame()

        return X_train, y_train, X_val, y_val, df_val, fwd_returns_val_seq

    def prepare_test_data(self, df_full_features: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        df = df_full_features.copy()

        if self.fitted_feature_names is None:
            self.logger.error("Scaler/features not fitted. Cannot prepare test data. Train first.")
            return (np.array([]),) * 4

        df_model_features = pd.DataFrame(columns=self.fitted_feature_names, index=df.index)
        for col in self.fitted_feature_names:
            if col in df.columns:
                df_model_features[col] = df[col]
            else:
                self.logger.warning(f"Missing feature {col} in test data, imputing with 0.")
                df_model_features[col] = 0

        df_model_features = self._handle_outliers_and_nans(df_model_features)

        if len(df_model_features) < self.sequence_length:
            self.logger.warning("Not enough test data for sequence creation.")
            return (np.array([]),) * 4

        df_labeled, target_returns_scaled, fwd_returns_raw = self._create_regression_labels(df)
        # Align df_model_features with df_labeled before normalization and sequence building
        df_model_features = df_model_features.loc[df_labeled.index]

        if self.normalize_method != "none":
            df_normalized_features = self._apply_scalers(df_model_features)
        else:
            df_normalized_features = df_model_features

        X_test, y_test_dummy, fwd_returns_seq = self._build_sequences(
            df_normalized_features.values.astype(np.float32),
            target_returns_scaled,  # Dummy target for test, already aligned
            fwd_returns_raw  # Already aligned
        )

        # df_test_actuals should correspond to the last element of each sequence in X_test
        # test_original_indices_in_df_labeled are integer positions in df_labeled
        test_original_indices_in_df_labeled = [i + self.sequence_length - 1 for i in range(len(X_test))]
        test_original_indices_in_df_labeled = [idx for idx in test_original_indices_in_df_labeled if
                                               idx < len(df_labeled)]

        if test_original_indices_in_df_labeled:
            datetime_indices_for_df_test = df_labeled.index[test_original_indices_in_df_labeled]
            df_test_actuals = df_full_features.loc[datetime_indices_for_df_test].copy()
        else:
            df_test_actuals = pd.DataFrame()

        return X_test, y_test_dummy, df_test_actuals, fwd_returns_seq

    def _fit_scalers(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        scalers = {}
        for feature in df_features.columns:
            col_data = df_features[feature].values.reshape(-1, 1)
            if self.normalize_method == "robust":
                scaler = RobustScaler(quantile_range=(5.0, 95.0))
            elif self.normalize_method == "minmax":
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif self.normalize_method == "feature_specific":
                if 'rsi' in feature.lower() or 'percent_b' in feature.lower() or 'range_position' in feature.lower():
                    scaler = MinMaxScaler(feature_range=(0, 1))
                elif 'regime' in feature.lower() or 'sin' in feature.lower() or 'cos' in feature.lower() or 'cycle' in feature.lower():
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                elif 'volume' in feature.lower() or 'atr' in feature.lower() or 'bb_width' in feature.lower():
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))
                else:
                    scaler = StandardScaler()
            else:
                scaler = StandardScaler()
            try:
                scaler.fit(col_data)
                scalers[feature] = scaler
            except ValueError as e:
                self.logger.warning(
                    f"Could not fit scaler for feature '{feature}' (all NaNs or constant?): {e}. Using passthrough.")

                # Create a dummy scaler that does nothing (identity transform)
                class PassthroughScaler:
                    def fit(self, X, y=None): return self

                    def transform(self, X): return X

                    def fit_transform(self, X, y=None): return X

                scalers[feature] = PassthroughScaler()
        return scalers

    def _apply_scalers(self, df_features: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df_features.copy()
        for feature in df_features.columns:
            if feature in self.scalers:
                scaler = self.scalers[feature]
                try:
                    df_scaled[feature] = scaler.transform(df_features[feature].values.reshape(-1, 1)).flatten()
                except ValueError as e:
                    self.logger.warning(
                        f"Could not transform feature '{feature}' (all NaNs or constant after fit?): {e}. Leaving as is.")
                    df_scaled[feature] = df_features[feature]  # Leave as is if transform fails
            else:
                self.logger.warning(f"No scaler found for feature {feature} during transform. Leaving as is.")
                df_scaled[feature] = df_features[feature]  # Leave as is
        return df_scaled

    def _create_regression_labels(self, df_with_actuals: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if self.price_column not in df_with_actuals.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in DataFrame for label creation.")

        price = df_with_actuals[self.price_column].copy()
        future_prices = price.shift(-self.horizon)

        fwd_return_raw = (future_prices / price - 1).fillna(0)

        # Only recalculate scale_factor if it's the first time or seems uninitialized
        if not hasattr(self,
                       'return_scale_factor') or self.return_scale_factor == 1.0 or self.return_scale_factor == 0.03:
            abs_returns_for_scaling = fwd_return_raw.abs()
            # Ensure there are non-zero returns to calculate quantile, otherwise default
            if not abs_returns_for_scaling.empty and abs_returns_for_scaling.max() > 0:
                q95 = abs_returns_for_scaling.quantile(0.95)
                self.return_scale_factor = max(q95, 0.005) * 3  # Target typical large returns to be around +/- 0.33
            else:
                self.return_scale_factor = 0.03  # Default if no variance in returns

            if self.return_scale_factor == 0: self.return_scale_factor = 0.03  # Final fallback

        scaled_returns = np.clip(fwd_return_raw / (self.return_scale_factor + 1e-9), -1.0, 1.0)

        # df_labeled should be the part of the original df that has valid labels
        # The last 'horizon' rows will have NaN labels, so we drop them from the features df alignment.
        valid_label_length = len(price) - self.horizon
        df_labeled_aligned = df_with_actuals.iloc[:valid_label_length].copy()

        return df_labeled_aligned, scaled_returns.iloc[:valid_label_length].values, fwd_return_raw.iloc[
                                                                                    :valid_label_length].values

    def _handle_outliers_and_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        for col in df_clean.columns:
            if df_clean[col].dtype == 'object': continue

            df_clean[col] = df_clean[col].ffill().bfill()
            if df_clean[col].isna().any():
                if 'rsi' in col.lower():
                    df_clean[col] = df_clean[col].fillna(50)
                elif 'volume' in col.lower():
                    df_clean[col] = df_clean[col].fillna(0)
                else:
                    df_clean[col] = df_clean[col].fillna(0)

            if col not in ['open', 'high', 'low', 'close'] and not col.startswith('actual_'):
                # Check if column has variance before trying to compute quantiles
                if df_clean[col].nunique() > 1:
                    q01 = df_clean[col].quantile(0.01)
                    q99 = df_clean[col].quantile(0.99)
                    if q01 < q99:  # Ensure quantiles are valid
                        df_clean[col] = df_clean[col].clip(lower=q01, upper=q99)
                    elif q01 == q99 and q01 != 0:  # If constant non-zero, small range around it
                        df_clean[col] = df_clean[col].clip(lower=q01 * 0.99, upper=q01 * 1.01)
                # If no variance (nunique=1), clipping is not meaningful or might cause issues.
        return df_clean

    def _build_sequences(self, data_array: np.ndarray, target_array: np.ndarray, fwd_returns_array: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:

        num_total_samples = len(data_array)
        # Number of sequences we can create.
        # If data_array has N rows, and sequence_length is L,
        # the last sequence starts at index N-L. So there are (N-L)+1 sequences.
        num_sequences = num_total_samples - self.sequence_length + 1

        if num_sequences <= 0:
            self.logger.warning(
                f"Not enough data (rows: {num_total_samples}, seq_len: {self.sequence_length}) to create any sequences.")
            return np.array([]), np.array([]), np.array([])

        feature_dim = data_array.shape[1]
        X = np.zeros((num_sequences, self.sequence_length, feature_dim), dtype=np.float32)
        y = np.zeros(num_sequences, dtype=np.float32)
        fwd_r = np.zeros(num_sequences, dtype=np.float32)

        for i in range(num_sequences):
            X[i] = data_array[i: i + self.sequence_length]

            # The label for the sequence X[i] (which ends at data_array[i + sequence_length - 1])
            # is target_array[i + sequence_length - 1].
            label_idx = i + self.sequence_length - 1

            # target_array and fwd_returns_array are already aligned with data_array (features)
            # up to the point where labels can be computed (i.e., they are shorter by `horizon` elements
            # than the original feature dataframe before label calculation).
            # So, label_idx must be within the bounds of target_array.
            if label_idx < len(target_array):
                y[i] = target_array[label_idx]
            else:
                # This case should ideally not be hit if inputs are correctly aligned and lengths checked.
                # It might indicate an off-by-one or misalignment earlier.
                self.logger.warning(
                    f"Label index {label_idx} out of bounds for target_array (len {len(target_array)}) at sequence {i}.")
                y[i] = 0.0

            if label_idx < len(fwd_returns_array):
                fwd_r[i] = fwd_returns_array[label_idx]
            else:
                self.logger.warning(
                    f"Label index {label_idx} out of bounds for fwd_returns_array (len {len(fwd_returns_array)}) at sequence {i}.")
                fwd_r[i] = 0.0

        return X, y, fwd_r