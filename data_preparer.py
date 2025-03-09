import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Any, Union
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


class DataPreparer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataPreparer")

        self.sequence_length = config.get("model", "sequence_length", 72)
        self.horizon = config.get("model", "horizon", 16)
        self.normalize_method = config.get("model", "normalize_method", "zscore")
        self.train_ratio = config.get("model", "train_ratio", 0.7)

        self.price_column = "close"

        self.scaler = None
        self.feature_names = None
        self.test_sequence_length = self.sequence_length

        self.essential_features = [
            "ema_20", "obv", "force_index", "bb_width", "bb_upper", "bb_lower", "bb_mid",
            "stoch_k", "stoch_d", "atr_14", "rsi_14", "macd", "macd_signal", "macd_histogram"
        ]

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, pd.DataFrame, np.ndarray]:
        df = df.copy()

        df.columns = [col.lower() for col in df.columns]

        actual_cols = [col for col in df.columns if col.startswith('actual_')]

        required_features = [
            'm30_ema_20', 'm30_bb_lower', 'm30_bb_width',
            'm30_bb_upper', 'm30_obv', 'm30_bb_mid',
            'open', 'high', 'low', 'close', 'volume', 'atr_14',
            'm30_rsi_14', 'm30_macd', 'm30_macd_signal', 'm30_macd_histogram',
            'm30_cmf', 'm30_mfi', 'm30_vwap'
        ]
        available_features = []
        for col in df.columns:
            if col in required_features and col not in available_features:
                available_features.append(col)

        if 'atr_14' not in available_features and 'm30_atr_14' in df.columns:
            available_features.append('m30_atr_14')

        indicator_pairs = [
            ('ema_20', 'm30_ema_20'),
            ('rsi_14', 'm30_rsi_14'),
            ('macd', 'm30_macd'),
            ('macd_signal', 'm30_macd_signal'),
            ('macd_histogram', 'm30_macd_histogram')
        ]

        for base, prefixed in indicator_pairs:
            if base not in available_features and prefixed in df.columns and prefixed not in available_features:
                available_features.append(prefixed)

        self.logger.info(f"Using features for model training: {available_features}")

        df_features = df[available_features].copy()

        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        critical_columns = ['open', 'high', 'low', 'close', 'volume']

        df_features.dropna(subset=critical_columns, inplace=True)

        try:
            if len(df_features) < (self.sequence_length + self.horizon):
                self.logger.warning("Not enough data for sequence creation")
                return (np.array([]),) * 6

            df_labeled, labels, fwd_returns = self._create_labels(df_features)

            df_labeled = self._handle_outliers(df_labeled)

            data_array = df_labeled.values.astype(np.float32)

            X_full, y_full, fwd_returns_full = self._build_sequences(
                data_array, labels, fwd_returns
            )

            if len(X_full) == 0:
                self.logger.warning("No sequences created during processing")
                return (np.array([]),) * 6

            train_size = int(self.train_ratio * len(X_full))
            X_train, X_val = X_full[:train_size], X_full[train_size:]
            y_train, y_val = y_full[:train_size], y_full[train_size:]
            fwd_returns_val = fwd_returns_full[train_size:]

            entry_indices = list(range(self.sequence_length - 1, len(df_labeled)))
            val_entry_indices = entry_indices[train_size:]

            if val_entry_indices:
                df_val = df_labeled.iloc[val_entry_indices].copy()

                for col in actual_cols:
                    if col in df.columns:
                        df_val.loc[:, col] = df[col].iloc[val_entry_indices].values
            else:
                df_val = pd.DataFrame()

            if self.normalize_method and len(X_train) > 0:
                X_train, X_val = self._normalize_data(X_train, X_val, df_labeled.columns.tolist())

            return X_train, y_train, X_val, y_val, df_val, fwd_returns_val

        except Exception as e:
            self.logger.error(f"Error in prepare_data: {e}")
            return (np.array([]),) * 6

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
        df = df.copy()

        df.columns = [col.lower() for col in df.columns]

        actual_cols = [col for col in df.columns if col.startswith('actual_')]

        required_features = [
            'm30_ema_20', 'm30_bb_lower', 'm30_bb_width',
            'm30_bb_upper', 'm30_obv', 'm30_bb_mid',
            'open', 'high', 'low', 'close', 'volume', 'atr_14',
            'm30_rsi_14', 'm30_macd', 'm30_macd_signal', 'm30_macd_histogram',
            'm30_cmf', 'm30_mfi', 'm30_vwap'
        ]

        available_features = []
        for col in df.columns:
            if col in required_features and col not in available_features:
                available_features.append(col)

        if 'atr_14' not in available_features and 'm30_atr_14' in df.columns:
            df['atr_14'] = df['m30_atr_14']
            available_features.append('atr_14')

        indicator_pairs = [
            ('ema_20', 'm30_ema_20'),
            ('rsi_14', 'm30_rsi_14'),
            ('macd', 'm30_macd'),
            ('macd_signal', 'm30_macd_signal'),
            ('macd_histogram', 'm30_macd_histogram')
        ]

        for base, prefixed in indicator_pairs:
            if base not in available_features and prefixed in df.columns and prefixed not in available_features:
                df[base] = df[prefixed]
                available_features.append(base)

        self.logger.info(f"Using features for model testing: {available_features}")

        df_features = df[available_features].copy()

        try:
            if len(df_features) < (self.sequence_length + self.horizon):
                self.logger.warning("Not enough data for test sequence creation")
                return np.array([]), np.array([]), df_features, np.array([])

            df_labeled, labels, fwd_returns = self._create_labels(df_features)

            df_labeled = df_labeled.copy()

            for col in actual_cols:
                if col in df.columns:
                    df_labeled.loc[:, col] = df[col].values[:len(df_labeled)]

            X_test, y_test, fwd_returns_test = self._build_sequences(
                df_labeled[available_features].values.astype(np.float32),
                labels,
                fwd_returns
            )

            if self.scaler and len(X_test) > 0:
                X_test = self._normalize_test_data(X_test)

            return X_test, y_test, df_labeled, fwd_returns_test

        except Exception as e:
            self.logger.error(f"Error in prepare_test_data: {e}")
            return np.array([]), np.array([]), df_features, np.array([])

    def _create_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        price = df[self.price_column]

        future_prices = price.shift(-self.horizon)

        fwd_return = (future_prices / price - 1).fillna(0)

        labels = np.zeros(len(df), dtype=int)

        percentiles = [15, 40, 60, 85]
        boundaries = [np.percentile(fwd_return, p) for p in percentiles]

        labels[fwd_return < boundaries[0]] = 0
        labels[(fwd_return >= boundaries[0]) & (fwd_return < boundaries[1])] = 1
        labels[(fwd_return >= boundaries[1]) & (fwd_return < boundaries[2])] = 2
        labels[(fwd_return >= boundaries[2]) & (fwd_return < boundaries[3])] = 3
        labels[fwd_return >= boundaries[3]] = 4

        return df.iloc[:len(labels)], labels, fwd_return.values[:len(labels)]

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        for col in df_clean.select_dtypes(include=np.number).columns:
            if col.startswith('actual_'):
                continue

            q_low = df_clean[col].quantile(0.05)
            q_high = df_clean[col].quantile(0.95)

            df_clean[col] = df_clean[col].clip(q_low, q_high)

        return df_clean

    def _build_sequences(self, data_array: np.ndarray, labels_array: np.ndarray,
                         fwd_returns_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_samples = len(data_array) - self.sequence_length + 1
        if num_samples <= 0:
            return np.array([]), np.array([]), np.array([])

        X = np.zeros((num_samples, self.sequence_length, data_array.shape[1]), dtype=np.float32)
        y = np.zeros((num_samples, 5), dtype=np.float32)
        fwd_r = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            X[i] = data_array[i:i + self.sequence_length]

            label_idx = i + self.sequence_length - 1

            label = labels_array[label_idx]
            if 0 <= label < 5:
                one_hot = np.zeros(5)
                one_hot[label] = 1.0
                y[i] = one_hot
            else:
                neutral = np.zeros(5)
                neutral[2] = 1.0
                y[i] = neutral

            fwd_r[i] = fwd_returns_array[label_idx]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0)
        fwd_r = np.nan_to_num(fwd_r, nan=0.0)

        return X, y, fwd_r

    def _normalize_data(self, X_train: np.ndarray, X_val: np.ndarray, feature_names: List[str]) -> Tuple[
        np.ndarray, np.ndarray]:
        self.scaler = StandardScaler()
        self.feature_names = feature_names

        X_train_flat = X_train.reshape(-1, X_train.shape[2])

        self.scaler.fit(X_train_flat)
        X_train_scaled = self.scaler.transform(X_train_flat)

        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

        X_train = X_train_scaled.reshape(X_train.shape)

        noise_scale = 0.003
        noise = np.random.normal(0, noise_scale, X_train.shape)
        X_train += noise

        if len(X_val) > 0:
            X_val_flat = X_val.reshape(-1, X_val.shape[2])
            X_val_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
            X_val = X_val_scaled.reshape(X_val.shape)

        return X_train, X_val

    def _normalize_test_data(self, X_test: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            self.logger.warning("No scaler available for normalization")
            return X_test

        try:
            original_shape = X_test.shape

            X_test_flat = X_test.reshape(-1, original_shape[2])

            if X_test_flat.shape[1] != self.scaler.n_features_in_:
                self.logger.warning(
                    f"Dimension mismatch: Test has {X_test_flat.shape[1]} features, "
                    f"scaler expects {self.scaler.n_features_in_}"
                )

                if X_test_flat.shape[1] > self.scaler.n_features_in_:
                    self.logger.info(
                        f"Selecting only the first {self.scaler.n_features_in_} features that were used during training")
                    X_test_flat = X_test_flat[:, :self.scaler.n_features_in_]
                else:
                    padded = np.zeros((X_test_flat.shape[0], self.scaler.n_features_in_))
                    padded[:, :X_test_flat.shape[1]] = X_test_flat
                    X_test_flat = padded

            X_test_scaled = self.scaler.transform(X_test_flat)

            X_test = X_test_scaled.reshape(original_shape[0], original_shape[1], self.scaler.n_features_in_)

            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            return X_test

        except Exception as e:
            self.logger.error(f"Error normalizing test data: {e}")
            return X_test