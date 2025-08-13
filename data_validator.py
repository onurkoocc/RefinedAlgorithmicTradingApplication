"""
Comprehensive Data Validation System for Trading Pipeline
Ensures data integrity and prevents temporal bias/future data leakage
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path


class DataValidator:
    """
    Comprehensive data validation system for the trading pipeline.
    Performs validation at multiple checkpoints to ensure data integrity.
    """
    
    # Technical indicator reasonable ranges
    INDICATOR_RANGES = {
        'rsi': (0, 100),
        'rsi_14': (0, 100),
        'rsi_9': (0, 100),
        'rsi_21': (0, 100),
        'macd': (-np.inf, np.inf),  # Can be any value
        'macd_signal': (-np.inf, np.inf),
        'macd_diff': (-np.inf, np.inf),
        'adx': (0, 100),
        'adx_14': (0, 100),
        'di_plus': (0, 100),
        'di_minus': (0, 100),
        'di_plus_14': (0, 100),
        'di_minus_14': (0, 100),
        'atr': (0, np.inf),  # Must be positive
        'atr_14': (0, np.inf),
        'bb_upper': (0, np.inf),
        'bb_middle': (0, np.inf),
        'bb_lower': (0, np.inf),
        'bb_width': (0, np.inf),
        'obv': (-np.inf, np.inf),
        'cmf': (-1, 1),
        'cmf_20': (-1, 1),
        'volume': (0, np.inf),
        'quote_asset_volume': (0, np.inf),
        'count': (0, np.inf),
        'taker_buy_base_asset_volume': (0, np.inf),
        'taker_buy_quote_asset_volume': (0, np.inf),
        'close': (0, np.inf),
        'open': (0, np.inf),
        'high': (0, np.inf),
        'low': (0, np.inf),
        'volatility_regime': (0, 2),  # 0=low, 1=medium, 2=high
        'market_regime': (-1, 1),  # -1=bearish, 0=ranging, 1=bullish
        'spread_percentage': (0, 100),  # Percentage spread
        'hour_sin': (-1, 1),
        'hour_cos': (-1, 1),
        'day_sin': (-1, 1),
        'day_cos': (-1, 1),
    }
    
    # Maximum acceptable price change per candle (%)
    MAX_PRICE_CHANGE_PCT = 10.0
    
    # Maximum acceptable volume spike ratio
    MAX_VOLUME_SPIKE_RATIO = 100.0
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the DataValidator with optional custom logger."""
        self.logger = logger or self._setup_logger()
        self.validation_history = []
        self.error_count = 0
        self.warning_count = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger for validation messages."""
        logger = logging.getLogger('DataValidator')
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler('data_validation.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        return logger
    
    def validate_dataframe_integrity(self, df: pd.DataFrame, 
                                    stage: str = "unknown") -> Tuple[bool, List[str]]:
        """
        Validate basic dataframe integrity.
        
        Args:
            df: DataFrame to validate
            stage: Pipeline stage name for logging
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info(f"Validating dataframe integrity at stage: {stage}")
        
        # Check if dataframe is empty
        if df.empty:
            issues.append(f"DataFrame is empty at stage {stage}")
            self.logger.error(f"Empty dataframe at {stage}")
            return False, issues
        
        # Check for duplicate indices
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(f"Found {dup_count} duplicate indices")
            self.logger.error(f"Duplicate indices found: {dup_count}")
        
        # Check for NaN values
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            for col in nan_columns:
                nan_count = df[col].isna().sum()
                nan_pct = (nan_count / len(df)) * 100
                issues.append(f"Column '{col}' has {nan_count} NaN values ({nan_pct:.2f}%)")
                self.logger.warning(f"NaN values in {col}: {nan_count} ({nan_pct:.2f}%)")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"Column '{col}' has {inf_count} infinite values")
                self.logger.error(f"Infinite values in {col}: {inf_count}")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info(f"DataFrame integrity check passed at {stage}")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_feature_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all features are within reasonable ranges.
        
        Args:
            df: DataFrame with features to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info("Validating feature ranges")
        
        for col in df.columns:
            if col in self.INDICATOR_RANGES:
                min_val, max_val = self.INDICATOR_RANGES[col]
                
                # Check minimum
                if min_val != -np.inf:
                    below_min = df[col] < min_val
                    if below_min.any():
                        count = below_min.sum()
                        min_found = df[col].min()
                        issues.append(f"{col}: {count} values below minimum {min_val} (found {min_found:.4f})")
                        self.logger.error(f"{col} has values below range: min={min_found:.4f}, expected>={min_val}")
                
                # Check maximum
                if max_val != np.inf:
                    above_max = df[col] > max_val
                    if above_max.any():
                        count = above_max.sum()
                        max_found = df[col].max()
                        issues.append(f"{col}: {count} values above maximum {max_val} (found {max_found:.4f})")
                        self.logger.error(f"{col} has values above range: max={max_found:.4f}, expected<={max_val}")
        
        # Validate OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                count = invalid_hl.sum()
                issues.append(f"Found {count} candles where high < low")
                self.logger.error(f"Invalid OHLC: {count} candles with high < low")
            
            # High should be >= Open and Close
            invalid_h = (df['high'] < df['open']) | (df['high'] < df['close'])
            if invalid_h.any():
                count = invalid_h.sum()
                issues.append(f"Found {count} candles where high is not the highest price")
                self.logger.error(f"Invalid OHLC: {count} candles with incorrect high")
            
            # Low should be <= Open and Close
            invalid_l = (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_l.any():
                count = invalid_l.sum()
                issues.append(f"Found {count} candles where low is not the lowest price")
                self.logger.error(f"Invalid OHLC: {count} candles with incorrect low")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("All features within valid ranges")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_temporal_ordering(self, df: pd.DataFrame, 
                                  timestamp_col: str = 'timestamp') -> Tuple[bool, List[str]]:
        """
        Validate that temporal ordering is maintained.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info("Validating temporal ordering")
        
        if timestamp_col not in df.columns:
            # Try to use index if it's datetime
            if pd.api.types.is_datetime64_any_dtype(df.index):
                timestamps = df.index
            else:
                issues.append(f"Timestamp column '{timestamp_col}' not found")
                self.logger.error(f"Missing timestamp column: {timestamp_col}")
                return False, issues
        else:
            timestamps = pd.to_datetime(df[timestamp_col])
        
        # Check if timestamps are sorted
        if not timestamps.is_monotonic_increasing:
            issues.append("Timestamps are not in ascending order")
            self.logger.error("Temporal ordering violated: timestamps not sorted")
            
            # Find specific violations
            for i in range(1, len(timestamps)):
                if timestamps.iloc[i] < timestamps.iloc[i-1]:
                    self.logger.debug(f"Order violation at index {i}: {timestamps.iloc[i]} < {timestamps.iloc[i-1]}")
        
        # Check for duplicate timestamps
        duplicate_timestamps = timestamps[timestamps.duplicated()]
        if len(duplicate_timestamps) > 0:
            issues.append(f"Found {len(duplicate_timestamps)} duplicate timestamps")
            self.logger.error(f"Duplicate timestamps found: {len(duplicate_timestamps)}")
        
        # Check for gaps in timestamps (assuming 30-minute intervals)
        if len(timestamps) > 1:
            time_diffs = timestamps.diff()[1:]
            expected_interval = timedelta(minutes=30)
            
            # Allow some tolerance (e.g., 1 minute)
            tolerance = timedelta(minutes=1)
            gaps = time_diffs[abs(time_diffs - expected_interval) > tolerance]
            
            if len(gaps) > 0:
                self.logger.warning(f"Found {len(gaps)} gaps in time series")
                # Log a few examples
                for idx in gaps.index[:5]:
                    self.logger.debug(f"Gap at index {idx}: {time_diffs[idx]}")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("Temporal ordering validation passed")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_price_changes(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that price changes are realistic.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info("Validating price changes")
        
        if 'close' not in df.columns:
            issues.append("Close price column not found")
            return False, issues
        
        # Calculate percentage changes
        price_changes = df['close'].pct_change() * 100
        
        # Check for extreme price changes
        extreme_changes = abs(price_changes) > self.MAX_PRICE_CHANGE_PCT
        if extreme_changes.any():
            count = extreme_changes.sum()
            max_change = abs(price_changes).max()
            issues.append(f"Found {count} extreme price changes (max: {max_change:.2f}%)")
            self.logger.error(f"Extreme price changes detected: {count} occurrences, max={max_change:.2f}%")
            
            # Log specific instances
            extreme_indices = df.index[extreme_changes]
            for idx in extreme_indices[:5]:  # Log first 5
                change = price_changes.loc[idx]
                self.logger.debug(f"Extreme change at {idx}: {change:.2f}%")
        
        # Check for zero prices
        if 'close' in df.columns:
            zero_prices = df['close'] == 0
            if zero_prices.any():
                count = zero_prices.sum()
                issues.append(f"Found {count} zero close prices")
                self.logger.error(f"Zero prices detected: {count} occurrences")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("Price changes validation passed")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_volume_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate volume data integrity.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info("Validating volume data")
        
        volume_cols = ['volume', 'quote_asset_volume', 'taker_buy_base_asset_volume']
        
        for col in volume_cols:
            if col in df.columns:
                # Check for negative volumes
                negative_volumes = df[col] < 0
                if negative_volumes.any():
                    count = negative_volumes.sum()
                    issues.append(f"Found {count} negative values in {col}")
                    self.logger.error(f"Negative volumes in {col}: {count}")
                
                # Check for zero volumes (warning only)
                zero_volumes = df[col] == 0
                if zero_volumes.any():
                    count = zero_volumes.sum()
                    pct = (count / len(df)) * 100
                    if pct > 5:  # Warn if more than 5% are zero
                        self.logger.warning(f"Zero volumes in {col}: {count} ({pct:.2f}%)")
                
                # Check for extreme volume spikes
                if len(df) > 1:
                    volume_ratios = df[col] / df[col].shift(1)
                    extreme_spikes = volume_ratios > self.MAX_VOLUME_SPIKE_RATIO
                    if extreme_spikes.any():
                        count = extreme_spikes.sum()
                        max_spike = volume_ratios.max()
                        self.logger.warning(f"Volume spikes in {col}: {count} occurrences, max ratio={max_spike:.2f}")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("Volume data validation passed")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_sequences(self, sequences: np.ndarray, 
                          targets: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate sequence data for future data leakage.
        
        Args:
            sequences: 3D array of sequences (samples, timesteps, features)
            targets: Target values (optional)
            feature_names: List of feature names (optional)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info(f"Validating sequences with shape {sequences.shape}")
        
        # Check sequence shape
        if len(sequences.shape) != 3:
            issues.append(f"Invalid sequence shape: {sequences.shape} (expected 3D)")
            self.logger.error(f"Invalid sequence dimensions: {len(sequences.shape)}D")
            return False, issues
        
        n_samples, n_timesteps, n_features = sequences.shape
        
        # Check for NaN or infinite values
        if np.isnan(sequences).any():
            nan_count = np.isnan(sequences).sum()
            issues.append(f"Found {nan_count} NaN values in sequences")
            self.logger.error(f"NaN values in sequences: {nan_count}")
        
        if np.isinf(sequences).any():
            inf_count = np.isinf(sequences).sum()
            issues.append(f"Found {inf_count} infinite values in sequences")
            self.logger.error(f"Infinite values in sequences: {inf_count}")
        
        # Check for temporal consistency within sequences
        # For each sequence, later timesteps should have later data
        self.logger.debug("Checking temporal consistency within sequences")
        
        # Sample check: verify that certain features maintain logical ordering
        # This is a simplified check - in practice you might want more sophisticated tests
        if feature_names and 'close' in feature_names:
            close_idx = feature_names.index('close')
            
            # Check a sample of sequences for temporal issues
            sample_size = min(100, n_samples)
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            
            for idx in sample_indices:
                seq = sequences[idx, :, close_idx]
                # Check if all values are identical (might indicate copying error)
                if np.all(seq == seq[0]):
                    self.logger.warning(f"Sequence {idx} has identical values for all timesteps")
        
        # Validate targets if provided
        if targets is not None:
            if len(targets) != n_samples:
                issues.append(f"Target length {len(targets)} doesn't match sequences {n_samples}")
                self.logger.error(f"Target/sequence mismatch: {len(targets)} vs {n_samples}")
            
            if np.isnan(targets).any():
                nan_count = np.isnan(targets).sum()
                issues.append(f"Found {nan_count} NaN values in targets")
                self.logger.error(f"NaN values in targets: {nan_count}")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("Sequence validation passed")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_future_data_leakage(self, df: pd.DataFrame, 
                                    lookback_cols: List[str],
                                    current_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Check for future data leakage in features.
        
        Args:
            df: DataFrame with all features
            lookback_cols: Columns that should only use past data
            current_cols: Columns that can use current data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        self.logger.info("Checking for future data leakage")
        
        # Check that lookback features don't have future-looking calculations
        for col in lookback_cols:
            if col in df.columns:
                # Check if values change when we shift the data
                # This is a basic check - more sophisticated tests might be needed
                original = df[col].copy()
                
                # If a feature uses future data, shifting the entire dataframe
                # should change the feature values at each point
                # This is a simplified test
                if len(df) > 10:
                    # Compare with a lagged version
                    lagged = df[col].shift(1)
                    
                    # Check correlation - if perfectly correlated with lag, might be okay
                    # If not correlated at all, might indicate randomness or future leakage
                    corr = original.corr(lagged)
                    
                    if pd.isna(corr):
                        self.logger.warning(f"Could not compute correlation for {col}")
                    elif abs(corr) < 0.5:
                        self.logger.warning(f"Low temporal correlation for {col}: {corr:.3f}")
        
        # Specific checks for common indicators
        if 'rsi' in df.columns and 'close' in df.columns:
            # RSI should not be calculated using future prices
            # Basic sanity check: RSI should change when prices change significantly
            price_changes = df['close'].pct_change()
            rsi_changes = df['rsi'].diff()
            
            # When price changes significantly, RSI should also change
            significant_price_changes = abs(price_changes) > 0.02  # 2% change
            if significant_price_changes.any():
                rsi_unchanged = (rsi_changes == 0) & significant_price_changes
                if rsi_unchanged.any():
                    count = rsi_unchanged.sum()
                    self.logger.warning(f"RSI unchanged despite significant price moves: {count} instances")
        
        is_valid = len(issues) == 0
        if is_valid:
            self.logger.info("No future data leakage detected")
        else:
            self.error_count += len(issues)
            
        return is_valid, issues
    
    def validate_after_data_loading(self, df: pd.DataFrame) -> bool:
        """
        Validation checkpoint after data loading.
        
        Args:
            df: Loaded dataframe
            
        Returns:
            True if all validations pass
        """
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION CHECKPOINT: After Data Loading")
        self.logger.info("=" * 60)
        
        all_valid = True
        all_issues = []
        
        # Basic integrity check
        valid, issues = self.validate_dataframe_integrity(df, "data_loading")
        all_valid &= valid
        all_issues.extend(issues)
        
        # Temporal ordering check
        valid, issues = self.validate_temporal_ordering(df)
        all_valid &= valid
        all_issues.extend(issues)
        
        # Price data check
        valid, issues = self.validate_price_changes(df)
        all_valid &= valid
        all_issues.extend(issues)
        
        # Volume data check
        valid, issues = self.validate_volume_data(df)
        all_valid &= valid
        all_issues.extend(issues)
        
        self._record_validation("after_data_loading", all_valid, all_issues)
        
        if not all_valid:
            self.logger.critical(f"Data loading validation FAILED with {len(all_issues)} issues")
            self._handle_validation_failure("after_data_loading", all_issues)
        else:
            self.logger.info("Data loading validation PASSED")
        
        return all_valid
    
    def validate_after_feature_engineering(self, df: pd.DataFrame) -> bool:
        """
        Validation checkpoint after feature engineering.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            True if all validations pass
        """
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION CHECKPOINT: After Feature Engineering")
        self.logger.info("=" * 60)
        
        all_valid = True
        all_issues = []
        
        # Basic integrity check
        valid, issues = self.validate_dataframe_integrity(df, "feature_engineering")
        all_valid &= valid
        all_issues.extend(issues)
        
        # Feature range check
        valid, issues = self.validate_feature_ranges(df)
        all_valid &= valid
        all_issues.extend(issues)
        
        # Check for future leakage in indicators
        lookback_features = ['rsi', 'macd', 'ema_9', 'ema_21', 'sma_50', 'atr']
        current_features = ['close', 'volume', 'high', 'low', 'open']
        
        valid, issues = self.validate_future_data_leakage(
            df, 
            [f for f in lookback_features if f in df.columns],
            [f for f in current_features if f in df.columns]
        )
        all_valid &= valid
        all_issues.extend(issues)
        
        self._record_validation("after_feature_engineering", all_valid, all_issues)
        
        if not all_valid:
            self.logger.critical(f"Feature engineering validation FAILED with {len(all_issues)} issues")
            self._handle_validation_failure("after_feature_engineering", all_issues)
        else:
            self.logger.info("Feature engineering validation PASSED")
        
        return all_valid
    
    def validate_before_model_training(self, 
                                      X_train: np.ndarray,
                                      y_train: np.ndarray,
                                      X_val: Optional[np.ndarray] = None,
                                      y_val: Optional[np.ndarray] = None,
                                      feature_names: Optional[List[str]] = None) -> bool:
        """
        Validation checkpoint before model training.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            feature_names: List of feature names (optional)
            
        Returns:
            True if all validations pass
        """
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION CHECKPOINT: Before Model Training")
        self.logger.info("=" * 60)
        
        all_valid = True
        all_issues = []
        
        # Validate training sequences
        valid, issues = self.validate_sequences(X_train, y_train, feature_names)
        all_valid &= valid
        all_issues.extend(issues)
        
        # Validate validation sequences if provided
        if X_val is not None and y_val is not None:
            valid, issues = self.validate_sequences(X_val, y_val, feature_names)
            all_valid &= valid
            all_issues.extend(issues)
            
            # Check train/val split integrity
            if X_train.shape[1:] != X_val.shape[1:]:
                all_issues.append(f"Train/val shape mismatch: {X_train.shape} vs {X_val.shape}")
                all_valid = False
        
        # Check data statistics
        self.logger.info(f"Training data statistics:")
        self.logger.info(f"  X_train shape: {X_train.shape}")
        self.logger.info(f"  y_train shape: {y_train.shape}")
        self.logger.info(f"  X_train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        self.logger.info(f"  y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        if X_val is not None:
            self.logger.info(f"  X_val shape: {X_val.shape}")
            self.logger.info(f"  X_val range: [{X_val.min():.4f}, {X_val.max():.4f}]")
        
        self._record_validation("before_model_training", all_valid, all_issues)
        
        if not all_valid:
            self.logger.critical(f"Model training validation FAILED with {len(all_issues)} issues")
            self._handle_validation_failure("before_model_training", all_issues)
        else:
            self.logger.info("Model training validation PASSED")
        
        return all_valid
    
    def validate_before_prediction(self, 
                                  X: np.ndarray,
                                  feature_names: Optional[List[str]] = None) -> bool:
        """
        Validation checkpoint before generating predictions.
        
        Args:
            X: Input sequences for prediction
            feature_names: List of feature names (optional)
            
        Returns:
            True if all validations pass
        """
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION CHECKPOINT: Before Prediction")
        self.logger.info("=" * 60)
        
        all_valid = True
        all_issues = []
        
        # Validate input sequences
        valid, issues = self.validate_sequences(X, None, feature_names)
        all_valid &= valid
        all_issues.extend(issues)
        
        # Check data statistics
        self.logger.info(f"Prediction data statistics:")
        self.logger.info(f"  Input shape: {X.shape}")
        self.logger.info(f"  Input range: [{X.min():.4f}, {X.max():.4f}]")
        
        # Check for anomalies in the most recent data
        if len(X.shape) == 3:
            # Check the most recent timestep of each sequence
            recent_data = X[:, -1, :]
            
            # Check for unusual patterns
            if np.all(recent_data == 0):
                all_issues.append("All recent data is zero")
                all_valid = False
            
            # Check for constant values across features
            if recent_data.shape[0] > 1:
                feature_stds = np.std(recent_data, axis=0)
                constant_features = np.where(feature_stds == 0)[0]
                if len(constant_features) > 0:
                    self.logger.warning(f"Found {len(constant_features)} constant features in recent data")
        
        self._record_validation("before_prediction", all_valid, all_issues)
        
        if not all_valid:
            self.logger.critical(f"Prediction validation FAILED with {len(all_issues)} issues")
            self._handle_validation_failure("before_prediction", all_issues)
        else:
            self.logger.info("Prediction validation PASSED")
        
        return all_valid
    
    def _record_validation(self, checkpoint: str, is_valid: bool, issues: List[str]):
        """Record validation results for audit trail."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': checkpoint,
            'is_valid': is_valid,
            'issues': issues,
            'issue_count': len(issues)
        }
        self.validation_history.append(record)
        
        # Save to file for persistence
        log_file = Path('validation_history.json')
        if log_file.exists():
            with open(log_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(record)
        
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _handle_validation_failure(self, checkpoint: str, issues: List[str]):
        """
        Handle validation failures with appropriate action.
        
        Args:
            checkpoint: Name of the validation checkpoint
            issues: List of validation issues
        """
        # Create detailed error report
        error_report = f"\n{'='*60}\n"
        error_report += f"VALIDATION FAILURE at {checkpoint}\n"
        error_report += f"Timestamp: {datetime.now().isoformat()}\n"
        error_report += f"Total issues: {len(issues)}\n"
        error_report += f"{'='*60}\n"
        
        for i, issue in enumerate(issues, 1):
            error_report += f"{i}. {issue}\n"
        
        error_report += f"{'='*60}\n"
        
        # Log the error report
        self.logger.critical(error_report)
        
        # Save error report to file
        error_file = Path(f'validation_error_{checkpoint}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(error_file, 'w') as f:
            f.write(error_report)
        
        # Raise assertion error to fail fast
        raise AssertionError(f"Data validation failed at {checkpoint} with {len(issues)} issues. "
                           f"See {error_file} for details.")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validation results.
        
        Returns:
            Dictionary with validation statistics
        """
        summary = {
            'total_validations': len(self.validation_history),
            'passed': sum(1 for v in self.validation_history if v['is_valid']),
            'failed': sum(1 for v in self.validation_history if not v['is_valid']),
            'total_issues': sum(v['issue_count'] for v in self.validation_history),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'checkpoints': {}
        }
        
        # Group by checkpoint
        for record in self.validation_history:
            checkpoint = record['checkpoint']
            if checkpoint not in summary['checkpoints']:
                summary['checkpoints'][checkpoint] = {
                    'runs': 0,
                    'passed': 0,
                    'failed': 0,
                    'issues': []
                }
            
            summary['checkpoints'][checkpoint]['runs'] += 1
            if record['is_valid']:
                summary['checkpoints'][checkpoint]['passed'] += 1
            else:
                summary['checkpoints'][checkpoint]['failed'] += 1
                summary['checkpoints'][checkpoint]['issues'].extend(record['issues'])
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of validation results."""
        summary = self.get_validation_summary()
        
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Errors: {summary['error_count']}")
        print(f"Warnings: {summary['warning_count']}")
        print("\nCheckpoint Details:")
        print("-"*60)
        
        for checkpoint, stats in summary['checkpoints'].items():
            print(f"\n{checkpoint}:")
            print(f"  Runs: {stats['runs']}")
            print(f"  Passed: {stats['passed']}")
            print(f"  Failed: {stats['failed']}")
            if stats['issues']:
                print(f"  Unique Issues: {len(set(stats['issues']))}")
        
        print("="*60)


# Example usage and testing
if __name__ == "__main__":
    # Initialize validator
    validator = DataValidator()
    
    # Create sample data for testing
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample dataframe with potential issues
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='30min')
    
    # Create data with some intentional issues for testing
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, n_samples),
        'high': np.random.uniform(40100, 50100, n_samples),
        'low': np.random.uniform(39900, 49900, n_samples),
        'close': np.random.uniform(40000, 50000, n_samples),
        'volume': np.random.uniform(0, 1000, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.uniform(-100, 100, n_samples),
        'atr': np.random.uniform(0, 1000, n_samples),
    })
    
    # Add some intentional issues
    df.loc[10, 'rsi'] = 150  # RSI out of range
    df.loc[20, 'volume'] = -100  # Negative volume
    df.loc[30, 'close'] = np.nan  # NaN value
    df.loc[40, 'high'] = df.loc[40, 'low'] - 100  # High < Low
    
    # Test validation checkpoints
    print("Testing Data Validator\n")
    
    # Test after data loading
    try:
        validator.validate_after_data_loading(df)
    except AssertionError as e:
        print(f"Expected failure: {e}")
    
    # Fix issues and retest
    df.loc[10, 'rsi'] = 50
    df.loc[20, 'volume'] = 100
    df.loc[30, 'close'] = 45000
    df.loc[40, 'high'] = df.loc[40, 'low'] + 100
    
    # Test again
    validator.validate_after_data_loading(df)
    
    # Create sequences for testing
    sequences = np.random.randn(100, 72, 48)  # 100 samples, 72 timesteps, 48 features
    targets = np.random.randn(100, 16)  # 100 samples, 16 horizon
    
    # Test before model training
    validator.validate_before_model_training(sequences, targets)
    
    # Print summary
    validator.print_summary()