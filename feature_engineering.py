"""
Feature Engineering module for Bitcoin trading system.
Uses pandas-ta for technical indicators with essential features only.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
import pandas_ta as ta


class FeatureEngineer:
    """
    Feature engineering class that creates essential technical indicators.
    Optimized for Bitcoin 30-minute timeframe trading.
    """
    
    # Essential features proven effective for crypto trading
    ESSENTIAL_FEATURES = {
        'returns': 'Simple returns',
        'log_returns': 'Log returns for better normalization',
        'realized_volatility': 'Rolling volatility measure',
        'volume_ratio': 'Volume relative to average',
        'dollar_volume': 'Log of dollar volume',
        'rsi_14': 'RSI normalized to [0,1]',
        'rate_of_change': 'Momentum indicator',
        'ema_cross_signal': 'EMA 9/21 crossover signal',
        'adx_14': 'Trend strength normalized',
        'price_vs_sma': 'Price deviation from SMA',
        'high_low_spread': 'Volatility from high-low spread',
        'volume_imbalance': 'Buy/sell volume imbalance',
        'macd_histogram': 'MACD histogram for momentum detection'
    }
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize with config for compatibility."""
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # Extract feature config safely
        try:
            if hasattr(config, 'get') and callable(getattr(config, 'get')):
                self.feature_config = config.get('features', {})
            elif hasattr(config, 'features'):
                self.feature_config = getattr(config, 'features', {})
            else:
                self.feature_config = {}
            
            if self.feature_config is None:
                self.feature_config = {}
                
        except Exception as e:
            self.logger.warning(f"Error extracting feature config: {e}. Using defaults.")
            self.feature_config = {}
        
        self.logger.info(f"FeatureEngineer initialized with {len(self.ESSENTIAL_FEATURES)} essential features")
        
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger."""
        logger = logging.getLogger('FeatureEngineer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def create_features(self, df: pd.DataFrame, chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Create essential features for trading.
        
        Args:
            df: Input DataFrame with OHLCV data
            chunk_size: Optional chunk size for processing large datasets
            
        Returns:
            DataFrame with essential features
        """
        self.logger.info("Creating essential features")
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = [col.lower() for col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process in chunks if specified
        if chunk_size and len(df) > chunk_size:
            return self._process_in_chunks(df, chunk_size)
        
        # Create features
        features_df = self._create_all_features(df)
        
        # Validate features
        self._validate_features(features_df)
        
        self.logger.info(f"Created {len(features_df.columns)} features")
        
        return features_df
    
    def _process_in_chunks(self, df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
        """Process large datasets in chunks."""
        chunks = []
        overlap = 100  # Keep overlap for indicator calculation
        
        for i in range(0, len(df), chunk_size - overlap):
            chunk_end = min(i + chunk_size, len(df))
            chunk = df.iloc[i:chunk_end].copy()
            
            # Process chunk
            chunk_features = self._create_all_features(chunk)
            
            # Remove overlap from all but last chunk
            if i > 0 and i + chunk_size < len(df):
                chunk_features = chunk_features.iloc[overlap:]
            elif i > 0:  # Last chunk
                chunk_features = chunk_features.iloc[overlap:]
            
            chunks.append(chunk_features)
        
        return pd.concat(chunks, ignore_index=False)
    
    def _create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all essential features."""
        # Start with original OHLCV data
        features = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # 1. Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatility
        features['realized_volatility'] = features['returns'].rolling(window=20).std()
        
        # 3. Volume features
        volume_ma = df['volume'].rolling(window=20).mean()
        features['volume_ratio'] = df['volume'] / volume_ma.where(volume_ma > 0, 1)
        features['dollar_volume'] = np.log(df['close'] * df['volume'] + 1)
        
        # 4. RSI (normalized to 0-1)
        rsi = ta.rsi(df['close'], length=14)
        features['rsi_14'] = rsi / 100.0
        
        # 5. Rate of Change
        features['rate_of_change'] = ta.roc(df['close'], length=10) / 100.0
        
        # 6. EMA Cross Signal
        ema_9 = ta.ema(df['close'], length=9)
        ema_21 = ta.ema(df['close'], length=21)
        features['ema_cross_signal'] = (ema_9 - ema_21) / df['close']
        
        # 7. ADX (normalized)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            features['adx_14'] = adx['ADX_14'] / 100.0
        else:
            features['adx_14'] = 0.25  # Default value
        
        # 8. Price vs SMA
        sma_20 = ta.sma(df['close'], length=20)
        features['price_vs_sma'] = (df['close'] - sma_20) / sma_20.where(sma_20 > 0, df['close'])
        
        # 9. High-Low Spread
        features['high_low_spread'] = (df['high'] - df['low']) / df['close']
        
        # 10. Volume Imbalance (simplified)
        up_moves = df['close'] > df['open']
        features['volume_imbalance'] = np.where(up_moves, df['volume'], -df['volume'])
        features['volume_imbalance'] = features['volume_imbalance'].rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
        
        # 11. MACD Histogram
        try:
            # Use pandas-ta to calculate MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                # Get the histogram column (difference between MACD and signal)
                if 'MACDh_12_26_9' in macd_result.columns:
                    features['macd_histogram'] = macd_result['MACDh_12_26_9'] / df['close'] * 100  # Normalize by price
                else:
                    # Fallback: calculate manually
                    ema12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['close'].ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9, adjust=False).mean()
                    features['macd_histogram'] = (macd - signal) / df['close'] * 100
            else:
                # Manual calculation if pandas-ta fails
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                features['macd_histogram'] = (macd - signal) / df['close'] * 100
                
            # Log the MACD values for debugging
            if not features['macd_histogram'].isna().all():
                macd_mean = features['macd_histogram'].mean()
                macd_std = features['macd_histogram'].std()
                self.logger.info(f"MACD histogram calculated: mean={macd_mean:.6f}, std={macd_std:.6f}")
            else:
                self.logger.warning("MACD histogram contains all NaN values")
                
        except Exception as e:
            self.logger.warning(f"Error calculating MACD: {e}. Using fallback calculation.")
            # Fallback calculation
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_histogram'] = (macd - signal) / df['close'] * 100
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        # Clip extreme values
        for col in features.columns:
            if col not in ['returns', 'log_returns']:
                features[col] = features[col].clip(lower=features[col].quantile(0.001),
                                                   upper=features[col].quantile(0.999))
        
        return features
    
    def _validate_features(self, features_df: pd.DataFrame) -> None:
        """Validate created features."""
        # Check for NaN values
        nan_counts = features_df.isna().sum()
        if nan_counts.any():
            self.logger.warning(f"NaN values found in features: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check for infinite values
        inf_counts = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
        if inf_counts.any():
            self.logger.warning(f"Infinite values found in features: {inf_counts[inf_counts > 0].to_dict()}")
        
        # Check feature ranges
        for col in features_df.columns:
            if features_df[col].std() == 0:
                self.logger.warning(f"Feature {col} has zero variance")
                
        # Specific validation for MACD histogram
        if 'macd_histogram' in features_df.columns:
            macd_values = features_df['macd_histogram'].dropna()
            if len(macd_values) > 0:
                if macd_values.std() == 0 or macd_values.abs().max() == 0:
                    self.logger.error(f"MACD histogram appears to be all zeros! Mean: {macd_values.mean():.6f}, Std: {macd_values.std():.6f}")
                else:
                    self.logger.info(f"MACD histogram validation passed. Range: [{macd_values.min():.6f}, {macd_values.max():.6f}], Non-zero: {(macd_values != 0).sum()}/{len(macd_values)}")
            else:
                self.logger.error("MACD histogram column exists but contains no valid values!")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names for compatibility."""
        return list(self.ESSENTIAL_FEATURES.keys())
    
    def get_feature_count(self) -> int:
        """Get number of features."""
        return len(self.ESSENTIAL_FEATURES)
    
    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """Validate features for compatibility."""
        try:
            self._validate_features(features_df)
            return True
        except Exception as e:
            self.logger.error(f"Feature validation failed: {e}")
            return False