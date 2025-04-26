import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import Dense, BatchNormalization, GRU, Dropout, Lambda, Concatenate, Bidirectional
from tensorflow.keras.layers import Add, LayerNormalization, GlobalAveragePooling1D, Multiply, Layer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import json
from pathlib import Path


class OptimizedGrowthMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, fwd_returns_val, monthly_target=0.08, threshold_pct=0.6, model_idx=None,
                 transaction_cost=0.001, drawdown_weight=1.8, avg_return_weight=2.0, consistency_weight=1.0,
                 fixed_threshold=None, volatility_lookback=30, adaptive_threshold=True, min_trades_penalty=True):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.fwd_returns_val = fwd_returns_val
        self.threshold_pct = threshold_pct
        self.model_idx = model_idx
        self.monthly_target = monthly_target
        self.transaction_cost = transaction_cost
        self.drawdown_weight = drawdown_weight
        self.avg_return_weight = avg_return_weight
        self.consistency_weight = consistency_weight
        self.fixed_threshold = fixed_threshold
        self.volatility_lookback = volatility_lookback
        self.adaptive_threshold = adaptive_threshold
        self.min_trades_penalty = min_trades_penalty
        self.trade_periods_per_month = 1440
        self.best_metrics = {
            'growth_score': -np.inf, 'monthly_growth': 0.0, 'val_sharpe': 0.0,
            'val_sortino': 0.0, 'val_calmar': 0.0, 'max_drawdown': 1.0,
            'trades_per_month': 0
        }
        self.historical_returns = []
        self.threshold_history = []
        self.max_drawdown_threshold = 0.25
        self.consecutive_loss_scale = 0.85
        self.max_position_size = 0.5

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.X_val) == 0:
            self._update_logs_with_defaults(logs)
            return

        try:
            batch_size = min(len(self.X_val), 128)
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=batch_size)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

            threshold = self._determine_threshold(y_pred)
            self.threshold_history.append(threshold)

            trade_indices = np.where(np.abs(y_pred) > threshold)[0]
            if len(trade_indices) == 0:
                self._update_logs_with_defaults(logs)
                return

            self._calculate_enhanced_metrics(trade_indices, y_pred, logs, epoch)

        except Exception as e:
            print(f"Error in OptimizedGrowthMetricCallback: {e}")
            import traceback
            print(traceback.format_exc())
            self._update_logs_with_defaults(logs)

    def _determine_threshold(self, y_pred):
        if self.fixed_threshold is not None:
            return self.fixed_threshold

        sorted_preds = np.sort(np.abs(y_pred).flatten())
        threshold_idx = max(1, int(len(sorted_preds) * (1 - self.threshold_pct)))
        base_threshold = sorted_preds[threshold_idx] if len(sorted_preds) > threshold_idx else 0.001
        base_threshold = max(base_threshold, 0.001)

        if not self.adaptive_threshold:
            return base_threshold

        volatility = self._calculate_volatility()
        base_volatility = 0.01
        volatility_factor = volatility / base_volatility

        market_regime = self._detect_market_regime()
        regime_factor = 1.0
        if market_regime == "bullish":
            regime_factor = 0.9
        elif market_regime == "bearish":
            regime_factor = 1.2
        elif market_regime == "volatile":
            regime_factor = 1.5

        performance_factor = 1.0
        if len(self.historical_returns) >= 5:
            recent_epochs = self.historical_returns[-5:]
            recent_scores = [h.get('growth_score', 0) for h in recent_epochs]
            avg_recent_score = np.mean(recent_scores) if recent_scores else 0
            performance_factor = 1.0 - np.clip((avg_recent_score - 0.5) * 0.4, -0.2, 0.2)

        return max(base_threshold * volatility_factor * regime_factor * performance_factor, 0.0005)

    def _calculate_volatility(self):
        lookback = min(self.volatility_lookback, len(self.fwd_returns_val))
        recent_returns = self.fwd_returns_val[-lookback:]
        return np.std(recent_returns) if len(recent_returns) > 0 else 0.01

    def _detect_market_regime(self):
        if len(self.fwd_returns_val) < 20:
            return "neutral"

        recent_returns = self.fwd_returns_val[-20:]
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)

        if volatility > 0.02:
            return "volatile"
        elif avg_return > 0.5 * volatility:
            return "bullish"
        elif avg_return < -0.5 * volatility:
            return "bearish"
        else:
            return "neutral"

    def _update_logs_with_defaults(self, logs):
        metrics = ['growth_score', 'monthly_growth', 'val_sharpe', 'val_sortino', 'val_calmar',
                   'max_drawdown', 'val_win_rate', 'val_profit_factor', 'val_trades_per_month', 'consistency_score']
        for m in metrics:
            logs[m] = 0.0
        if self.model_idx is not None:
            logs[f'growth_score_model_{self.model_idx}'] = 0.0

    def _calculate_enhanced_metrics(self, trade_indices, y_pred, logs, epoch):
        trade_returns = []
        win_count = 0
        profit_sum = 0
        loss_sum = 0
        win_amounts = []
        loss_amounts = []
        equity_curve = [1.0]
        trade_timestamps = []
        market_volatility = self._calculate_volatility()
        transaction_cost = self.transaction_cost * (
                    1 + 0.5 * max(0, (market_volatility / 0.01) - 1))  # Added max(0,...)
        consecutive_losses = 0
        max_consecutive_losses = 0
        max_trades_per_day = 5
        trades_today = 0
        current_day = -1  # Initialize to handle first trade day correctly

        for idx in trade_indices:
            if idx >= len(self.fwd_returns_val): continue

            # Day logic correction - integer division floor // ensures correct day bucket assignment
            day_of_trade = idx // 48
            if day_of_trade != current_day:
                current_day = day_of_trade
                trades_today = 0
            if trades_today >= max_trades_per_day: continue

            trades_today += 1
            actual_return = float(self.fwd_returns_val[idx])
            pred_direction = np.sign(y_pred[idx].flatten()[0])
            pred_confidence = abs(y_pred[idx].flatten()[0])

            # Ensure minimum confidence/threshold isn't zero leading to tiny position sizes
            base_position_size = max(0.001, pred_confidence / 0.005)
            position_size = min(self.max_position_size, max(0.2, base_position_size))

            if consecutive_losses >= 3:
                # Apply gradual reduction, prevent excessive scaling down
                reduction_factor = max(0.3, self.consecutive_loss_scale ** (consecutive_losses - 2))
                position_size *= reduction_factor

            trade_return = float(pred_direction * actual_return * position_size) - transaction_cost
            trade_returns.append(trade_return)

            if trade_return > 0:
                win_count += 1
                profit_sum += trade_return
                win_amounts.append(trade_return)
                consecutive_losses = 0
            else:
                loss_sum += abs(trade_return)
                loss_amounts.append(trade_return)
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

            # Defensive update to prevent division by zero if previous equity is somehow zero
            prev_equity = equity_curve[-1] if len(equity_curve) > 0 else 1.0
            current_equity = max(1e-9, prev_equity * (1.0 + trade_return))  # Ensure non-negative/zero equity
            equity_curve.append(current_equity)
            trade_timestamps.append(idx)

            peak_equity = max(equity_curve) if equity_curve else 1.0
            current_drawdown = (peak_equity - current_equity) / max(peak_equity, 1e-9)  # Avoid division by zero
            if current_drawdown > self.max_drawdown_threshold:
                # Exit loop if max drawdown is breached
                break

        n_trades = len(trade_returns)
        if n_trades == 0:
            self._update_logs_with_defaults(logs)
            return

        avg_return = np.mean(trade_returns) if n_trades > 0 else 0.0  # Use np.mean for safety
        win_rate = win_count / n_trades if n_trades > 0 else 0.0
        profit_factor = profit_sum / max(loss_sum, 1e-10)

        # Calculate daily returns safely
        daily_equity = [equity_curve[0]]
        periods_per_day = 48
        for i in range(1, len(equity_curve)):
            if i % periods_per_day == 0:
                daily_equity.append(equity_curve[i])
        if len(equity_curve) % periods_per_day != 1:  # Ensure last point is captured
            daily_equity.append(equity_curve[-1])

        # Handle cases where equity drops to zero or very close
        daily_returns = [np.log(max(1e-9, daily_equity[i]) / max(1e-9, daily_equity[i - 1])) for i in
                         range(1, len(daily_equity))]

        if len(daily_returns) > 1:
            returns_mean = np.mean(daily_returns)
            returns_std = max(np.std(daily_returns), 1e-10)
            down_returns = np.array([r for r in daily_returns if r < 0])
            # Check if down_returns is empty before calculating std
            downside_std = np.std(down_returns) if len(down_returns) > 1 else returns_std
            sharpe_ratio = returns_mean / returns_std * np.sqrt(365)
            sortino_ratio = returns_mean / max(downside_std, 1e-10) * np.sqrt(365)
        else:
            returns_std = max(np.std(trade_returns), 1e-10)
            down_returns = np.array([r for r in trade_returns if r < 0])
            downside_std = np.std(down_returns) if len(down_returns) > 1 else returns_std
            trades_per_year = (n_trades / len(self.X_val)) * self.trade_periods_per_month * 12
            sharpe_ratio = avg_return / returns_std * np.sqrt(max(trades_per_year, 0))  # ensure non-negative sqrt
            sortino_ratio = avg_return / max(downside_std, 1e-10) * np.sqrt(max(trades_per_year, 0))

        # Drawdown calculation needs peak tracking
        peak = equity_curve[0]
        drawdowns = []
        for equity in equity_curve:
            peak = max(peak, equity)
            dd = (peak - equity) / max(peak, 1e-9)  # avoid division by zero
            drawdowns.append(dd)

        max_drawdown = max(drawdowns) if drawdowns else 0.0
        total_return = equity_curve[-1] / equity_curve[0] - 1.0
        calmar_ratio = total_return / max(max_drawdown, 0.01)  # Ensure non-zero divisor

        # Max consecutive wins/losses calculation reset was correct but init value was not
        # Use the value calculated during the trade loop
        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        trades_per_period = n_trades / len(self.X_val) if len(self.X_val) > 0 else 0.0
        trades_per_month = trades_per_period * self.trade_periods_per_month
        monthly_growth = ((equity_curve[-1] / equity_curve[0]) ** (
                    self.trade_periods_per_month / n_trades) - 1.0) if n_trades > 0 else -1.0  # Direct compound calculation
        monthly_growth = min(monthly_growth, 10.0)  # Cap unrealistic growth

        window_size = min(max(5, n_trades // 10), 20, n_trades - 1) if n_trades > 1 else 0
        consistency_score = 0
        if window_size > 0:
            returns_sequence = np.array(trade_returns)
            rolling_returns = [np.mean(returns_sequence[i:i + window_size]) for i in
                               range(len(returns_sequence) - window_size + 1)]
            consistency_score = 1.0 / (np.std(rolling_returns) + 1e-10)

        # Ensure recovery periods logic is robust
        recovery_periods = []
        in_drawdown = False
        drawdown_start_idx = 0
        peak_equity = equity_curve[0]
        recovery_threshold_ratio = 0.95  # Recover 95% of peak before considering recovery

        for i in range(1, len(equity_curve)):
            if not in_drawdown:
                peak_equity = max(peak_equity, equity_curve[i - 1])  # Peak is updated *before* current equity check
                current_drawdown = (peak_equity - equity_curve[i]) / max(peak_equity, 1e-9)
                if current_drawdown > self.max_drawdown_threshold:
                    in_drawdown = True
                    drawdown_start_idx = i
            elif in_drawdown:
                peak_equity = max(peak_equity, equity_curve[i - 1])  # Keep tracking peak even in drawdown
                # Recovery definition: return close to the *peak before the drawdown started*
                if equity_curve[i] >= peak_equity * recovery_threshold_ratio:  # Removed redundant threshold
                    in_drawdown = False
                    recovery_periods.append(i - drawdown_start_idx)
        avg_recovery = np.mean(recovery_periods) if recovery_periods else n_trades
        recovery_efficiency = n_trades / max(avg_recovery, 1.0)  # Based on trade counts, not time index directly

        # Reward/Penalty calculation
        target_achieved_bonus = 0.0
        if monthly_growth >= self.monthly_target:
            target_achieved_bonus = min(0.5, (monthly_growth / self.monthly_target - 1.0) * 0.5)

        growth_distance = 0.0
        if monthly_growth > 0:
            growth_distance = (monthly_growth / self.monthly_target) ** 0.7  # Only positive growth contributes
        elif monthly_growth < 0:  # Add penalty for negative growth relative to target
            growth_distance = -np.abs(
                monthly_growth / self.monthly_target) * 0.5  # Penalize missing target when negative

        dd_penalty = np.exp(-6.0 * max_drawdown)  # Exponential penalty for drawdown

        consistency_score = np.clip(consistency_score, 0, 500)  # Cap consistency

        # Normalizing factors based on typical ranges observed might be needed empirically
        growth_score = (
                               self.avg_return_weight * growth_distance +
                               self.drawdown_weight * dd_penalty +
                               self.consistency_weight * (consistency_score / 500.0) +  # Normalize consistency
                               target_achieved_bonus
                       ) / (self.avg_return_weight + self.drawdown_weight + self.consistency_weight + 1.0)

        # Clamp sortino before using, positive impact should scale reasonably
        sortino_contribution = min(1.0, max(0, sortino_ratio / 2.0))  # Normalize sortino effect
        growth_score += 0.2 * sortino_contribution

        # Profit factor contribution adjustment
        profit_factor_term = 0.4 + 0.6 * min(1.0, max(0, profit_factor / 2.0))  # Ensure > 0
        growth_score *= profit_factor_term

        # Adjust consecutive loss penalty (ensure it doesn't overpower excessively)
        consecutive_loss_penalty = min(0.3, max(0, max_consecutive_losses * 0.005))  # Less aggressive penalty
        growth_score *= (1.0 - consecutive_loss_penalty)

        # Add recovery efficiency contribution more smoothly
        recovery_eff_contribution = 0.65 + 0.35 * min(1.0, max(0,
                                                               recovery_efficiency / 10.0))  # Normalize recovery eff effect
        growth_score *= recovery_eff_contribution

        if self.min_trades_penalty and trades_per_month < 20:
            growth_score *= max(0.5, (trades_per_month / 40))  # Scale penalty based on trades

        # Final sanity check for bounds
        growth_score = max(-100.0, min(100.0, growth_score))

        logs.update({
            'growth_score': float(growth_score),
            'monthly_growth': float(monthly_growth),
            'val_sharpe': float(sharpe_ratio),
            'val_sortino': float(sortino_ratio),
            'val_calmar': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'val_win_rate': float(win_rate),
            'val_profit_factor': float(profit_factor),
            'val_trades_per_month': float(trades_per_month),
            'consistency_score': float(consistency_score),
            'recovery_efficiency': float(recovery_efficiency),
            'max_consecutive_losses': float(max_consecutive_losses)
        })

        if self.model_idx is not None:
            logs[f'growth_score_model_{self.model_idx}'] = float(growth_score)

        self.historical_returns.append({
            'epoch': epoch, 'growth_score': growth_score, 'monthly_growth': monthly_growth,
            'max_drawdown': max_drawdown, 'trades_per_month': trades_per_month,
            'sharpe': sharpe_ratio, 'sortino': sortino_ratio,
            'threshold': self.threshold_history[-1] if self.threshold_history else 0.0,
            'consecutive_losses': max_consecutive_losses
        })

        if growth_score > self.best_metrics['growth_score']:
            self.best_metrics.update({
                'growth_score': growth_score, 'monthly_growth': monthly_growth,
                'val_sharpe': sharpe_ratio, 'val_sortino': sortino_ratio,
                'val_calmar': calmar_ratio, 'max_drawdown': max_drawdown,
                'trades_per_month': trades_per_month, 'epoch': epoch,
                'max_consecutive_losses': max_consecutive_losses
            })

        # Adjusted reporting to avoid double printing the Growth Score
        if epoch % 5 == 0 or epoch == 0:
            print(f"\nGrowth Metrics - Epoch {epoch}")
            print(f"Growth Score: {growth_score:.4f}, Monthly Growth: {monthly_growth * 100:.2f}%")
            print(f"Trades/Month: {trades_per_month:.1f}, Win Rate: {win_rate:.2f}, Profit Factor: {profit_factor:.2f}")
            print(f"Sharpe: {sharpe_ratio:.2f}, Sortino: {sortino_ratio:.2f}, Calmar: {calmar_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown * 100:.2f}%, Consistency: {consistency_score:.2f}")
            print(f"Max Consecutive Losses: {max_consecutive_losses}")
            if monthly_growth >= self.monthly_target:
                print(f"✅ On track for target growth of {self.monthly_target * 100:.1f}% monthly")
            else:
                print(f"⚠️ Below target growth of {self.monthly_target * 100:.1f}% monthly")
            if max_drawdown > 0.12:
                print(f"⚠️ High drawdown risk: {max_drawdown * 100:.1f}%")


class NoisySequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, noise_level=0.01, data_augmentation=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.std_dev = np.std(X, axis=(0, 1)) + 1e-7  # Avoid zero std dev
        self.data_augmentation = data_augmentation
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def __len__(self):
        # Ensure all samples are covered, even if the last batch is smaller
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]

        noise = np.random.normal(0, self.noise_level * self.std_dev, batch_X.shape)
        batch_X = batch_X + noise

        if self.data_augmentation:
            for i in range(len(batch_X)):
                # Time Shifting
                if np.random.random() < 0.3:
                    start = np.random.randint(0, batch_X.shape[1] // 4)
                    batch_X[i] = np.roll(batch_X[i], shift=start, axis=0)

                # Feature Masking
                if np.random.random() < 0.2:
                    mask_prob = 0.2
                    mask = np.random.random(batch_X[i].shape) > mask_prob
                    batch_X[i] = batch_X[i] * mask

        return batch_X, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class OptimizedHybridModel:
    def __init__(self, config, input_shape, feature_names=None):
        self.config = config
        self.input_shape = input_shape
        self.feature_names = feature_names
        self.model = None
        self.logger = logging.getLogger("OptimizedHybridModel")
        self.training_metrics = {}
        # Slightly adjusted defaults based on log analysis / planned changes
        self.best_params = {
            "recurrent_units": 48,
            "recurrent_dropout": 0.2,
            "dropout": 0.3,
            "dense_units1": 48,
            "dense_units2": 24,
            "l2_lambda": 1e-3,  # Adjusted from 5e-3
            "learning_rate": 5e-5,
            "epochs": 32
        }

    def build_model(self):
        tf.keras.backend.clear_session()
        inputs = Input(shape=self.input_shape, dtype=tf.float32, name="hybrid_input")

        x = BatchNormalization(momentum=0.99, epsilon=1e-5)(inputs)

        recurrent_units = self.best_params["recurrent_units"]
        recurrent_dropout = self.best_params["recurrent_dropout"]
        dropout_rate = self.best_params["dropout"]

        bidirectional_gru = Bidirectional(
            GRU(recurrent_units, return_sequences=True, recurrent_dropout=recurrent_dropout),
            name='bidirectional_gru'
        )(x)

        x = bidirectional_gru
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        # --- Attention Removed ---
        # Use GlobalAveragePooling instead
        x = GlobalAveragePooling1D()(x)
        # --------------------------

        dense_units1 = self.best_params["dense_units1"]
        dense_units2 = self.best_params["dense_units2"]
        l2_lambda = self.best_params["l2_lambda"]

        x = Dense(dense_units1, activation='swish', kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units2, activation='swish', kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='tanh', kernel_regularizer=l2(l2_lambda), dtype='float32')(x)

        model = Model(inputs, outputs, name="optimized_hybrid_model")
        learning_rate = self.best_params["learning_rate"]
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0),
                      metrics=['mae'])  # Switched to Huber Loss

        self.model = model
        return model

    def train(self, train_seq, X_val, y_val, fwd_returns_val, callbacks=None):
        if self.model is None:
            self.build_model()

        epochs = self.best_params["epochs"]
        default_callbacks = [
            OptimizedGrowthMetricCallback(X_val, y_val, fwd_returns_val),
            tf.keras.callbacks.EarlyStopping(monitor='growth_score', patience=8, mode='max', restore_best_weights=True)
        ]
        used_callbacks = callbacks if callbacks else default_callbacks

        history = self.model.fit(
            train_seq,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=used_callbacks,
            verbose=1  # Set verbose=1 to see progress per step
        )

        # Update training metrics using the best epoch values if available
        if 'val_monthly_growth' in history.history:  # Check if callback metrics exist
            best_epoch_idx = np.argmax(history.history.get('growth_score', [0]))
            self.training_metrics = {
                'monthly_growth': history.history.get('monthly_growth', [0])[best_epoch_idx],
                'val_win_rate': history.history.get('val_win_rate', [0])[best_epoch_idx],
                'val_profit_factor': history.history.get('val_profit_factor', [0])[best_epoch_idx],
                'growth_score': history.history.get('growth_score', [-np.inf])[best_epoch_idx]
            }
        else:  # Fallback if metrics not found (e.g., interrupted training)
            self.training_metrics = {'growth_score': -np.inf}

        return history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call build_model() or train() first.")
        return self.model.predict(X, verbose=0)

    def save_model(self, path):
        if self.model:
            self.model.save(path)
            self.logger.info(f"Saved model to {path}")
            params_path = f"{os.path.splitext(path)[0]}_params.json"
            try:
                with open(params_path, 'w') as f:
                    json.dump(self.best_params, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save parameters: {e}")

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(
                    path,
                    # Custom objects can be added here if needed later
                    # custom_objects={'CustomLoss': self._custom_loss_placeholder}
                    compile=True  # Set compile=True assuming the loaded model was compiled
                )
                self.logger.info(f"Loaded model from {path}")
                params_path = f"{os.path.splitext(path)[0]}_params.json"
                if os.path.exists(params_path):
                    try:
                        with open(params_path, 'r') as f:
                            loaded_params = json.load(f)
                            # Optionally update only if necessary or merge intelligently
                            self.best_params.update(loaded_params)
                    except Exception as e:
                        self.logger.warning(f"Failed to load parameters file {params_path}: {e}. Using defaults.")

                return True
            except Exception as e:
                self.logger.error(f"Failed to load model from {path}: {e}")
                self.model = None
                return False
        self.logger.warning(f"Model file not found at {path}")
        return False


class TradingModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("OptimizedTradingModel")
        # Use a more specific name based on section/key or default
        self.model_path = config.get("model", "model_path")
        self.sequence_length = config.get_typed("model", "sequence_length", 72)
        self.horizon = config.get_typed("model", "horizon", 16)
        self.batch_size = config.get_typed("model", "batch_size", 128)
        self.epochs = config.get_typed("model", "epochs", 32)
        self.early_stopping_patience = config.get_typed("model", "early_stopping_patience", 12)
        self.initial_learning_rate = config.get_typed("model", "initial_learning_rate", 5e-5)
        self.results_dir = Path(config.results_dir)
        self.model_dir = self.results_dir / "models"
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.seed = 42
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.model = None
        self._configure_environment()

    def _configure_environment(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.logger.info(f"Found {len(gpus)} GPU(s)" if gpus else "No GPU found, using CPU")
        except Exception as e:
            self.logger.warning(f"Error configuring GPU: {e}")

    def _get_callbacks(self, X_val, y_val, fwd_returns_val, feature_names=None):
        checkpoint_path = self.model_path  # Already defined with a default in __init__
        # Ensure the directory exists, create if not
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        lr_decay_factor = self.config.get_typed("model", "lr_decay_factor", 0.85)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: self.initial_learning_rate * (
                    lr_decay_factor ** ((epoch - 5) // 3)) if epoch >= 5 else self.initial_learning_rate
        )

        # Fetch growth callback settings from config
        growth_cb_config = self.config.get("model", "growth_metric_callback", {})

        growth_callback = OptimizedGrowthMetricCallback(
            X_val, y_val, fwd_returns_val,
            monthly_target=growth_cb_config.get('monthly_target', 0.08),
            threshold_pct=growth_cb_config.get('threshold_pct', 0.6),
            transaction_cost=growth_cb_config.get('transaction_cost', 0.001),
            drawdown_weight=growth_cb_config.get('drawdown_weight', 1.8),
            avg_return_weight=growth_cb_config.get('avg_return_weight', 2.0),
            consistency_weight=growth_cb_config.get('consistency_weight', 1.0),
            adaptive_threshold=growth_cb_config.get('adaptive_threshold', True),
            min_trades_penalty=growth_cb_config.get('min_trades_penalty', True)
        )

        return [
            growth_callback,
            tf.keras.callbacks.EarlyStopping(
                monitor='growth_score', patience=self.early_stopping_patience, mode='max',
                restore_best_weights=True, min_delta=0.0005, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor='growth_score', save_best_only=True, mode='max', verbose=0,
                # Reduced verbosity
                save_weights_only=False
            ),
            lr_scheduler
        ]

    def train_model(self, X_train, y_train, X_val, y_val, df_val=None, fwd_returns_val=None, class_weight=None):
        self.logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

        # Ensure inputs are robust numpy arrays and handle potential issues
        X_train = np.asarray(np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
        y_train = np.asarray(np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
        X_val = np.asarray(np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
        y_val = np.asarray(np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

        y_train = np.clip(y_train, -1.0, 1.0).reshape(-1, 1) if len(y_train.shape) == 1 else np.clip(y_train, -1.0, 1.0)
        y_val = np.clip(y_val, -1.0, 1.0).reshape(-1, 1) if len(y_val.shape) == 1 else np.clip(y_val, -1.0, 1.0)

        self.model = None
        tf.keras.backend.clear_session()

        input_shape = (self.sequence_length, X_train.shape[2])
        feature_names = df_val.columns.tolist() if hasattr(df_val, 'columns') else [f"feature_{i}" for i in
                                                                                    range(X_train.shape[2])]

        callbacks = self._get_callbacks(X_val, y_val, fwd_returns_val, feature_names)

        # Get data augmentation settings from config
        aug_config = self.config.get("model", "data_augmentation", {})
        train_seq = NoisySequence(X_train, y_train, self.batch_size,
                                  noise_level=aug_config.get('noise_level', 0.01),
                                  data_augmentation=aug_config.get('enabled', True))

        self.logger.info("Creating model with optimized parameters")
        hybrid_model = OptimizedHybridModel(self.config, input_shape, feature_names)
        self.model = hybrid_model.build_model()

        history = hybrid_model.train(train_seq, X_val, y_val, fwd_returns_val, callbacks=callbacks)

        # Save the best model found by ModelCheckpoint callback implicitly (restore_best_weights=True in EarlyStopping)
        hybrid_model.save_model(self.model_path)

        return self.model

    def predict(self, X, batch_size=None):
        if self.model:
            X = np.asarray(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
            # Use configured batch_size if None provided
            predict_batch_size = batch_size if batch_size else self.batch_size
            return self.model.predict(X, batch_size=predict_batch_size, verbose=0)
        self.logger.error("No model available for prediction")
        return np.zeros((len(X), 1))  # Ensure consistent return shape

    def load_model(self, model_path=None):
        path = model_path or self.model_path
        hybrid_model = OptimizedHybridModel(self.config,
                                            (self.sequence_length, 43))  # Example input dim, adjust if known better
        if hybrid_model.load_model(path):
            self.model = hybrid_model.model
            return self.model
        return None
