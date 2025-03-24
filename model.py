import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout, Lambda
from tensorflow.keras.layers import Add, Activation, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate, TimeDistributed, Multiply, Attention, Bidirectional, GRU, Layer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import pickle
import json
from pathlib import Path
from datetime import datetime


class SoftmaxLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.softmax(inputs, axis=self.axis)


class OptimizedGrowthMetricCallback(tf.keras.callbacks.Callback):
    def __init__(
            self,
            X_val,
            y_val,
            fwd_returns_val,
            monthly_target=0.08,
            threshold_pct=0.4,
            model_idx=None,
            transaction_cost=0.001,
            drawdown_weight=1.2,
            avg_return_weight=1.0,
            consistency_weight=0.8,
            fixed_threshold=None
    ):
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
        self.trade_periods_per_month = 1440
        self.best_metrics = {
            'growth_score': -np.inf,
            'monthly_growth': 0.0,
            'val_sharpe': 0.0,
            'val_sortino': 0.0,
            'val_calmar': 0.0,
            'max_drawdown': 1.0,
            'trades_per_month': 0
        }
        self.historical_returns = []
        self.threshold_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.X_val) == 0:
            self._update_logs_with_defaults(logs)
            return

        try:
            batch_size = min(len(self.X_val), 256)
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=batch_size)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

            if self.fixed_threshold is not None:
                threshold = self.fixed_threshold
            else:
                sorted_preds = np.sort(np.abs(y_pred).flatten())
                threshold_idx = max(1, int(len(sorted_preds) * (1 - self.threshold_pct)))
                threshold = sorted_preds[threshold_idx] if len(sorted_preds) > threshold_idx else 0.0007
                threshold = max(threshold, 0.0004)

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

    def _update_logs_with_defaults(self, logs):
        logs['growth_score'] = 0.0
        logs['monthly_growth'] = 0.0
        logs['val_sharpe'] = 0.0
        logs['val_sortino'] = 0.0
        logs['val_calmar'] = 0.0
        logs['max_drawdown'] = 1.0
        logs['val_win_rate'] = 0.0
        logs['val_profit_factor'] = 0.0
        logs['val_trades_per_month'] = 0
        logs['consistency_score'] = 0.0

        if self.model_idx is not None:
            logs[f'growth_score_model_{self.model_idx}'] = 0.0

    def _calculate_enhanced_metrics(self, trade_indices, y_pred, logs, epoch):
        trade_returns = []
        win_count = 0
        profit_sum = 0
        loss_sum = 0

        equity_curve = [1.0]
        trade_timestamps = []

        for idx in trade_indices:
            if idx >= len(self.fwd_returns_val):
                continue

            actual_return = float(self.fwd_returns_val[idx])
            pred_direction = np.sign(y_pred[idx].flatten()[0])
            trade_return = float(pred_direction * actual_return) - self.transaction_cost
            trade_returns.append(trade_return)

            if trade_return > 0:
                win_count += 1
                profit_sum += trade_return
            else:
                loss_sum += abs(trade_return)

            equity_curve.append(equity_curve[-1] * (1.0 + trade_return))
            trade_timestamps.append(idx)

        n_trades = len(trade_returns)
        if n_trades == 0:
            self._update_logs_with_defaults(logs)
            return

        avg_return = sum(trade_returns) / n_trades
        win_rate = win_count / n_trades
        profit_factor = profit_sum / max(loss_sum, 1e-10)

        daily_equity = [equity_curve[0]]
        periods_per_day = 48

        for i in range(1, (len(equity_curve) // periods_per_day) + 1):
            day_end = min(i * periods_per_day, len(equity_curve) - 1)
            daily_equity.append(equity_curve[day_end])

        if len(equity_curve) % periods_per_day != 0:
            daily_equity.append(equity_curve[-1])

        daily_returns = []
        for i in range(1, len(daily_equity)):
            daily_returns.append(np.log(daily_equity[i] / daily_equity[i - 1]))

        if len(daily_returns) > 1:
            returns_mean = np.mean(daily_returns)
            returns_std = max(np.std(daily_returns), 1e-10)
            down_returns = np.array([r for r in daily_returns if r < 0])
            downside_std = np.std(down_returns) if len(down_returns) > 1 else returns_std

            sharpe_ratio = returns_mean / returns_std * np.sqrt(365)
            sortino_ratio = returns_mean / max(downside_std, 1e-10) * np.sqrt(365)
        else:
            returns_std = max(np.std(trade_returns), 1e-10)
            down_returns = np.array([r for r in trade_returns if r < 0])
            downside_std = np.std(down_returns) if len(down_returns) > 1 else returns_std

            trades_per_year = (n_trades / len(self.X_val)) * self.trade_periods_per_month * 12
            sharpe_ratio = avg_return / returns_std * np.sqrt(trades_per_year)
            sortino_ratio = avg_return / max(downside_std, 1e-10) * np.sqrt(trades_per_year)

        drawdowns = []
        peak = 1.0
        for value in equity_curve:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)

        max_drawdown = max(drawdowns)
        calmar_ratio = ((equity_curve[-1] / equity_curve[0]) - 1) / max(max_drawdown, 0.01)

        consecutive_losses = 0
        max_consecutive_losses = 0
        consecutive_wins = 0
        max_consecutive_wins = 0

        for i in range(1, len(equity_curve)):
            if equity_curve[i] > equity_curve[i - 1]:
                consecutive_wins += 1
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                consecutive_wins = 0

            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)

        trades_per_period = n_trades / len(self.X_val)
        trades_per_month = trades_per_period * self.trade_periods_per_month

        growth_factor = (1.0 + avg_return) ** trades_per_month
        monthly_growth = growth_factor - 1.0

        returns_sequence = np.array(trade_returns)
        rolling_returns = []
        window_size = min(10, len(returns_sequence))

        for i in range(len(returns_sequence) - window_size + 1):
            rolling_returns.append(np.mean(returns_sequence[i:i + window_size]))

        consistency_score = 1.0 / (np.std(rolling_returns) + 1e-10)

        recovery_periods = []
        in_drawdown = False
        drawdown_start = 0
        drawdown_threshold = 0.03
        recovery_threshold = 0.01

        for i, dd in enumerate(drawdowns):
            if not in_drawdown and dd > drawdown_threshold:
                in_drawdown = True
                drawdown_start = i
            elif in_drawdown and dd < recovery_threshold:
                in_drawdown = False
                recovery_periods.append(i - drawdown_start)

        avg_recovery = np.mean(recovery_periods) if recovery_periods else n_trades
        recovery_efficiency = n_trades / max(avg_recovery, 1.0)

        growth_distance = min(2.0, monthly_growth / self.monthly_target)

        dd_penalty_factor = 6.0
        dd_penalty = 1.0 - min(1.0, max_drawdown * dd_penalty_factor)

        self.drawdown_weight = 1.2

        growth_score = (
                               self.avg_return_weight * growth_distance +
                               self.drawdown_weight * dd_penalty +
                               self.consistency_weight * min(3.0, consistency_score)
                       ) / (self.avg_return_weight + self.drawdown_weight + self.consistency_weight)

        growth_score *= (0.4 + 0.6 * min(1.0, profit_factor / 2.0))

        consecutive_loss_penalty = min(0.3, max_consecutive_losses * 0.05)
        growth_score *= (1.0 - consecutive_loss_penalty)

        growth_score *= (0.65 + 0.35 * min(1.0, recovery_efficiency))

        if trades_per_month < 20:
            growth_score *= 0.75
        if max_drawdown > 0.20:
            growth_score *= 0.5
        elif max_drawdown > 0.15:
            growth_score *= 0.75

        logs['growth_score'] = float(growth_score)
        logs['monthly_growth'] = float(monthly_growth)
        logs['val_sharpe'] = float(sharpe_ratio)
        logs['val_sortino'] = float(sortino_ratio)
        logs['val_calmar'] = float(calmar_ratio)
        logs['max_drawdown'] = float(max_drawdown)
        logs['val_win_rate'] = float(win_rate)
        logs['val_profit_factor'] = float(profit_factor)
        logs['val_trades_per_month'] = float(trades_per_month)
        logs['consistency_score'] = float(consistency_score)
        logs['recovery_efficiency'] = float(recovery_efficiency)
        logs['max_consecutive_losses'] = float(max_consecutive_losses)

        if self.model_idx is not None:
            logs[f'growth_score_model_{self.model_idx}'] = float(growth_score)

        self.historical_returns.append({
            'epoch': epoch,
            'growth_score': growth_score,
            'monthly_growth': monthly_growth,
            'max_drawdown': max_drawdown,
            'trades_per_month': trades_per_month,
            'sharpe': sharpe_ratio,
            'sortino': sortino_ratio,
            'threshold': self.threshold_history[-1] if self.threshold_history else 0.0,
            'consecutive_losses': max_consecutive_losses
        })

        if growth_score > self.best_metrics['growth_score']:
            self.best_metrics.update({
                'growth_score': growth_score,
                'monthly_growth': monthly_growth,
                'val_sharpe': sharpe_ratio,
                'val_sortino': sortino_ratio,
                'val_calmar': calmar_ratio,
                'max_drawdown': max_drawdown,
                'trades_per_month': trades_per_month,
                'epoch': epoch,
                'max_consecutive_losses': max_consecutive_losses
            })

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
    def __init__(self, X, y, batch_size, noise_level=0.015, data_augmentation=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.std_dev = np.std(X, axis=(0, 1))
        self.data_augmentation = data_augmentation
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]

        noise = np.random.normal(0, self.noise_level * self.std_dev, batch_X.shape)
        batch_X = batch_X + noise

        if self.data_augmentation:
            for i in range(len(batch_X)):
                if np.random.random() < 0.4:
                    start = np.random.randint(0, batch_X.shape[1] // 4)
                    batch_X[i] = np.roll(batch_X[i], shift=start, axis=0)

            mask_prob = 0.25
            mask = np.random.random(batch_X.shape) > mask_prob
            batch_X = batch_X * mask

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
        self.training_metrics = []

        # Best parameters from optimization
        self.best_params = {
            "projection_size": 96,
            "transformer_heads": 8,
            "transformer_dropout": 0.31,
            "recurrent_units": 60,
            "recurrent_dropout": 0.11,
            "dropout": 0.2,
            "dense_units1": 68,
            "dense_units2": 18,
            "l2_lambda": 3.058656666978529e-05,
            "learning_rate": 5.4120091907504824e-05,
            "epochs": 15
        }

    def build_model(self):
        inputs = Input(shape=self.input_shape, dtype=tf.float32, name="hybrid_input")
        x = BatchNormalization(momentum=0.99, epsilon=1e-5)(inputs)

        # First stage: transformer with optimized parameters
        projection_size = self.best_params["projection_size"]
        transformer_heads = self.best_params["transformer_heads"]
        transformer_dropout = self.best_params["transformer_dropout"]

        x = Dense(projection_size, activation='linear')(x)

        # Apply position encoding
        pos_encoding = self._positional_encoding(self.input_shape[0], projection_size)
        x = Lambda(lambda x: x + tf.cast(pos_encoding, x.dtype))(x)

        # Transformer encoder layers
        for i in range(2):
            x = self._transformer_encoder_layer(x, units=projection_size,
                                                num_heads=transformer_heads,
                                                dropout=transformer_dropout,
                                                name=f"hybrid_transformer_{i}")

        # Second stage: GRU
        recurrent_units = self.best_params["recurrent_units"]
        recurrent_dropout = self.best_params["recurrent_dropout"]
        dropout_rate = self.best_params["dropout"]

        recurrent_out = GRU(recurrent_units, return_sequences=True,
                            recurrent_dropout=recurrent_dropout)(x)
        recurrent_out = BatchNormalization()(recurrent_out)
        recurrent_out = Dropout(dropout_rate)(recurrent_out)
        x = recurrent_out

        # Final attention mechanism
        attention_score = Dense(1, activation='tanh')(x)
        attention_weights = SoftmaxLayer(axis=1)(attention_score)
        context_vector = Multiply()([x, attention_weights])
        x = GlobalAveragePooling1D()(context_vector)

        # Final dense layers
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
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss=self._direction_enhanced_mse, metrics=['mae'])

        self.model = model
        return model

    def _positional_encoding(self, max_len, d_model):
        angle_rads = self._get_angles(np.arange(max_len)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _transformer_encoder_layer(self, inputs, units, num_heads, dropout, name):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads, dropout=dropout,
                                                     name=f"{name}_attention")(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(inputs + attention_output)
        ffn_output = Dense(units * 2, activation='swish', name=f"{name}_ffn1")(attention_output)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(units, name=f"{name}_ffn2")(ffn_output)
        return LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(attention_output + ffn_output)

    def _direction_enhanced_mse(self, y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        direction_true = tf.sign(y_true)
        direction_pred = tf.sign(y_pred)
        direction_match = tf.cast(tf.equal(direction_true, direction_pred), tf.float32)
        direction_loss = 1.0 - direction_match

        gamma = 2.0
        focal_weight = tf.pow(1.0 - direction_match, gamma)
        focal_direction_loss = focal_weight * direction_loss

        return mse_loss + (self.config.get("model", "direction_loss_weight", 0.65) * focal_direction_loss)

    def train(self, train_seq, X_val, y_val, fwd_returns_val, callbacks=None):
        if self.model is None:
            self.build_model()

        epochs = self.best_params["epochs"]
        default_callbacks = [
            OptimizedGrowthMetricCallback(X_val, y_val, fwd_returns_val),
            tf.keras.callbacks.EarlyStopping(
                monitor='growth_score',
                patience=5,
                mode='max',
                restore_best_weights=True
            )
        ]

        used_callbacks = callbacks if callbacks else default_callbacks

        history = self.model.fit(
            train_seq,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=used_callbacks,
            verbose=1
        )

        monthly_growth = max(history.history.get('monthly_growth', [0]))
        val_win_rate = max(history.history.get('val_win_rate', [0]))
        val_profit_factor = max(history.history.get('val_profit_factor', [0]))
        growth_score = max(history.history.get('growth_score', [0]))

        self.training_metrics = {
            'monthly_growth': monthly_growth,
            'val_win_rate': val_win_rate,
            'val_profit_factor': val_profit_factor,
            'growth_score': growth_score
        }

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
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)

    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, compile=True)
            self.logger.info(f"Loaded model from {path}")

            params_path = f"{os.path.splitext(path)[0]}_params.json"
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.best_params = json.load(f)

            return True
        return False


class TradingModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("OptimizedTradingModel")
        self.model_path = config.get("model", "model_path")
        self.sequence_length = config.get("model", "sequence_length", 72)
        self.horizon = config.get("model", "horizon", 16)
        self.batch_size = config.get("model", "batch_size", 256)
        self.epochs = config.get("model", "epochs", 20)
        self.early_stopping_patience = config.get("model", "early_stopping_patience", 5)
        self.initial_learning_rate = config.get("model", "initial_learning_rate", 3e-4)
        self.direction_loss_weight = config.get("model", "direction_loss_weight", 0.65)
        self.results_dir = Path(config.results_dir)
        self.model_dir = self.results_dir / "models"
        self.model_dir.mkdir(exist_ok=True, parents=True)

        if config.get("model", "use_mixed_precision", True):
            try:
                from tensorflow.keras.mixed_precision import set_global_policy
                set_global_policy('mixed_float16')
            except:
                self.logger.warning("Could not set mixed precision policy")

        self._configure_environment()
        self.seed = 42
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.model = None

    def _configure_environment(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.logger.info(f"Found {len(gpus)} GPU(s)" if gpus else "No GPU found, using CPU")

            try:
                import psutil
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
                self.logger.info(f"Available system memory: {available_memory:.2f} GB")
                if available_memory < 4:
                    self.batch_size = min(self.batch_size, 64)
                    self.logger.info(f"Limited memory detected, reducing batch size to {self.batch_size}")
            except:
                pass
        except Exception as e:
            self.logger.warning(f"Error configuring GPU: {e}")

    def _calculate_optimal_batch_size(self, X_train):
        try:
            import psutil
            data_size_mb = X_train.nbytes / (1024 * 1024)
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            memory_limit = available_memory * 0.2
            samples_per_batch = max(16, int(memory_limit / (data_size_mb / len(X_train)) / 4))
            power_of_2 = 2 ** int(np.log2(samples_per_batch))
            return max(16, min(power_of_2, self.batch_size))
        except:
            return self.batch_size

    def _get_callbacks(self, X_val, y_val, fwd_returns_val, feature_names=None):
        checkpoint_path = self.model_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        callbacks = [
            OptimizedGrowthMetricCallback(X_val, y_val, fwd_returns_val),
            tf.keras.callbacks.EarlyStopping(
                monitor='growth_score',
                patience=self.early_stopping_patience,
                mode='max',
                restore_best_weights=True,
                min_delta=0.0001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='growth_score',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            )
        ]

        class GradientNoiseCallback(tf.keras.callbacks.Callback):
            def __init__(self, start_epoch=5, noise_stddev=1e-4, decay_rate=0.55):
                super().__init__()
                self.start_epoch = start_epoch
                self.initial_stddev = noise_stddev
                self.decay_rate = decay_rate
                self.noise_stddev = noise_stddev

            def on_epoch_begin(self, epoch, logs=None):
                if epoch >= self.start_epoch:
                    self.noise_stddev = self.initial_stddev * (
                    (1.0 / (1.0 + self.decay_rate * (epoch - self.start_epoch))))

            def on_batch_end(self, batch, logs=None):
                if hasattr(self.model.optimizer, 'get_weights') and hasattr(self.model.optimizer, 'set_weights'):
                    weights = self.model.optimizer.get_weights()
                    for i in range(len(weights)):
                        noise = np.random.normal(0, self.noise_stddev, weights[i].shape)
                        weights[i] = weights[i] + noise
                    self.model.optimizer.set_weights(weights)

        callbacks.append(GradientNoiseCallback(start_epoch=3, noise_stddev=1e-4, decay_rate=0.6))

        initial_lr = self.initial_learning_rate

        def lr_schedule(epoch, lr):
            if epoch < 5:
                return initial_lr * (epoch + 1) / 5.0
            return initial_lr * (0.4 ** ((epoch - 5) // 4))

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

        return callbacks

    def train_model(self, X_train, y_train, X_val, y_val, df_val=None, fwd_returns_val=None, class_weight=None):
        self.logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.clip(y_train, -1.0, 1.0).reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        y_val = np.clip(y_val, -1.0, 1.0).reshape(-1, 1) if len(y_val.shape) == 1 else y_val

        input_shape = (X_train.shape[1], X_train.shape[2])
        feature_names = df_val.columns.tolist() if hasattr(df_val, 'columns') else None
        batch_size = self._calculate_optimal_batch_size(X_train)
        self.logger.info(f"Using batch size: {batch_size}")

        callbacks = self._get_callbacks(X_val, y_val, fwd_returns_val, feature_names)
        train_seq = NoisySequence(X_train, y_train, batch_size, noise_level=0.015, data_augmentation=True)

        self.logger.info("Creating model with optimized parameters")
        hybrid_model = OptimizedHybridModel(self.config, input_shape, feature_names)
        self.model = hybrid_model.build_model()

        hybrid_model.train(
            train_seq, X_val, y_val, fwd_returns_val,
            callbacks=callbacks
        )

        model_path = self.model_path
        hybrid_model.save_model(model_path)

        return self.model

    def predict(self, X, batch_size=None):
        if self.model:
            return self.model.predict(X, verbose=0)
        self.logger.error("No model available for prediction")
        return np.zeros((len(X), 1))

    def load_model(self, model_path=None):
        path = model_path or self.model_path
        hybrid_model = OptimizedHybridModel(self.config, (self.sequence_length, 50))
        if hybrid_model.load_model(path):
            self.model = hybrid_model.model
            return self.model
        return None