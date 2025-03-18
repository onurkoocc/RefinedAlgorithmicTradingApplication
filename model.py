import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Conv1D, Dropout, Lambda
from tensorflow.keras.layers import Add, Activation, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate, TimeDistributed, Multiply, Attention, Bidirectional, GRU, Layer
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l2
import pickle
import json
from pathlib import Path
import platform
import psutil
import math


class SoftmaxLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.softmax(inputs, axis=self.axis)


class GradientNoiseCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch=5, noise_stddev=1e-4, decay_rate=0.55):
        super().__init__()
        self.start_epoch = start_epoch
        self.initial_stddev = noise_stddev
        self.decay_rate = decay_rate
        self.noise_stddev = noise_stddev

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            self.noise_stddev = self.initial_stddev * ((1.0 / (1.0 + self.decay_rate * (epoch - self.start_epoch))))

    def on_batch_end(self, batch, logs=None):
        if hasattr(self.model.optimizer, 'get_weights') and hasattr(self.model.optimizer, 'set_weights'):
            weights = self.model.optimizer.get_weights()
            for i in range(len(weights)):
                noise = np.random.normal(0, self.noise_stddev, weights[i].shape)
                weights[i] = weights[i] + noise
            self.model.optimizer.set_weights(weights)


class TradeMetricCallback(Callback):
    def __init__(self, X_val, y_val, fwd_returns_val, threshold_pct=0.4):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.fwd_returns_val = fwd_returns_val
        self.threshold_pct = threshold_pct
        self.best_metrics = {
            'val_avg_return': -np.inf,
            'val_win_rate': 0.0,
            'val_profit_factor': 0.0,
            'trades': 0
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if len(self.X_val) == 0:
            self._update_logs_with_defaults(logs)
            return

        try:
            batch_size = min(len(self.X_val), 256)
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=batch_size)

            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

            sorted_preds = np.sort(np.abs(y_pred).flatten())
            if len(sorted_preds) > 0:
                threshold_idx = max(1, int(len(sorted_preds) * (1 - self.threshold_pct)))
                threshold = sorted_preds[threshold_idx]
                threshold = max(threshold, 0.0004)
            else:
                threshold = 0.0007

            trade_indices = np.where(np.abs(y_pred) > threshold)[0]

            if len(trade_indices) == 0:
                self._update_logs_with_defaults(logs)
                return

            self._calculate_trade_metrics(trade_indices, y_pred, logs, epoch)

        except Exception as e:
            print(f"Error in TradeMetricCallback: {e}")
            import traceback
            print(traceback.format_exc())
            self._update_logs_with_defaults(logs)

    def _update_logs_with_defaults(self, logs):
        logs['val_avg_return'] = 0.0
        logs['val_win_rate'] = 0.0
        logs['val_profit_factor'] = 0.0
        logs['val_sharpe'] = 0.0
        logs['val_trades'] = 0

    def _calculate_trade_metrics(self, trade_indices, y_pred, logs, epoch):
        trade_returns = []
        win_count = 0
        profit_sum = 0
        loss_sum = 0

        for idx in trade_indices:
            if idx >= len(self.fwd_returns_val):
                continue

            actual_return = float(self.fwd_returns_val[idx])
            pred_direction = np.sign(y_pred[idx][0]) if y_pred[idx].size > 1 else np.sign(y_pred[idx])

            trade_return = float(pred_direction * actual_return)
            trade_returns.append(trade_return)

            if trade_return > 0:
                win_count += 1
                profit_sum += trade_return
            else:
                loss_sum += abs(trade_return)

        n_trades = len(trade_returns)
        if n_trades > 0:
            avg_return = sum(trade_returns) / n_trades
            win_rate = win_count / n_trades
            profit_factor = profit_sum / max(loss_sum, 1e-10)
        else:
            avg_return, win_rate, profit_factor = 0.0, 0.0, 0.0

        if len(trade_returns) > 1:
            returns_std = max(np.std(trade_returns), 1e-10)
        else:
            returns_std = 1.0

        sharpe_ratio = (avg_return / returns_std) * np.sqrt(365 * 24)

        logs['val_avg_return'] = float(avg_return)
        logs['val_win_rate'] = float(win_rate)
        logs['val_profit_factor'] = float(profit_factor)
        logs['val_sharpe'] = float(sharpe_ratio)
        logs['val_trades'] = n_trades

        if avg_return > self.best_metrics['val_avg_return']:
            self.best_metrics = {
                'val_avg_return': avg_return,
                'val_win_rate': win_rate,
                'val_profit_factor': profit_factor,
                'trades': n_trades,
                'epoch': epoch
            }

        if epoch % 5 == 0 or epoch == 0:
            print(f"\nTrading Metrics - Trades: {n_trades}, "
                  f"Win Rate: {win_rate:.2f}, Avg Return: {float(avg_return):.4f}, "
                  f"Profit Factor: {float(profit_factor):.2f}, Sharpe: {float(sharpe_ratio):.2f}")


class FeatureImportanceCallback(Callback):
    def __init__(self, X_train, feature_names=None):
        super().__init__()
        self.X_train = X_train
        self.feature_names = feature_names
        self.importance_scores = None

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 != 0 or epoch == 0:
            return

        try:
            baseline_pred = self.model.predict(self.X_train[:100], verbose=0)

            importance = []

            n_features = self.X_train.shape[2]
            check_features = min(n_features, 10)

            for i in range(check_features):
                X_permuted = self.X_train[:100].copy()
                feature_vals = X_permuted[:, :, i].flatten()
                np.random.shuffle(feature_vals)
                X_permuted[:, :, i] = feature_vals.reshape(X_permuted[:, :, i].shape)

                permuted_pred = self.model.predict(X_permuted, verbose=0)

                importance.append(np.mean((baseline_pred - permuted_pred) ** 2))

            self.importance_scores = np.array(importance)
            if np.sum(self.importance_scores) > 0:
                self.importance_scores = self.importance_scores / np.sum(self.importance_scores)

            if self.feature_names and len(self.feature_names) == n_features:
                top_idx = np.argsort(self.importance_scores)[-3:][::-1]
                feature_info = ", ".join([f"{self.feature_names[i]}: {self.importance_scores[i]:.4f}" for i in top_idx])
                print(f"Top features: {feature_info}")

        except Exception as e:
            print(f"Error calculating feature importance: {e}")


class EnsembleModel:
    def __init__(self, config, input_shape, feature_names=None):
        self.config = config
        self.input_shape = input_shape
        self.feature_names = feature_names
        self.models = []
        self.weights = []
        self.training_metrics = []
        self.logger = logging.getLogger("EnsembleModel")

    def build_ensemble(self, num_models=3):
        self.models = []

        # Create diverse model architectures
        transformer_model = self._build_transformer_model()
        lstm_cnn_model = self._build_lstm_cnn_model()

        self.models = [transformer_model, lstm_cnn_model]
        self.weights = [0.5, 0.5]

        return self.models

    def _build_transformer_model(self):
        inputs = Input(shape=self.input_shape, dtype=tf.float32, name="transformer_input")

        # Initial normalization
        x = BatchNormalization(momentum=0.99, epsilon=1e-5)(inputs)

        # Project input to d_model=128 to match transformer encoder units
        x = Dense(128, activation='relu')(x)

        # Generate positional encoding for d_model=128
        pos_encoding = self._positional_encoding(self.input_shape[0], 128)
        pos_encoding_layer = tf.keras.layers.Lambda(
            lambda x: x + tf.cast(pos_encoding, x.dtype),
            output_shape=lambda input_shape: input_shape,
            name="pos_encoding"
        )
        x = pos_encoding_layer(x)

        # Transformer encoder blocks with consistent units=128
        for i in range(3):
            x = self._transformer_encoder_layer(x, units=128, num_heads=8, dropout=0.1, name=f"transformer_{i}")

        # Global context extraction
        x = GlobalAveragePooling1D()(x)

        # Output projection
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        outputs = Dense(1, activation='tanh', kernel_regularizer=l2(1e-6), dtype=tf.float32)(x)

        model = Model(inputs, outputs, name="transformer_model")

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        model.compile(
            optimizer=optimizer,
            loss=self._direction_enhanced_mse,
            metrics=['mae']
        )

        return model

    def _build_lstm_cnn_model(self):
        inputs = Input(shape=self.input_shape, dtype=tf.float32, name="lstm_cnn_input")

        # Initial normalization
        x = BatchNormalization(momentum=0.99, epsilon=1e-5)(inputs)

        # CNN feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        conv2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)

        # Combine CNN features
        x = Concatenate()([conv1, conv2])
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        # LSTM processing
        x = LSTM(128, return_sequences=True, recurrent_dropout=0)(x)
        x = BatchNormalization()(x)

        # Self-attention layer
        attention_score = Dense(1, activation='tanh')(x)
        attention_weights = SoftmaxLayer(axis=1)(attention_score)
        x = Multiply()([x, attention_weights])

        # Global context extraction
        x = GlobalAveragePooling1D()(x)

        # Output projection
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        outputs = Dense(1, activation='tanh', kernel_regularizer=l2(1e-6), dtype=tf.float32)(x)

        model = Model(inputs, outputs, name="lstm_cnn_model")

        optimizer = tf.keras.optimizers.Adam(learning_rate=4e-4)

        model.compile(
            optimizer=optimizer,
            loss=self._direction_enhanced_mse,
            metrics=['mae']
        )

        return model

    def _build_tft_model(self):
        inputs = Input(shape=self.input_shape, dtype=tf.float32, name="tft_input")

        # Initial normalization
        x = BatchNormalization(momentum=0.99, epsilon=1e-5)(inputs)

        # Gated residual network for feature processing
        x = self._gated_residual_network(x, 128)

        # Variable selection
        x = self._variable_selection_network(x, self.input_shape[1])

        # Temporal processing with attention
        for i in range(2):
            x = self._temporal_self_attention_layer(x, 64, num_heads=4, dropout=0.1, name=f"tft_attn_{i}")

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Output projection
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        outputs = Dense(1, activation='tanh', kernel_regularizer=l2(1e-6), dtype=tf.float32)(x)

        model = Model(inputs, outputs, name="tft_model")

        optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-4)

        model.compile(
            optimizer=optimizer,
            loss=self._direction_enhanced_mse,
            metrics=['mae']
        )

        return model

    def _gated_residual_network(self, x, units):
        skip_connection = x

        # Ensure dimensions match for residual connection
        if x.shape[-1] != units:
            skip_connection = Dense(units)(skip_connection)

        # Main path
        a = Dense(units)(x)
        a = LayerNormalization()(a)
        a = Activation('elu')(a)

        a = Dense(units)(a)
        a = LayerNormalization()(a)
        a = Activation('elu')(a)

        # Gate
        gate = Dense(units, activation='sigmoid')(x)

        # Apply gate and add residual connection
        x = Add()([gate * a, skip_connection])

        return x

    def _variable_selection_network(self, x, num_features):
        # Feature-wise dense layers
        processed_features = []

        for i in range(num_features):
            # Extract single feature across all time steps
            feature = Lambda(lambda x: x[:, :, i:i + 1])(x)

            # Process through dense layer
            processed = Dense(16, activation='elu')(feature)
            processed = Dense(16, activation='elu')(processed)

            processed_features.append(processed)

        # Concatenate processed features
        processed_x = Concatenate(axis=-1)(processed_features)

        # Attention weights for feature selection
        attention_input = GlobalAveragePooling1D()(x)
        attention_weights = Dense(num_features, activation='softmax')(attention_input)

        # Apply attention weights to features
        weighted_features = []
        for i in range(num_features):
            # Extract weight for this feature
            weight = Lambda(lambda x: x[:, i:i + 1])(attention_weights)

            # Extract processed feature
            feature = Lambda(lambda x: x[:, :, i:i + 1])(processed_x)

            # Weight the feature
            weighted = Multiply()([feature, weight])
            weighted_features.append(weighted)

        # Combine weighted features
        x = Add()(weighted_features)

        return x

    def _temporal_self_attention_layer(self, x, units, num_heads, dropout, name):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units // num_heads, dropout=dropout,
            name=f"{name}_attention"
        )(x, x)

        # Add & norm
        attention_output = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(x + attention_output)

        # Feed-forward network
        ffn_output = Dense(units * 2, activation='elu', name=f"{name}_ffn1")(attention_output)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(units, name=f"{name}_ffn2")(ffn_output)

        # Add & norm
        return LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(attention_output + ffn_output)

    def _positional_encoding(self, max_len, d_model):
        angle_rads = self._get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _transformer_encoder_layer(self, inputs, units, num_heads, dropout, name):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units // num_heads, dropout=dropout,
            name=f"{name}_attention"
        )(inputs, inputs)

        # Add & norm
        attention_output = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(inputs + attention_output)

        # Feed-forward network
        ffn_output = Dense(units * 2, activation='relu', name=f"{name}_ffn1")(attention_output)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(units, name=f"{name}_ffn2")(ffn_output)

        # Add & norm
        return LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(attention_output + ffn_output)

    def _direction_enhanced_mse(self, y_true, y_pred):
        # Basic MSE component
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Direction component
        direction_true = tf.sign(y_true)
        direction_pred = tf.sign(y_pred)
        direction_match = tf.cast(tf.equal(direction_true, direction_pred), tf.float32)
        direction_loss = 1.0 - direction_match

        # Combined loss with reasonable weighting
        combined_loss = mse_loss + (0.2 * direction_loss)

        return combined_loss

    def train_models(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=256, callbacks=None):
        self.training_metrics = []

        for i, model in enumerate(self.models):
            model_name = model.name
            self.logger.info(f"Training {model_name} ({i + 1}/{len(self.models)})")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Store training metrics
            val_avg_return = max(history.history.get('val_avg_return', [0]))
            val_win_rate = max(history.history.get('val_win_rate', [0]))
            val_profit_factor = max(history.history.get('val_profit_factor', [0]))

            self.training_metrics.append({
                'model_name': model_name,
                'val_avg_return': val_avg_return,
                'val_win_rate': val_win_rate,
                'val_profit_factor': val_profit_factor
            })

            # Update weights based on validation performance
            self._update_ensemble_weights()

            tf.keras.backend.clear_session()

        return self.models

    def _update_ensemble_weights(self):
        if not self.training_metrics:
            return

        # Extract performance metrics
        avg_returns = [m.get('val_avg_return', 0) for m in self.training_metrics]
        win_rates = [m.get('val_win_rate', 0) for m in self.training_metrics]
        profit_factors = [m.get('val_profit_factor', 0) for m in self.training_metrics]

        # Convert to numpy arrays and handle non-positive values
        avg_returns = np.array(avg_returns)
        avg_returns = np.where(avg_returns <= 0, 1e-6, avg_returns)

        win_rates = np.array(win_rates)
        win_rates = np.where(win_rates <= 0, 1e-6, win_rates)

        profit_factors = np.array(profit_factors)
        profit_factors = np.where(profit_factors <= 0, 1e-6, profit_factors)

        # Normalize to get weights
        combined_score = (avg_returns * 0.5) + (win_rates * 0.3) + (profit_factors * 0.2)
        self.weights = combined_score / np.sum(combined_score)

        self.logger.info(f"Updated ensemble weights: {self.weights}")

    def predict(self, X):
        if not self.models:
            raise ValueError("No models in ensemble. Call build_ensemble() first.")

        all_predictions = []

        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X, verbose=0)
                all_predictions.append(pred * self.weights[i])
            except Exception as e:
                self.logger.error(f"Error in model {i} prediction: {e}")
                # Return zeros if model fails
                all_predictions.append(np.zeros((len(X), 1)))

        # Combine predictions weighted by model performance
        combined_pred = np.zeros_like(all_predictions[0])
        for pred in all_predictions:
            combined_pred += pred

        return combined_pred

    def save_models(self, base_path):
        if not self.models:
            return

        for i, model in enumerate(self.models):
            model_path = f"{base_path}_ensemble_{i}.keras"
            model.save(model_path)
            self.logger.info(f"Saved model {i} to {model_path}")

        # Save ensemble weights
        weights_path = f"{base_path}_ensemble_weights.json"
        with open(weights_path, 'w') as f:
            json.dump({
                'weights': self.weights.tolist(),
                'metrics': self.training_metrics
            }, f)

    def load_models(self, base_path, num_models=3):
        self.models = []

        for i in range(num_models):
            model_path = f"{base_path}_ensemble_{i}.keras"
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path, compile=True)
                self.models.append(model)
                self.logger.info(f"Loaded model {i} from {model_path}")

        # Load ensemble weights
        weights_path = f"{base_path}_ensemble_weights.json"
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                data = json.load(f)
                self.weights = np.array(data.get('weights', []))
                self.training_metrics = data.get('metrics', [])
        else:
            # If weights not found, use equal weighting
            self.weights = np.ones(len(self.models)) / len(self.models)

        return len(self.models) > 0


class TradingModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TradingModel")

        self.model_path = config.get("model", "model_path")
        self.sequence_length = config.get("model", "sequence_length", 72)
        self.horizon = config.get("model", "horizon", 16)
        self.batch_size = config.get("model", "batch_size", 256)
        self.epochs = config.get("model", "epochs", 20)
        self.early_stopping_patience = config.get("model", "early_stopping_patience", 5)
        self.project_name = config.get("model", "project_name", "btc_trading")

        self.dropout_rate = config.get("model", "dropout_rate", 0.2)
        self.l2_reg = config.get("model", "l2_reg", 5e-6)

        self.attention_enabled = config.get("model", "attention_enabled", True)
        self.use_feature_importance = config.get("model", "use_feature_importance", True)
        self.cnn_filters = config.get("model", "cnn_filters", 64)
        self.lstm_units = config.get("model", "lstm_units", 96)

        self.use_lr_schedule = config.get("model", "use_lr_schedule", True)
        self.initial_learning_rate = config.get("model", "initial_learning_rate", 3e-4)

        self.use_ensemble = config.get("model", "use_ensemble", True)

        results_dir = Path(config.results_dir)
        self.model_dir = results_dir / "models"
        self.model_dir.mkdir(exist_ok=True, parents=True)

        self.importance_path = self.model_dir / "feature_importance.json"

        if config.get("model", "use_mixed_precision", True):
            set_global_policy('mixed_float16')

        self._configure_environment()

        self.seed = 42
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.model = None
        self.ensemble_models = None
        self.feature_importance = None

        self._load_feature_importance()

    def _configure_environment(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if gpus:
                self.logger.info(f"Found {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    self.logger.info(f"GPU {i}: {gpu.name}")
            else:
                self.logger.info("No GPU found, using CPU")

            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            self.logger.info(f"Available system memory: {available_memory:.2f} GB")

            if available_memory < 4:
                self.batch_size = min(self.batch_size, 64)
                self.logger.info(f"Limited memory detected, reducing batch size to {self.batch_size}")

        except Exception as e:
            self.logger.warning(f"Error configuring GPU: {e}")
            return False

    def _load_feature_importance(self):
        try:
            if self.importance_path.exists():
                with open(self.importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                self.logger.info(f"Loaded feature importance from {self.importance_path}")
        except Exception as e:
            self.logger.warning(f"Error loading feature importance: {e}")

    def _save_feature_importance(self, feature_importance, feature_names=None):
        try:
            if feature_importance is not None:
                importance_dict = {}

                if feature_names is not None and len(feature_names) == len(feature_importance):
                    for i, name in enumerate(feature_names):
                        importance_dict[name] = float(feature_importance[i])
                else:
                    for i, importance in enumerate(feature_importance):
                        importance_dict[f"feature_{i}"] = float(importance)

                with open(self.importance_path, 'w') as f:
                    json.dump(importance_dict, f, indent=2)

                self.logger.info(f"Saved feature importance to {self.importance_path}")
                self.feature_importance = importance_dict
        except Exception as e:
            self.logger.warning(f"Error saving feature importance: {e}")

    def build_model(self, input_shape, feature_names=None):
        inputs = Input(shape=input_shape, dtype=tf.float32, name="input_features")

        # Initial normalization
        x = BatchNormalization(momentum=0.99, epsilon=1e-5, name="initial_norm")(inputs)

        # Add positional encoding for transformer
        pos_encoding = self._positional_encoding(input_shape[0], input_shape[1])
        x = x + pos_encoding

        # Transformer encoder blocks
        for i in range(3):
            x = self._transformer_encoder_layer(x, units=128, num_heads=8, dropout=0.1, name=f"transformer_{i}")

        # Global pooling
        x = GlobalAveragePooling1D()(x)

        # Output projection layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        x = Dense(32, activation='relu', kernel_regularizer=l2(1e-6))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        # Final output with tanh activation for [-1, 1] range
        outputs = Dense(1, activation='tanh', kernel_regularizer=l2(1e-6), dtype='float32')(x)

        model = Model(inputs, outputs, name="transformer_trading_model")

        def direction_enhanced_mse(y_true, y_pred):
            # Basic MSE component
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

            # Direction component
            direction_true = tf.sign(y_true)
            direction_pred = tf.sign(y_pred)
            direction_match = tf.cast(tf.equal(direction_true, direction_pred), tf.float32)
            direction_loss = 1.0 - direction_match

            # Combined loss with reasonable weighting
            combined_loss = mse_loss + (0.2 * direction_loss)

            return combined_loss

        optimizer = Adam(learning_rate=self.initial_learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=direction_enhanced_mse,
            metrics=['mae']
        )

        return model

    def _positional_encoding(self, max_len, d_model):
        angle_rads = self._get_angles(
            np.arange(max_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _transformer_encoder_layer(self, inputs, units, num_heads, dropout, name):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units // num_heads, dropout=dropout,
            name=f"{name}_attention"
        )(inputs, inputs)

        # Add & norm
        attention_output = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(inputs + attention_output)

        # Feed-forward network
        ffn_output = Dense(units * 2, activation='relu', name=f"{name}_ffn1")(attention_output)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(units, name=f"{name}_ffn2")(ffn_output)

        # Add & norm
        return LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(attention_output + ffn_output)

    def train_model(self, X_train, y_train, X_val, y_val, df_val=None, fwd_returns_val=None, class_weight=None):
        self.logger.info(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
        self.logger.info(f"Validation data shape: {X_val.shape}, labels shape: {y_val.shape}")

        # Ensure clean data before training - prevent NaN issues
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply clipping to prevent extreme values
        y_train = np.clip(y_train, -1.0, 1.0)
        y_val = np.clip(y_val, -1.0, 1.0)

        y_train = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val

        input_shape = (X_train.shape[1], X_train.shape[2])

        feature_names = None
        if hasattr(df_val, 'columns'):
            feature_names = df_val.columns.tolist()

        # Use ensemble approach if configured
        if self.use_ensemble:
            self.logger.info("Building ensemble trading model architecture")

            # Create ensemble
            self.ensemble_models = EnsembleModel(self.config, input_shape, feature_names)
            models = self.ensemble_models.build_ensemble(num_models=2)

            # Train each model in the ensemble
            callbacks = self._get_callbacks(X_val, y_val, fwd_returns_val, feature_names)

            self.ensemble_models.train_models(
                X_train, y_train,
                X_val, y_val,
                epochs=self.epochs,
                batch_size=self._calculate_optimal_batch_size(X_train),
                callbacks=callbacks
            )

            # Save ensemble models
            self.ensemble_models.save_models(self.model_path)

            return models
        else:
            # Single model approach
            self.logger.info("Building trading model architecture")
            self.model = self.build_model(input_shape, feature_names)

            callbacks = self._get_callbacks(X_val, y_val, fwd_returns_val, feature_names)

            # Add gradient clipping to optimizer
            self.model.optimizer.clipnorm = 1.0

            # Add gradient noise for better local minima escape
            noise_callback = GradientNoiseCallback(
                start_epoch=5,
                noise_stddev=1e-5,
                decay_rate=0.55
            )
            callbacks.append(noise_callback)

            actual_batch_size = self._calculate_optimal_batch_size(X_train)
            self.logger.info(f"Using batch size: {actual_batch_size}")

            # Mixed precision is already set with set_global_policy in the class initialization
            self.logger.info(f"Starting training for {self.epochs} epochs")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=actual_batch_size,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )

            self.model.save(self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")

            feature_importance_callback = next((c for c in callbacks if isinstance(c, FeatureImportanceCallback)), None)
            if feature_importance_callback and feature_importance_callback.importance_scores is not None:
                self._save_feature_importance(
                    feature_importance_callback.importance_scores,
                    feature_names
                )

            tf.keras.backend.clear_session()

            return self.model

    def _calculate_optimal_batch_size(self, X_train):
        data_size_mb = X_train.nbytes / (1024 * 1024)

        try:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
        except:
            available_memory = 4000

        memory_limit = available_memory * 0.2

        samples_per_batch = max(16, int(memory_limit / (data_size_mb / len(X_train)) / 4))

        power_of_2 = 2 ** int(np.log2(samples_per_batch))
        optimal_batch_size = min(power_of_2, self.batch_size)

        return max(16, optimal_batch_size)

    def _get_callbacks(self, X_val, y_val, fwd_returns_val, feature_names=None):
        checkpoint_path = self.model_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        trade_callback = TradeMetricCallback(X_val, y_val, fwd_returns_val)

        importance_callback = FeatureImportanceCallback(X_val[:1000], feature_names)

        early_stopping = EarlyStopping(
            monitor='val_avg_return',
            patience=self.early_stopping_patience,
            mode='max',
            restore_best_weights=True,
            min_delta=0.0001,
            verbose=1
        )

        try:
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                monitor='val_avg_return',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            )
        except Exception as e:
            self.logger.warning(f"Error creating checkpoint callback: {e}")
            checkpoint = ModelCheckpoint(
                str(checkpoint_path) + ".weights.h5",
                monitor='val_avg_return',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=True
            )

        callbacks = [trade_callback, importance_callback, early_stopping, checkpoint]

        # Only add ReduceLROnPlateau if we're NOT using a learning rate schedule
        if not self.use_lr_schedule:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_avg_return',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                mode='max',
                verbose=1
            )
            callbacks.append(reduce_lr)

        return callbacks

    def predict(self, X, batch_size=None):
        # Check if we're using the ensemble
        if hasattr(self, 'ensemble_models') and self.ensemble_models is not None:
            try:
                return self.ensemble_models.predict(X)
            except Exception as e:
                self.logger.error(f"Error in ensemble prediction: {e}")
                # Fall back to single model if available
                if self.model is None:
                    self.load_model()

        # Single model prediction
        if self.model is None:
            self.load_model()
            if self.model is None:
                self.logger.error("No model available for prediction")
                return np.zeros((len(X), 1))

        if len(X) == 0:
            return np.array([])

        if np.isnan(X).any() or np.isinf(X).any():
            self.logger.warning("NaN or Inf values in input data, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if batch_size is None:
            data_size_mb = X.nbytes / (1024 * 1024)
            try:
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                batch_size = max(16, min(256, int((available_memory * 0.1) / (data_size_mb / len(X)))))
            except:
                batch_size = min(self.batch_size, 256)

        try:
            predictions = self.model.predict(X, verbose=0, batch_size=batch_size)

            if np.isnan(predictions).any() or np.isinf(predictions).any():
                self.logger.warning("NaN or Inf values in predictions, replacing with zeros")
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

            predictions = np.clip(predictions, -1.0, 1.0)

            return predictions

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return np.zeros((len(X), 1))

    def load_model(self, model_path=None):
        path = model_path or self.model_path

        # Check if we have ensemble models
        ensemble_base = str(path).rstrip(".keras")
        ensemble_weights_path = f"{ensemble_base}_ensemble_weights.json"

        if os.path.exists(ensemble_weights_path):
            self.logger.info("Loading ensemble models")

            # Initialize ensemble
            self.ensemble_models = EnsembleModel(self.config, (72, 50))  # Default shape, will be updated
            if self.ensemble_models.load_models(ensemble_base):
                return self.ensemble_models

        # Fall back to single model loading
        if not os.path.exists(path):
            self.logger.warning(f"Model not found at {path}")
            return None

        try:
            self.logger.info(f"Loading model from {path}")

            def direction_enhanced_mse(y_true, y_pred):
                mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
                direction_true = tf.sign(y_true)
                direction_pred = tf.sign(y_pred)
                direction_match = tf.cast(tf.equal(direction_true, direction_pred), tf.float32)
                direction_loss = 1.0 - direction_match
                return mse_loss + (0.2 * direction_loss)

            # Register custom layer for model loading
            custom_objects = {
                'direction_enhanced_mse': direction_enhanced_mse,
                'SoftmaxLayer': SoftmaxLayer
            }

            self.model = tf.keras.models.load_model(
                path,
                custom_objects=custom_objects,
                compile=True
            )

            return self.model

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            try:
                self.logger.info("Attempting to load model with alternative method")
                self.model = tf.keras.models.load_model(
                    path,
                    custom_objects={
                        'direction_enhanced_mse': tf.keras.losses.Huber(delta=0.25),
                        'SoftmaxLayer': SoftmaxLayer
                    },
                    compile=True
                )
                return self.model
            except Exception as e2:
                self.logger.error(f"Error in fallback loading: {e2}")
                return None

    def evaluate(self, X_test, y_test, fwd_returns_test=None):
        # Check if we're using ensemble
        if hasattr(self, 'ensemble_models') and self.ensemble_models is not None:
            try:
                y_pred = self.ensemble_models.predict(X_test)
                metrics = self._calculate_trading_metrics(y_pred, X_test, y_test, fwd_returns_test)
                return metrics
            except Exception as e:
                self.logger.error(f"Error in ensemble evaluation: {e}")
                # Fall back to single model if available

        # Single model evaluation
        if self.model is None:
            self.load_model()
            if self.model is None:
                self.logger.error("No model available for evaluation")
                return None

        self.logger.info("Evaluating model on test data")

        y_test = y_test.reshape(-1, 1) if len(y_test.shape) == 1 else y_test

        test_loss, test_mse, test_mae = self.model.evaluate(X_test, y_test, verbose=0)

        y_pred = self.predict(X_test)

        metrics = self._calculate_trading_metrics(y_pred, X_test, y_test, fwd_returns_test)
        metrics['loss'] = test_loss
        metrics['mse'] = test_mse
        metrics['mae'] = test_mae

        return metrics

    def _calculate_trading_metrics(self, y_pred, X_test, y_test, fwd_returns_test):
        metrics = {}

        metrics['predictions'] = y_pred.flatten()
        metrics['actual_returns'] = y_test.flatten()

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)

        metrics['prediction_correlation'] = np.corrcoef(y_pred.flatten(), y_test.flatten())[0, 1]

        if fwd_returns_test is not None:
            sorted_preds = np.sort(np.abs(y_pred).flatten())
            if len(sorted_preds) > 0:
                threshold_idx = max(1, int(len(sorted_preds) * 0.7))  # Using 30% of top predictions
                threshold = sorted_preds[threshold_idx]
                threshold = max(threshold, 0.0007)  # Reduced from 0.002
            else:
                threshold = 0.0007  # Reduced from 0.002

            trade_indices = np.where(np.abs(y_pred) > threshold)[0]

            if len(trade_indices) > 0:
                trade_returns = []
                win_count = 0
                profit_sum = 0
                loss_sum = 0

                for idx in trade_indices:
                    if idx >= len(fwd_returns_test):
                        continue

                    actual_return = fwd_returns_test[idx]
                    pred_direction = np.sign(y_pred[idx])
                    trade_return = pred_direction * actual_return

                    trade_returns.append(trade_return)

                    if trade_return > 0:
                        win_count += 1
                        profit_sum += trade_return
                    else:
                        loss_sum += abs(trade_return)

                total_trades = len(trade_returns)

                metrics['win_rate'] = win_count / total_trades if total_trades > 0 else 0
                metrics['avg_return'] = np.mean(trade_returns) if trade_returns else 0
                metrics['total_trades'] = total_trades
                metrics['profit_factor'] = profit_sum / loss_sum if loss_sum > 0 else float('inf')

                winning_returns = [r for r in trade_returns if r > 0]
                losing_returns = [abs(r) for r in trade_returns if r <= 0]

                if winning_returns:
                    metrics['avg_win'] = np.mean(winning_returns)
                    metrics['max_win'] = np.max(winning_returns)
                if losing_returns:
                    metrics['avg_loss'] = np.mean(losing_returns)
                    metrics['max_loss'] = np.max(losing_returns)

                returns_std = np.std(trade_returns) if len(trade_returns) > 1 else 1e-10
                metrics['sharpe_ratio'] = (metrics['avg_return'] / returns_std) * np.sqrt(365 * 24)

                negative_returns = [r for r in trade_returns if r < 0]
                downside_std = np.std(negative_returns) if negative_returns else 1e-10
                metrics['sortino_ratio'] = (metrics['avg_return'] / downside_std) * np.sqrt(365 * 24)

                if winning_returns and losing_returns:
                    win_prob = len(winning_returns) / total_trades
                    avg_win_pct = np.mean(winning_returns)
                    avg_loss_pct = np.mean(losing_returns)
                    if avg_loss_pct > 0:
                        kelly = win_prob - ((1 - win_prob) / (avg_win_pct / avg_loss_pct))
                        metrics['kelly_criterion'] = max(0, kelly)
            else:
                metrics['win_rate'] = 0
                metrics['avg_return'] = 0
                metrics['profit_factor'] = 0
                metrics['total_trades'] = 0
                metrics['sharpe_ratio'] = 0
                metrics['sortino_ratio'] = 0

        return metrics