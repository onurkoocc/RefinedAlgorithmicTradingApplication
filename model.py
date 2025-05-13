import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout, Lambda, Add, LayerNormalization, \
    GlobalAveragePooling1D, Concatenate, Multiply, Bidirectional, GRU, Layer
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import pickle
import json
from pathlib import Path


class SoftmaxLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class OptimizedGrowthMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val_dummy, fwd_returns_val, config, model_idx=None):
        super().__init__()
        self.X_val = X_val
        self.fwd_returns_val = fwd_returns_val
        self.config = config.get("model", "growth_metric", {})
        self.model_idx = model_idx

        self.monthly_target = self.config.get("monthly_target", 0.10)
        self.min_target = self.config.get("min_target", 0.07)
        self.max_target = self.config.get("max_target", 0.13)
        self.drawdown_weight = self.config.get("drawdown_weight", 1.8)
        self.consistency_weight = self.config.get("consistency_weight", 1.2)
        self.avg_return_weight = self.config.get("avg_return_weight", 0.8)
        self.threshold_pct_config = self.config.get("threshold_pct", 0.30)

        self.transaction_cost = 0.001
        self.trade_periods_per_month = (24 * 30 * 60) / 30

        self.best_growth_score = -np.inf
        self.logger = logging.getLogger(
            f"GrowthMetricCallback_M{model_idx}" if model_idx is not None else "GrowthMetricCallback")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.X_val) == 0 or len(self.fwd_returns_val) == 0:
            self._update_logs_with_defaults(logs)
            return

        try:
            y_pred_scaled = self.model.predict(self.X_val, verbose=0, batch_size=min(len(self.X_val), 256))
            y_pred_scaled = np.nan_to_num(y_pred_scaled.flatten(), nan=0.0)

            if len(y_pred_scaled) > 10:
                abs_preds = np.abs(y_pred_scaled)
                threshold_val = np.percentile(abs_preds, (1 - self.threshold_pct_config) * 100)
                threshold_val = max(threshold_val, 0.05)
            else:
                threshold_val = 0.1

            trade_indices = np.where(np.abs(y_pred_scaled) > threshold_val)[0]

            if len(trade_indices) == 0:
                self._update_logs_with_defaults(logs)
                return

            trade_returns_pct = []
            equity_curve = [1.0]

            for idx in trade_indices:
                if idx >= len(self.fwd_returns_val): continue

                actual_fwd_return_pct = float(self.fwd_returns_val[idx])
                pred_direction = np.sign(y_pred_scaled[idx])

                trade_pnl_pct = pred_direction * actual_fwd_return_pct - self.transaction_cost
                trade_returns_pct.append(trade_pnl_pct)
                equity_curve.append(equity_curve[-1] * (1 + trade_pnl_pct))

            if not trade_returns_pct:
                self._update_logs_with_defaults(logs)
                return

            n_trades = len(trade_returns_pct)
            avg_trade_pnl_pct = np.mean(trade_returns_pct)

            trades_per_val_period = n_trades / len(self.X_val) if len(self.X_val) > 0 else 0
            trades_per_month_sim = trades_per_val_period * self.trade_periods_per_month

            if 1 + avg_trade_pnl_pct > 0 and trades_per_month_sim > 0:
                monthly_growth_sim = ((1 + avg_trade_pnl_pct) ** trades_per_month_sim) - 1
            else:
                monthly_growth_sim = avg_trade_pnl_pct * trades_per_month_sim

            peak_sim = np.maximum.accumulate(equity_curve)
            drawdown_sim = (peak_sim - equity_curve) / peak_sim
            max_drawdown_sim = np.max(drawdown_sim) if len(drawdown_sim) > 0 else 1.0

            growth_dist_score = 0.0
            if monthly_growth_sim < self.min_target:
                growth_dist_score = (monthly_growth_sim / self.min_target) if self.min_target > 0 else 0
            elif monthly_growth_sim > self.max_target:
                growth_dist_score = max(0, 1 - (monthly_growth_sim - self.max_target) / (self.max_target * 0.5))
            else:
                growth_dist_score = 1.0 + (
                            1.0 - abs(monthly_growth_sim - self.monthly_target) / (self.monthly_target + 1e-9))
            growth_dist_score = np.clip(growth_dist_score, 0, 2.0)

            dd_penalty = np.clip(1.0 - (max_drawdown_sim / 0.25) * self.drawdown_weight, 0, 1.0)

            consistency = 1.0 / (np.std(trade_returns_pct) + 0.1)
            consistency_score = np.clip(consistency, 0, 2.0)

            growth_score = (self.avg_return_weight * growth_dist_score + \
                            self.drawdown_weight * dd_penalty + \
                            self.consistency_weight * consistency_score) / \
                           (self.avg_return_weight + self.drawdown_weight + self.consistency_weight)

            logs['growth_score'] = float(growth_score)
            logs['monthly_growth_sim'] = float(monthly_growth_sim)
            logs['max_drawdown_sim'] = float(max_drawdown_sim)
            logs['trades_per_month_sim'] = float(trades_per_month_sim)

            if self.model_idx is not None: logs[f'growth_score_m{self.model_idx}'] = float(growth_score)

            if growth_score > self.best_growth_score:
                self.best_growth_score = growth_score
                self.logger.info(
                    f"Epoch {epoch + 1}: New best growth score: {growth_score:.4f} (Monthly: {monthly_growth_sim * 100:.2f}%, DD: {max_drawdown_sim * 100:.2f}%, Trades/Mo: {trades_per_month_sim:.1f})")

        except Exception as e:
            self.logger.error(f"Error in GrowthMetricCallback: {e}")
            self._update_logs_with_defaults(logs)

    def _update_logs_with_defaults(self, logs):
        logs['growth_score'] = 0.0
        logs['monthly_growth_sim'] = 0.0
        logs['max_drawdown_sim'] = 1.0
        logs['trades_per_month_sim'] = 0.0
        if self.model_idx is not None: logs[f'growth_score_m{self.model_idx}'] = 0.0


class NoisySequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, config):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentation_config = config.get("model", "data_augmentation", {})
        self.noise_level = self.augmentation_config.get("noise_level", 0.015)
        self.roll_prob = self.augmentation_config.get("roll_probability", 0.3)
        self.mask_prob = self.augmentation_config.get("mask_probability", 0.25)
        self.scale_prob = self.augmentation_config.get("scale_probability", 0.15)

        self.std_dev_per_feature = np.std(X, axis=(0, 1), keepdims=True)
        self.std_dev_per_feature[self.std_dev_per_feature == 0] = 1e-6

        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices].copy()
        batch_y = self.y[batch_indices]

        noise = np.random.normal(0, self.noise_level * self.std_dev_per_feature, batch_X.shape)
        batch_X += noise

        for i in range(len(batch_X)):
            if np.random.random() < self.roll_prob:
                shift = np.random.randint(-batch_X.shape[1] // 4, batch_X.shape[1] // 4)
                batch_X[i] = np.roll(batch_X[i], shift=shift, axis=0)

            if np.random.random() < self.mask_prob:
                if np.random.random() < 0.5:
                    num_features_to_mask = np.random.randint(1, max(2, batch_X.shape[2] // 10))
                    features_to_mask = np.random.choice(batch_X.shape[2], num_features_to_mask, replace=False)
                    batch_X[i, :, features_to_mask] = 0
                else:
                    num_ts_to_mask = np.random.randint(1, max(2, batch_X.shape[1] // 10))
                    ts_to_mask = np.random.choice(batch_X.shape[1], num_ts_to_mask, replace=False)
                    batch_X[i, ts_to_mask, :] = 0

            if np.random.random() < self.scale_prob:
                scale_factor = np.random.uniform(0.9, 1.1)
                batch_X[i] *= scale_factor
        return batch_X, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class OptimizedHybridModel:
    def __init__(self, config, input_shape):
        self.config_full = config
        self.config = config.get("model", "architecture")
        self.input_shape = input_shape
        self.model = None
        self.logger = logging.getLogger("OptimizedHybridModel")

    def build_model(self):
        inputs = Input(shape=self.input_shape, dtype=tf.float32, name="hybrid_input")
        x = BatchNormalization(momentum=0.99)(inputs)

        proj_size = self.config.get("projection_size", 96)
        x = Dense(proj_size, activation='linear')(x)

        pos_encoding = self._positional_encoding(self.input_shape[0], proj_size)
        x = x + pos_encoding

        num_transformer_layers = self.config.get("transformer_layers", 2)
        num_heads = self.config.get("transformer_heads", 6)
        transformer_dropout = self.config.get("transformer_dropout", 0.20)
        for i in range(num_transformer_layers):
            x = self._transformer_encoder_layer(x, units=proj_size, num_heads=num_heads,
                                                dropout_rate=transformer_dropout, name=f"transformer_enc_{i}")

        rec_units = self.config.get("recurrent_units", 50)
        rec_out = GRU(rec_units, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)
        rec_out = LayerNormalization()(rec_out)

        x_pooled = GlobalAveragePooling1D()(rec_out)

        l2_reg_val = self.config_full.get("model", "l2_reg", 5e-5)
        dropout_rate_dense = self.config_full.get("model", "dropout_rate", 0.20)

        d1_units = self.config.get("dense_units1", 64)
        d1 = Dense(d1_units, activation='swish', kernel_regularizer=l2(l2_reg_val))(x_pooled)
        d1 = BatchNormalization()(d1)
        d1 = Dropout(dropout_rate_dense)(d1)

        d2_units = self.config.get("dense_units2", 24)
        d2 = Dense(d2_units, activation='swish', kernel_regularizer=l2(l2_reg_val))(d1)
        d2 = BatchNormalization()(d2)
        d2 = Dropout(dropout_rate_dense)(d2)

        outputs = Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg_val), dtype='float32')(d2)

        model = Model(inputs, outputs, name="optimized_hybrid_v2")

        initial_lr = self.config_full.get("model", "initial_learning_rate", 3e-4)
        optimizer = AdamW(learning_rate=initial_lr, weight_decay=1e-5)

        model.compile(optimizer=optimizer, loss=self._direction_aware_mse, metrics=['mae'])
        self.model = model
        return model

    def _positional_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def _transformer_encoder_layer(self, inputs, units, num_heads, dropout_rate, name):
        norm1 = LayerNormalization(epsilon=1e-6, name=f"{name}_norm1")(inputs)
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units // num_heads, dropout=dropout_rate, name=f"{name}_mha"
        )(query=norm1, value=norm1, key=norm1)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = Add(name=f"{name}_add1")([inputs, attn_output])

        norm2 = LayerNormalization(epsilon=1e-6, name=f"{name}_norm2")(out1)
        ffn_output = Dense(units * 2, activation="swish", name=f"{name}_ffn_dense1")(norm2)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(units, name=f"{name}_ffn_dense2")(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        return Add(name=f"{name}_add2")([out1, ffn_output])

    def _direction_aware_mse(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        true_sign = tf.sign(y_true)
        pred_sign = tf.sign(y_pred)
        direction_mismatch = tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)

        mismatch_penalty = direction_mismatch * (1 + tf.abs(y_true))

        direction_loss_weight = self.config_full.get("model", "direction_loss_weight", 0.6)
        total_loss = mse + direction_loss_weight * tf.reduce_mean(mismatch_penalty)
        return total_loss

    def save_model(self, path):
        if self.model:
            self.model.save(path)
            self.logger.info(f"Saved model to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path, compile=False, custom_objects={"SoftmaxLayer": SoftmaxLayer})
            initial_lr = self.config_full.get("model", "initial_learning_rate", 3e-4)
            optimizer = AdamW(learning_rate=initial_lr, weight_decay=1e-5)
            self.model.compile(optimizer=optimizer, loss=self._direction_aware_mse, metrics=['mae'])
            self.logger.info(f"Loaded model from {path} and recompiled.")
            return True
        return False


class TradingModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TradingModel")
        self.model_path_base = config.get("model", "model_path")

        self.use_ensemble = config.get("model", "use_ensemble", True)
        self.ensemble_size = config.get("model", "ensemble_size", 3)
        self.ensemble_models = []
        self.main_model = None

        if config.get("model", "use_mixed_precision", True):
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except Exception as e:
                self.logger.warning(f"Could not set mixed precision policy: {e}")

        self._configure_gpu_memory()
        tf.random.set_seed(config.get("model", "seed", 42))
        np.random.seed(config.get("model", "seed", 42))

    def _configure_gpu_memory(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                self.logger.error(f"Error setting memory growth: {e}")

    def _get_callbacks(self, X_val, y_val_dummy, fwd_returns_val, model_idx=None):
        model_filename = os.path.basename(self.model_path_base)
        model_dir = os.path.dirname(self.model_path_base)

        if model_idx is not None:
            base, ext = os.path.splitext(model_filename)
            checkpoint_filename = f"{base}_ensemble_{model_idx}{ext}"
        else:
            checkpoint_filename = model_filename
        checkpoint_path = os.path.join(model_dir, checkpoint_filename)
        os.makedirs(model_dir, exist_ok=True)

        callbacks = [
            OptimizedGrowthMetricCallback(X_val, y_val_dummy, fwd_returns_val, self.config, model_idx=model_idx),
            tf.keras.callbacks.EarlyStopping(
                monitor='growth_score', patience=self.config.get("model", "early_stopping_patience", 5),
                mode='max', restore_best_weights=True, min_delta=0.0005, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor='growth_score', save_best_only=True,
                mode='max', verbose=0, save_weights_only=False
            )
        ]

        def lr_schedule(epoch, lr):
            if epoch > 0 and epoch % 5 == 0:
                return lr * self.config.get("model", "lr_decay_factor", 0.7)
            return lr

        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0))
        return callbacks

    def train_model(self, X_train, y_train, X_val, y_val_dummy, df_val_unused, fwd_returns_val,
                    class_weight_unused=None):
        self.logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

        input_shape = (X_train.shape[1], X_train.shape[2])
        batch_size = self.config.get("model", "batch_size", 128)
        train_seq = NoisySequence(X_train, y_train.reshape(-1, 1), batch_size, self.config)

        if self.use_ensemble and self.ensemble_size > 1:
            self.ensemble_models = []
            best_overall_growth_score = -np.inf

            for i in range(self.ensemble_size):
                self.logger.info(f"Training ensemble model {i + 1}/{self.ensemble_size}")
                tf.random.set_seed(self.config.get("model", "seed", 42) + i * 10)
                np.random.seed(self.config.get("model", "seed", 42) + i * 10)

                hybrid_model_instance = OptimizedHybridModel(self.config, input_shape)
                current_model = hybrid_model_instance.build_model()

                callbacks = self._get_callbacks(X_val, y_val_dummy, fwd_returns_val, model_idx=i)
                history = current_model.fit(train_seq, validation_data=(X_val, y_val_dummy.reshape(-1, 1)),
                                            epochs=self.config.get("model", "epochs", 25), callbacks=callbacks,
                                            verbose=1)

                self.ensemble_models.append(current_model)

                current_growth_score = max(history.history.get('growth_score', [-np.inf]))
                if current_growth_score > best_overall_growth_score:
                    best_overall_growth_score = current_growth_score
                    self.main_model = current_model
                    hybrid_model_instance.save_model(self.model_path_base)
                    self.logger.info(f"Ensemble model {i + 1} is new best. Saved to main path.")

            if not self.main_model and self.ensemble_models:
                self.main_model = self.ensemble_models[0]
                self.ensemble_models[0].save(self.model_path_base)
                self.logger.info("No ensemble model improved score, using first ensemble model as main.")

            return self.main_model
        else:
            hybrid_model_instance = OptimizedHybridModel(self.config, input_shape)
            self.main_model = hybrid_model_instance.build_model()
            callbacks = self._get_callbacks(X_val, y_val_dummy, fwd_returns_val)
            self.main_model.fit(train_seq, validation_data=(X_val, y_val_dummy.reshape(-1, 1)),
                                epochs=self.config.get("model", "epochs", 25), callbacks=callbacks, verbose=1)
            hybrid_model_instance.save_model(self.model_path_base)
            return self.main_model

    def predict(self, X, batch_size=None):
        effective_batch_size = batch_size or self.config.get("model", "batch_size", 128)
        if self.use_ensemble and self.ensemble_models:
            predictions = [model.predict(X, verbose=0, batch_size=effective_batch_size) for model in
                           self.ensemble_models]
            return np.mean(predictions, axis=0)
        elif self.main_model:
            return self.main_model.predict(X, verbose=0, batch_size=effective_batch_size)

        self.logger.error("No model available for prediction. Load or train a model first.")
        return np.zeros((len(X), 1))

    def load_model(self, model_path_override=None):
        if self.use_ensemble and self.ensemble_size > 0:
            loaded_ensemble = []
            base, ext = os.path.splitext(model_path_override or self.model_path_base)
            for i in range(self.ensemble_size):
                ensemble_member_path = f"{base}_ensemble_{i}{ext}"
                if os.path.exists(ensemble_member_path):
                    try:
                        model_instance = OptimizedHybridModel(self.config, input_shape=(
                        self.config.get("model", "sequence_length", 60), self.config.get("model", "max_features", 60)))
                        if model_instance.load_model(ensemble_member_path):
                            loaded_ensemble.append(model_instance.model)
                        else:
                            self.logger.warning(f"Could not load ensemble member: {ensemble_member_path}")
                    except Exception as e:
                        self.logger.error(f"Error loading ensemble member {ensemble_member_path}: {e}")

            if loaded_ensemble:
                self.ensemble_models = loaded_ensemble
                self.main_model = self.ensemble_models[0]
                self.logger.info(f"Successfully loaded {len(self.ensemble_models)} ensemble models.")
                if os.path.exists(model_path_override or self.model_path_base):
                    main_instance = OptimizedHybridModel(self.config, input_shape=(
                    self.config.get("model", "sequence_length", 60), self.config.get("model", "max_features", 60)))
                    if main_instance.load_model(model_path_override or self.model_path_base):
                        self.main_model = main_instance.model
                        self.logger.info(
                            f"Successfully loaded main model from {model_path_override or self.model_path_base}")
                return self.main_model is not None

        main_instance = OptimizedHybridModel(self.config, input_shape=(
        self.config.get("model", "sequence_length", 60), self.config.get("model", "max_features", 60)))
        if main_instance.load_model(model_path_override or self.model_path_base):
            self.main_model = main_instance.model
            self.use_ensemble = False
            self.ensemble_models = []
            self.logger.info(
                f"Successfully loaded single main model from {model_path_override or self.model_path_base}")
            return True

        self.logger.error(f"Failed to load any model from path: {model_path_override or self.model_path_base}")
        return False