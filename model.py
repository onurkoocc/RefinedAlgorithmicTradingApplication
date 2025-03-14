import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Conv1D, Dropout, Layer
from tensorflow.keras.layers import MaxPooling1D, Bidirectional, GRU, Add, LayerNormalization, GlobalAveragePooling1D, \
    MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.mixed_precision import set_global_policy


class SimpleTradeMetric(Callback):
    def __init__(self, X_val, y_val, fwd_returns_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.fwd_returns_val = fwd_returns_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if len(self.X_val) == 0:
            logs['val_avg_return'] = 0.0
            logs['val_win_rate'] = 0.0
            return

        try:
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)

            trade_indices = np.where(y_pred_classes != 2)[0]

            if len(trade_indices) == 0:
                logs['val_avg_return'] = 0.0
                logs['val_win_rate'] = 0.0
                return

            trade_returns = []
            win_count = 0

            for idx in trade_indices:
                if idx >= len(self.fwd_returns_val):
                    continue

                actual_return = self.fwd_returns_val[idx]
                pred_class = y_pred_classes[idx]

                if pred_class in [3, 4]:
                    trade_return = actual_return
                else:
                    trade_return = -actual_return

                trade_returns.append(trade_return)
                if trade_return > 0:
                    win_count += 1

            avg_return = np.mean(trade_returns) if trade_returns else 0.0
            win_rate = win_count / len(trade_returns) if trade_returns else 0.0

            logs['val_avg_return'] = avg_return
            logs['val_win_rate'] = win_rate

        except Exception as e:
            print(f"Error in SimpleTradeMetric: {e}")
            logs['val_avg_return'] = 0.0
            logs['val_win_rate'] = 0.0


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W = None
        self.b = None
        self.input_spec = None

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        self.built = True
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        a = tf.nn.softmax(e, axis=1)

        weighted = x * a

        output = tf.reduce_sum(weighted, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ShapeLayer(Layer):
    def __init__(self, **kwargs):
        super(ShapeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class TradingModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TradingModel")

        self.model_path = config.get("model", "model_path")
        self.sequence_length = config.get("model", "sequence_length", 72)
        self.batch_size = config.get("model", "batch_size", 256)
        self.epochs = config.get("model", "epochs", 20)
        self.early_stopping_patience = config.get("model", "early_stopping_patience", 5)
        self.project_name = config.get("model", "project_name", "btc_trading")

        self.dropout_rate = config.get("model", "dropout_rate", 0.2)

        if config.get("model", "use_mixed_precision", True):
            set_global_policy('mixed_float16')

        self._configure_gpu()

        self.seed = 42
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.model = None

    def _configure_gpu(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if gpus:
                self.logger.info(f"Found {len(gpus)} GPU(s)")
            else:
                self.logger.info("No GPU found, using CPU")
        except Exception as e:
            self.logger.warning(f"Error configuring GPU: {e}")

    def build_model(self, input_shape):
        model = self._build_simpler_model(input_shape)

        batch_size = 1
        dummy_input = np.zeros((batch_size,) + input_shape)
        _ = model(dummy_input, training=False)

        self.logger.info(f"Built hybrid model with shape {input_shape}")

        return model

    def train_model(self, X_train, y_train, X_val, y_val, df_val=None, fwd_returns_val=None, class_weight=None):
        self.logger.info(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
        self.logger.info(f"Validation data shape: {X_val.shape}, labels shape: {y_val.shape}")

        input_shape = (X_train.shape[1], X_train.shape[2])

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.logger.info("Building hybrid model")
        self.model = self.build_model(input_shape)

        callbacks = self._get_callbacks(X_val, y_val, fwd_returns_val)

        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

        self.model.save(self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")

        tf.keras.backend.clear_session()

        return self.model

    def _get_callbacks(self, X_val, y_val, fwd_returns_val, model_path=None):
        checkpoint_path = model_path or self.model_path

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        trade_callback = SimpleTradeMetric(X_val, y_val, fwd_returns_val)

        early_stopping = EarlyStopping(
            monitor='val_avg_return',
            patience=self.early_stopping_patience,
            mode='max',
            restore_best_weights=True,
            min_delta=0.001,
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
                checkpoint_path + ".weights.h5",
                monitor='val_avg_return',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=True
            )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_avg_return',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )

        return [trade_callback, early_stopping, checkpoint, reduce_lr]

    def predict(self, X, batch_size=None):
        if self.model is None:
            self.load_model()
            if self.model is None:
                self.logger.error("No model available for prediction")
                neutral = np.zeros((len(X), 5))
                neutral[:, 2] = 1.0
                return neutral

        if len(X) == 0:
            return np.array([])

        if np.isnan(X).any() or np.isinf(X).any():
            self.logger.warning("NaN or Inf values in input data, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size = batch_size or min(self.batch_size, 256)

        try:
            predictions = self.model.predict(X, verbose=0, batch_size=batch_size)

            if np.isnan(predictions).any() or np.isinf(predictions).any():
                self.logger.warning("NaN or Inf values in predictions, replacing with zeros")
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

                row_sums = predictions.sum(axis=1, keepdims=True)
                valid_rows = (row_sums > 1e-10).flatten()

                if np.any(valid_rows):
                    predictions[valid_rows] = predictions[valid_rows] / row_sums[valid_rows]

                invalid_rows = ~valid_rows
                if np.any(invalid_rows):
                    predictions[invalid_rows, 2] = 1.0

            return predictions

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            neutral = np.zeros((len(X), 5))
            neutral[:, 2] = 1.0
            return neutral

    def load_model(self, model_path=None):
        path = model_path or self.model_path

        custom_objects = {
            "AttentionLayer": AttentionLayer,
            "ShapeLayer": ShapeLayer
        }

        if not os.path.exists(path):
            self.logger.warning(f"Model not found at {path}")
            return None

        try:
            self.logger.info(f"Loading model from {path}")
            self.model = tf.keras.models.load_model(
                path,
                custom_objects=custom_objects
            )
            return self.model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None

    def evaluate(self, X_test, y_test, fwd_returns_test=None):
        if self.model is None:
            self.load_model()
            if self.model is None:
                self.logger.error("No model available for evaluation")
                return None

        self.logger.info("Evaluating model on test data")

        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)

        y_pred_prob = self.predict(X_test)

        metrics = self._calculate_metrics(y_pred_prob, X_test, y_test, fwd_returns_test)
        metrics['loss'] = test_loss
        metrics['accuracy'] = test_acc

        return metrics

    def _calculate_metrics(self, y_pred_prob, X_test, y_test, fwd_returns_test):
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        y_true_class = np.argmax(y_test, axis=1)

        metrics = {
            'predicted_classes': y_pred_class,
            'true_classes': y_true_class
        }

        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_true_class, y_pred_class)
        metrics['confusion_matrix'] = cm

        report = classification_report(y_true_class, y_pred_class, output_dict=True)
        metrics['classification_report'] = report

        if fwd_returns_test is not None:
            trade_indices = np.where(y_pred_class != 2)[0]

            if len(trade_indices) > 0:
                trade_returns = []
                win_count = 0

                for idx in trade_indices:
                    if idx >= len(fwd_returns_test):
                        continue

                    actual_return = fwd_returns_test[idx]
                    pred_class = y_pred_class[idx]

                    if pred_class in [3, 4]:
                        trade_return = actual_return
                    else:
                        trade_return = -actual_return

                    trade_returns.append(trade_return)
                    if trade_return > 0:
                        win_count += 1

                total_trades = len(trade_returns)

                metrics['win_rate'] = win_count / total_trades if total_trades > 0 else 0
                metrics['avg_return'] = np.mean(trade_returns) if trade_returns else 0
                metrics['total_trades'] = total_trades

                winning_returns = [r for r in trade_returns if r > 0]
                losing_returns = [abs(r) for r in trade_returns if r <= 0]

                gross_profit = sum(winning_returns)
                gross_loss = sum(losing_returns)

                metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            else:
                metrics['win_rate'] = 0
                metrics['avg_return'] = 0
                metrics['profit_factor'] = 0
                metrics['total_trades'] = 0

        return metrics

    def _build_simpler_model(self, input_shape):
        inputs = Input(shape=input_shape, dtype=tf.float32)

        # Feature Extraction Layers
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)

        # Sequence Modeling
        x = Bidirectional(GRU(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)

        # Apply ShapeLayer to get shape info
        x = ShapeLayer()(x)

        # Get feature dimension (static)
        feature_dim = x.shape[2]

        # Self-Attention
        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = Dropout(0.3)(x)

        # Transformer Encoder (2 layers)
        for i in range(2):
            # Multi-head attention
            attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
            attention_output = Dropout(0.3)(attention_output)
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)

            # Feed-forward network
            ffn = Dense(128, activation='relu')(x)
            ffn = Dense(feature_dim)(ffn)  # Match feature dimension of x
            ffn = Dropout(0.3)(ffn)
            x = Add()([x, ffn])
            x = LayerNormalization()(x)

        # Global Pooling
        x = GlobalAveragePooling1D()(x)

        # Output Layers
        x = Dense(32, activation='relu')(x)
        outputs = Dense(5, activation='softmax', dtype=tf.float32)(x)

        model = Model(inputs, outputs, name="hybrid_transformer_model")

        model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=['accuracy']
        )

        return model