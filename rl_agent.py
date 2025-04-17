import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from collections import deque
import random
import os
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(config.get("rl", "buffer_capacity", 10000))
        self.gamma = config.get("rl", "gamma", 0.95)
        self.epsilon = config.get("rl", "epsilon_start", 1.0)
        self.epsilon_min = config.get("rl", "epsilon_min", 0.01)
        self.epsilon_decay = config.get("rl", "epsilon_decay", 0.995)
        self.learning_rate = config.get("rl", "learning_rate", 0.001)
        self.update_target_frequency = config.get("rl", "update_target_frequency", 10)
        self.batch_size = config.get("rl", "batch_size", 128)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.train_step_counter = 0
        self.model_dir = Path(config.results_dir) / "models" / "rl"
        self.model_dir.mkdir(exist_ok=True, parents=True)

    def _build_model(self):
        # Using the Functional API instead of Sequential with input_shape
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.action_size)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, exploration=True):
        if exploration and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_frequency == 0:
            self.update_target_model()

        return loss

    def save(self, filename=None):
        if filename is None:
            filename = self.model_dir / "dqn_model.keras"
        self.model.save(filename)

        metadata = {
            "epsilon": self.epsilon,
            "train_step_counter": self.train_step_counter
        }

        with open(str(filename).replace('.keras', '_metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def load(self, filename=None):
        if filename is None:
            filename = self.model_dir / "dqn_model.keras"

        if os.path.exists(filename):
            self.model = tf.keras.models.load_model(filename)
            self.target_model = tf.keras.models.load_model(filename)

            metadata_file = str(filename).replace('.keras', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.epsilon = metadata.get("epsilon", self.epsilon)
                    self.train_step_counter = metadata.get("train_step_counter", 0)

            return True
        return False


class RLTradeEnvironment:
    def __init__(self, config):
        self.config = config
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.total_pnl = 0
        self.current_step = 0
        self.state = None
        self.done = False
        self.max_position_held = 0
        self.historical_pnl = []

    def reset(self, df=None, step=0):
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.total_pnl = 0
        self.current_step = step
        self.done = False
        self.max_position_held = 0
        self.df = df

        if df is not None and step < len(df):
            self.state = self._get_state(step)
        else:
            self.state = None
            self.done = True

        return self.state

    def _get_state(self, step):
        if self.df is None or step >= len(self.df):
            return None

        features = []

        # Market features
        for feature in self._get_feature_list():
            if feature in self.df.columns:
                features.append(self.df[feature].iloc[step])
            else:
                features.append(0)

        # Position features
        features.append(self.position)  # Current position: 1 for long, -1 for short, 0 for none
        features.append(self.position_size)  # Size of current position

        if self.position != 0 and self.entry_price > 0:
            current_price = self.df['close'].iloc[step]
            pnl = (current_price / self.entry_price - 1) * self.position
            features.append(pnl)  # Current unrealized PnL
        else:
            features.append(0)

        features.append(self.total_pnl)  # Total realized PnL

        return np.array(features, dtype=np.float32)

    def _get_feature_list(self):
        return [
            'close', 'rsi_14', 'macd_histogram_12_26_9', 'bb_width_20',
            'market_regime', 'volatility_regime', 'trend_strength',
            'atr_14', 'adx_14', 'ema_9', 'ema_21', 'ema_50'
        ]

    def step(self, action):
        if self.done or self.df is None:
            return self.state, 0, True, {}

        reward = 0

        # Process action (0: Hold, 1: Buy, 2: Sell)
        self._take_action(action)

        # Move to next step
        self.current_step += 1

        if self.current_step >= len(self.df):
            self.done = True
            # Close any open position at the end
            if self.position != 0:
                reward += self._close_position(self.current_step - 1)

        if not self.done:
            self.state = self._get_state(self.current_step)

            # Calculate reward based on position
            if self.position != 0:
                self.max_position_held += 1
                current_price = self.df['close'].iloc[self.current_step]
                prev_price = self.df['close'].iloc[self.current_step - 1]
                price_change = (current_price / prev_price) - 1

                reward = price_change * self.position * 100  # Scale for better learning

                # Add time decay penalty for holding positions too long
                max_hold_time = self.config.get("rl", "max_hold_periods", 48)
                if self.max_position_held > max_hold_time:
                    reward -= 0.02 * (self.max_position_held - max_hold_time)
        else:
            self.state = None

        info = {
            "position": self.position,
            "total_pnl": self.total_pnl,
            "position_held": self.max_position_held
        }

        return self.state, reward, self.done, info

    def _take_action(self, action):
        current_price = self.df['close'].iloc[self.current_step]

        if action == 1 and self.position <= 0:  # Buy
            if self.position < 0:  # Close short position first
                reward = self._close_position(self.current_step)

            self.position = 1
            self.entry_price = current_price
            self.position_size = 1.0
            self.max_position_held = 0

        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:  # Close long position first
                reward = self._close_position(self.current_step)

            self.position = -1
            self.entry_price = current_price
            self.position_size = 1.0
            self.max_position_held = 0

    def _close_position(self, step):
        if self.position == 0 or step >= len(self.df):
            return 0

        exit_price = self.df['close'].iloc[step]

        if self.position > 0:  # Long position
            pnl = (exit_price / self.entry_price - 1) * self.position_size
        else:  # Short position
            pnl = (self.entry_price / exit_price - 1) * self.position_size

        transaction_cost = self.config.get("rl", "transaction_cost", 0.001)
        pnl -= transaction_cost

        self.total_pnl += pnl
        self.historical_pnl.append(pnl)

        self.position = 0
        self.entry_price = 0
        self.position_size = 0

        return pnl * 100  # Scale for better learning

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
import tensorflow as tf
from pathlib import Path

from rl_agent import DQNAgent, RLTradeEnvironment


class RLSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("RLSignalGenerator")

        self.enabled = config.get("rl", "enabled", True)
        self.mode = config.get("rl", "mode", "hybrid")

        if not self.enabled:
            self.logger.info("RL Signal Generator is disabled")
            return

        self.feature_list = [
            'close', 'rsi_14', 'macd_histogram_12_26_9', 'bb_width_20',
            'market_regime', 'volatility_regime', 'trend_strength',
            'atr_14', 'adx_14', 'ema_9', 'ema_21', 'ema_50',
            'position', 'position_size', 'unrealized_pnl', 'total_pnl'
        ]

        self.state_size = len(self.feature_list)
        self.action_size = config.get("rl", "action_size", 3)

        self.agent = DQNAgent(self.state_size, self.action_size, config)
        self.environment = RLTradeEnvironment(config)

        self.use_pretrained = config.get("rl", "use_pretrained", True)
        self.train_during_backtest = config.get("rl", "train_during_backtest", True)

        self.confidence_threshold = config.get("rl", "confidence_threshold", 0.65)
        self.ensemble_weight = config.get("rl", "ensemble_weight", 0.5)

        if self.use_pretrained:
            self.load_model()

        self.training_frequency = config.get("rl", "training_frequency", 5)
        self.step_counter = 0

    def generate_signal(self, state, df=None, **kwargs):
        if not self.enabled or self.mode == "disabled":
            return {"signal_type": "NoTrade", "reason": "RL_Disabled"}

        self.step_counter += 1

        if df is None or len(df) == 0:
            return {"signal_type": "NoTrade", "reason": "NoData"}

        # Extract features for state
        try:
            state_vector = self._extract_state_vector(df)

            # Get RL agent's action
            action = self.agent.act(state_vector, exploration=self.train_during_backtest)

            # Remember the state for training
            if self.train_during_backtest:
                self.environment.reset(df, len(df) - 1)
                next_state, reward, done, _ = self.environment.step(action)
                self.agent.remember(state_vector, action, reward,
                                    next_state if next_state is not None else state_vector, done)

                # Train periodically
                if self.step_counter % self.training_frequency == 0:
                    loss = self.agent.train()

            # Convert action to signal
            signal = self._action_to_signal(action, df)

            # Add RL-specific data to signal
            signal["rl_action"] = action
            signal["rl_confidence"] = self._calculate_action_confidence(state_vector)

            return signal

        except Exception as e:
            self.logger.error(f"Error generating RL signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"signal_type": "NoTrade", "reason": f"RL_Error_{str(e)}"}

    def _extract_state_vector(self, df):
        state = []

        for feature in self.feature_list[:-4]:  # Exclude position features
            if feature in df.columns:
                value = df[feature].iloc[-1]
                state.append(float(value) if not pd.isna(value) else 0.0)
            else:
                state.append(0.0)

        # Position features (will be set by environment)
        state.extend([0.0, 0.0, 0.0, 0.0])  # position, size, unrealized_pnl, total_pnl

        return np.array(state, dtype=np.float32)

    def _action_to_signal(self, action, df):
        current_price = df['close'].iloc[-1]

        # Properly extract and convert market phase to string
        if 'market_regime' in df.columns:
            market_regime_value = df['market_regime'].iloc[-1]
            # Convert numeric market regime to string representation
            if isinstance(market_regime_value, (float, np.float64, np.float32, int, np.int64, np.int32)):
                # Map numeric values to string phases
                if market_regime_value > 0.5:
                    market_phase = "uptrend"
                elif market_regime_value < -0.5:
                    market_phase = "downtrend"
                elif 0.2 <= market_regime_value <= 0.5:
                    market_phase = "weak_uptrend"
                elif -0.5 <= market_regime_value <= -0.2:
                    market_phase = "weak_downtrend"
                elif -0.2 < market_regime_value < 0.2:
                    market_phase = "neutral"
                else:
                    market_phase = "neutral"
            else:
                # If it's already a string, use it directly
                market_phase = str(market_regime_value)
        else:
            market_phase = "neutral"

        if action == 1:  # Buy
            return {
                "signal_type": "StrongBuy",
                "direction": "long",
                "predicted_return": 0.005,
                "market_phase": market_phase,
                "ensemble_score": self.confidence_threshold + 0.1
            }
        elif action == 2:  # Sell
            return {
                "signal_type": "StrongSell",
                "direction": "short",
                "predicted_return": -0.005,
                "market_phase": market_phase,
                "ensemble_score": self.confidence_threshold + 0.1
            }
        else:  # Hold
            return {
                "signal_type": "NoTrade",
                "reason": "RL_Hold",
                "direction": "neutral",
                "predicted_return": 0.0,
                "market_phase": market_phase,
                "ensemble_score": 0.4
            }

    def _calculate_action_confidence(self, state):
        act_values = self.agent.model.predict(np.array([state]), verbose=0)[0]
        max_q_value = np.max(act_values)
        min_q_value = np.min(act_values)
        range_q = max(abs(max_q_value - min_q_value), 1e-5)

        # Normalize the confidence
        normalized_confidence = 0.5 + 0.5 * (max_q_value / range_q)

        # Clip to reasonable values
        return min(normalized_confidence, 0.95)

    def save_model(self):
        if self.enabled and hasattr(self, 'agent'):
            self.agent.save()
            self.logger.info("RL model saved")

    def load_model(self):
        if self.enabled and hasattr(self, 'agent'):
            success = self.agent.load()
            if success:
                self.logger.info("RL model loaded successfully")
            else:
                self.logger.warning("No pretrained RL model found, using new model")