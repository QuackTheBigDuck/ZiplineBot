import pandas as pd
import numpy as np
import time
from collections import deque
import random
from scipy.optimize import minimize
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging
import pyimgur
import yfinance as yf
from discord_webhook import DiscordWebhook, DiscordEmbed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import io
import base64
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
CLIENT_ID = "IMGUR CLIENT ID"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Discord webhook URL
discordwebhookurl = "DISCORD URL"

class MLModel:
    def __init__(self):
        self.lstm_model = None
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, data):
        data = data.copy()
        logger.info(f"Available columns: {data.columns.tolist()}")
        
        # Calculate technical indicators
        if 'close' in data.columns:
            data['rsi'] = self.calculate_rsi(data['close'])
            data['macd'] = self.calculate_macd(data['close'])
            data['ema9'] = data['close'].ewm(span=9, adjust=False).mean()
            data['ema21'] = data['close'].ewm(span=21, adjust=False).mean()
        
        # If we have 'high' and 'low', calculate ATR
        if 'high' in data.columns and 'low' in data.columns:
            data['atr'] = (data['high'] - data['low']).rolling(window=14).mean()
        
        # If we have 'open' and 'close', calculate price range
        if 'open' in data.columns and 'close' in data.columns:
            data['price_range'] = (data['close'] - data['open']).abs().rolling(window=14).mean()
        
        # Calculate Bollinger Bands if 'close' is available
        if 'close' in data.columns:
            data['bb_upper'], data['bb_lower'] = self.calculate_bollinger_bands(data['close'])
        
        # Calculate returns if 'close' is available
        if 'close' in data.columns:
            data['returns'] = data['close'].pct_change()
        
        # Add day of week and hour as features
        data['day_of_week'] = data.index.dayofweek
        data['hour'] = data.index.hour
        
        data.dropna(inplace=True)

        # Use only available features
        available_features = ['day_of_week', 'hour']
        for feature in ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema9', 'ema21', 'atr', 'bb_upper', 'bb_lower', 'price_range', 'returns']:
            if feature in data.columns:
                available_features.append(feature)

        X = data[available_features].values
        y = data['close'].values if 'close' in data.columns else np.zeros(len(data))

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.price_scaler.fit_transform(y.reshape(-1, 1))

        X_reshaped = []
        y_reshaped = []
        for i in range(60, len(X_scaled)):
            X_reshaped.append(X_scaled[i-60:i])
            y_reshaped.append(y_scaled[i])

        return np.array(X_reshaped), np.array(y_reshaped), X, y

    def build_lstm_model(self, n_features):
        model = Sequential([
            LSTM(256, return_sequences=True, input_shape=(60, n_features)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self, data):
        X, y, X_rf, y_rf = self.prepare_data(data)
        if X.shape[0] > 0 and y.shape[0] > 0:
            try:
                # Build and train LSTM model
                if self.lstm_model is None:
                    self.lstm_model = self.build_lstm_model(X.shape[2])
                self.lstm_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
                lstm_loss = self.lstm_model.evaluate(X, y, verbose=0)
                logger.info(f"LSTM model loss: {lstm_loss:.4f}")

                # Train Random Forest model
                X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
                self.rf_model.fit(X_train, y_train)
                rf_predictions = self.rf_model.predict(X_test)
                rf_mse = mean_squared_error(y_test, rf_predictions)
                logger.info(f"Random Forest MSE: {rf_mse:.4f}")

                # Save models
                self.lstm_model.save('lstm_model.keras')
                joblib.dump(self.rf_model, 'rf_model.joblib')
            except Exception as e:
                logger.exception(f"An error occurred during model training: {e}")
        else:
            logger.warning("Not enough data to train the models")

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def predict(self, data):
        X, _, X_rf, _ = self.prepare_data(data)
        if X.shape[0] > 0:
            lstm_prediction = self.lstm_model.predict(X)[-1][0]
            lstm_prediction = self.price_scaler.inverse_transform([[lstm_prediction]])[0][0]
                
            rf_prediction = self.rf_model.predict(X_rf[-1].reshape(1, -1))[0]
                
            # Ensemble prediction (weighted average)
            ensemble_prediction = 0.6 * lstm_prediction + 0.4 * rf_prediction
            return ensemble_prediction
        else:
            logger.warning("Not enough data for prediction")
            return None

class Portfolio:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.positions = {}
        self.trade_history = []

    def update_portfolio_value(self, asset_prices):
        self.portfolio_value = self.cash + sum(self.positions.get(asset, 0) * price for asset, price in asset_prices.items())

    def execute_trade(self, asset, action, quantity, price, timestamp):
        if action == 'buy':
            cost = quantity * price
            if self.cash >= cost:
                self.cash -= cost
                self.positions[asset] = self.positions.get(asset, 0) + quantity
                self.trade_history.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'cost': cost
                })
                return True
            else:
                return False
        elif action == 'sell':
            if self.positions.get(asset, 0) >= quantity:
                revenue = quantity * price
                self.cash += revenue
                self.positions[asset] -= quantity
                self.trade_history.append({
                    'timestamp': timestamp,
                    'asset': asset,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'revenue': revenue
                })
                return True
            else:
                return False
            
class TradeMemory:
    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)
        self.batch_size = 32

    def add_trade(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def size(self):
        return len(self.memory)

class TradingBot:
    def __init__(self, assets, initial_cash=100000):
        self.assets = assets
        self.ml_models = {asset: MLModel() for asset in assets}
        self.portfolio = Portfolio(initial_cash)
        self.trade_memory = TradeMemory()
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.optimal_weights = None
        self.consecutive_losses = {asset: 0 for asset in assets}
        self.loss_threshold = 3
        self.profit_threshold = 0.015
        self.take_profit = 0.03
        self.stop_loss = 0.025
        self.max_position_size = 0.08
        self.max_drawdown = 0.35

    def tune_hyperparameters(self, data, portfolio):
        param_grid = {
            'profit_threshold': [0.01, 0.015, 0.02],
            'take_profit': [0.02, 0.03, 0.04],
            'stop_loss': [0.015, 0.02, 0.025],
            'max_position_size': [0.05, 0.08, 0.1],
            'learning_rate': [0.05, 0.1, 0.15],
            'discount_factor': [0.9, 0.95, 0.99],
            'epsilon': [0.05, 0.1, 0.15]
        }

        param_combinations = list(itertools.product(*param_grid.values()))

        best_performance = float('-inf')
        best_params = None

        for params in param_combinations:
            try:
                self.profit_threshold, self.take_profit, self.stop_loss, self.max_position_size, \
                self.learning_rate, self.discount_factor, self.epsilon = params

                self.positions = {asset: 0 for asset in self.assets}
                self.buy_prices = {asset: None for asset in self.assets}
                self.highest_prices = {asset: 0 for asset in self.assets}
                self.trade_history = {asset: [] for asset in self.assets}
                self.portfolio_value_history = []
                self.trade_memory = TradeMemory()
                self.q_table = {}

                test_portfolio = Portfolio(portfolio.cash)
                for i in range(len(data)):
                    self.handle_data(data.iloc[:i+1], test_portfolio)

                performance = test_portfolio.portfolio_value

                if performance > best_performance:
                    best_performance = performance
                    best_params = params

                logger.info(f"Params: {dict(zip(param_grid.keys(), params))}, Performance: {performance}")

            except Exception as e:
                logger.error(f"Error during hyperparameter tuning: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        if best_params:
            self.profit_threshold, self.take_profit, self.stop_loss, self.max_position_size, \
            self.learning_rate, self.discount_factor, self.epsilon = best_params

            logger.info(f"Best parameters found: {dict(zip(param_grid.keys(), best_params))}")
            logger.info(f"Best performance: {best_performance}")
        else:
            logger.warning("No valid parameter combination found. Using default parameters.")

        return best_params, best_performance

    def check_for_loss(self, asset, current_price):
        if self.buy_prices[asset] is not None:
            return current_price < self.buy_prices[asset]
        return False

    def retrain_model(self, asset, data):
        logger.info(f"Retraining model for {asset} due to consecutive losses")
        self.ml_models[asset].train(data)
        self.consecutive_losses[asset] = 0

    def get_state(self, history):
        if len(history) < 2:
            return (0, 0, 0, 0)

        current_price = history['close'].iloc[-1]
        previous_price = history['close'].iloc[-2]
        rsi = self.ml_models[list(self.assets)[0]].calculate_rsi(history['close']).iloc[-1]
        macd = self.ml_models[list(self.assets)[0]].calculate_macd(history['close']).iloc[-1]
        
        price_change = (current_price - previous_price) / previous_price
        rsi_state = 1 if rsi > 70 else (-1 if rsi < 30 else 0)
        macd_state = 1 if macd > 0 else -1
        trend = 1 if current_price > history['close'].mean() else -1

        return (price_change, rsi_state, macd_state, trend)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(['buy', 'sell', 'hold'])
        
        q_values = self.q_table.get(state, {'buy': 0, 'sell': 0, 'hold': 0})
        return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {'buy': 0, 'sell': 0, 'hold': 0}
        
        current_q = self.q_table[state][action]
        next_q = max(self.q_table.get(next_state, {'buy': 0, 'sell': 0, 'hold': 0}).values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state][action] = new_q

    def optimize_portfolio(self, data):
        returns = {}
        for asset in self.assets:
            asset_without_hyphen = asset.replace('-', '')
            if asset_without_hyphen in data.columns:
                returns[asset] = data[asset_without_hyphen].pct_change().dropna()

        if not returns:
            logger.warning("No valid return data for optimization")
            return

        returns_df = pd.DataFrame(returns)

        mu = returns_df.mean()
        Sigma = returns_df.cov()

        def objective(weights):
            portfolio_return = np.sum(mu * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            return -portfolio_return / portfolio_volatility

        n_assets = len(self.assets)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))

        result = minimize(objective, n_assets*[1./n_assets], method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            self.optimal_weights = result.x
            logger.info(f"Optimal portfolio weights: {dict(zip(self.assets, self.optimal_weights))}")
        else:
            logger.warning("Portfolio optimization failed")

    def handle_data(self, data, portfolio):
        try:
            current_date = data.index[-1]
            portfolio_value = portfolio.cash

            for asset in self.assets:
                current_price = data[asset].iloc[-1]
                position = self.positions[asset]
                profit = 0  # Initialize profit to 0 at the start

                if position > 0:
                    # Calculate profit if there's an open position
                    buy_price = self.buy_prices[asset]
                    profit = (current_price - buy_price) / buy_price


                asset_without_hyphen = asset.replace('-', '')
                if asset_without_hyphen not in data.columns:
                    logger.error(f"Required column '{asset_without_hyphen}' not found in data")
                    continue

                asset_data = data[asset_without_hyphen].to_frame(name='close')
                current_price = asset_data['close'].iloc[-1]
                history = asset_data.iloc[-90:]

                # Check for loss and update consecutive loss counter
                if self.check_for_loss(asset, current_price):
                    self.consecutive_losses[asset] += 1
                    if self.consecutive_losses[asset] >= self.loss_threshold:
                        self.retrain_model(asset, data)
                else:
                    self.consecutive_losses[asset] = 0  # Reset counter if not a loss

                state = self.get_state(history)
                action = self.get_action(state)

                reward = 0
                next_state = None

                if action == 'buy' and portfolio.cash > current_price:
                    shares_to_buy = min(portfolio.cash // current_price, 0.1 * portfolio.portfolio_value // current_price)
                    if shares_to_buy > 0:
                        self.positions[asset] += shares_to_buy
                        portfolio.cash -= shares_to_buy * current_price
                        self.buy_prices[asset] = current_price
                        logger.info(f"BUY ORDER: {asset} - {shares_to_buy:.2f} units at {current_price:.2f}")
                        send_discord_message("Buy", f"BUY EXECUTED: {asset} - Price: {current_price:.2f}, Amount: {shares_to_buy:.2f}", "")
                        self.trade_history[asset].append({
                            'date': current_date,
                            'action': 'buy',
                            'price': current_price,
                            'quantity': shares_to_buy
                        })

                elif action == 'sell' and self.positions[asset] > 0:
                    shares_to_sell = self.positions[asset]
                    self.positions[asset] = 0
                    portfolio.cash += shares_to_sell * current_price
                    logger.info(f"SELL ORDER: {asset} - {shares_to_sell:.2f} units at {current_price:.2f}")
                    send_discord_message("Sell", f"SELL EXECUTED: {asset} - Price: {current_price:.2f}, Amount: {shares_to_sell:.2f}", "")
                    self.trade_history[asset].append({
                        'date': current_date,
                        'action': 'sell',
                        'price': current_price,
                        'quantity': shares_to_sell
                    })
                    self.buy_prices[asset] = None

                    reward = profit

                # Check for stop loss and take profit
                if self.positions[asset] > 0:
                    unrealized_profit = (current_price - self.buy_prices[asset]) / self.buy_prices[asset]
                    if unrealized_profit <= -self.stop_loss or unrealized_profit >= self.take_profit:
                        shares_to_sell = self.positions[asset]
                        self.positions[asset] = 0
                        profit = (current_price - self.buy_prices[asset]) * shares_to_sell
                        portfolio.cash += shares_to_sell * current_price
                        logger.info(f"{'STOP LOSS' if unrealized_profit <= -self.stop_loss else 'TAKE PROFIT'}: {asset} - {shares_to_sell:.2f} units at {current_price:.2f}")
                        logger.info(f"PROFIT: ${profit:.2f}")
                        send_discord_message("Sell", f"{'STOP LOSS' if unrealized_profit <= -self.stop_loss else 'TAKE PROFIT'}: {asset} - Price: {current_price:.2f}, Amount: {shares_to_sell:.2f}", "")
                        send_discord_message("Profit", f"PROFIT: ${profit:.2f}", f"<@&1266514940274020445>\n{profit:.2f}$" if profit <= 0 else f"<@&1266514890785292379>\n{profit:.2f}$")
                        
                        self.trade_history[asset].append({
                            'date': current_date,
                            'action': 'STOP LOSS' if unrealized_profit <= -self.stop_loss else 'TAKE PROFIT',
                            'price': current_price,
                            'amount': shares_to_sell,
                            'value': shares_to_sell * current_price,
                            'profit': profit
                        })
                        reward = profit

                next_state = self.get_state(history)
                self.trade_memory.add_trade(state, action, reward, next_state, False)
                self.update_q_table(state, action, reward, next_state)

                portfolio_value += self.positions[asset] * current_price

            self.portfolio_value_history.append((current_date, portfolio_value))
            portfolio.portfolio_value = portfolio_value
            
            logger.info(f"Date: {current_date}, Portfolio Value: {portfolio_value}")

            # Check for maximum drawdown
            max_portfolio_value = max(value for _, value in self.portfolio_value_history)
            current_drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
            if current_drawdown > self.max_drawdown:
                logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.2%}. Pausing trading.")
                return

            # Learn from past experiences
            if self.trade_memory.size() > self.trade_memory.batch_size:
                experiences = self.trade_memory.sample()
                for state, action, reward, next_state, _ in experiences:
                    self.update_q_table(state, action, reward, next_state)

            # Decay epsilon and learning rate
            self.epsilon = max(0.01, self.epsilon * 0.995)
            self.learning_rate = max(0.01, self.learning_rate * 0.995)

            portfolio.update_portfolio_value([self.positions[asset] * data[asset].iloc[-1] for asset in self.assets])
            self.portfolio_value_history.append(portfolio.portfolio_value)

            # Optimize portfolio weights periodically
            if len(self.portfolio_value_history) % 30 == 0:
                self.optimize_portfolio(data)

        except Exception as e:
            logger.error(f"An error occurred in handle_data(): {str(e)}")
            logger.error(traceback.format_exc())

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        if len(self.portfolio_value_history) < 2:
            return None
        
        returns = np.array([(b[1] - a[1]) / a[1] for a, b in zip(self.portfolio_value_history[:-1], self.portfolio_value_history[1:])])
        excess_returns = returns - (risk_free_rate / 252)  # Assuming 252 trading days in a year
        
        if np.std(excess_returns) == 0:
            return None
        
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio

    def calculate_sortino_ratio(self, risk_free_rate=0.02, target_return=0):
        if len(self.portfolio_value_history) < 2:
            return None
        
        returns = np.array([(b[1] - a[1]) / a[1] for a, b in zip(self.portfolio_value_history[:-1], self.portfolio_value_history[1:])])
        excess_returns = returns - (risk_free_rate / 252)  # Assuming 252 trading days in a year
        
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return None
        
        sortino_ratio = np.sqrt(252) * (np.mean(excess_returns) - target_return) / np.std(downside_returns)
        return sortino_ratio

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_max_drawdown(self):
        if len(self.portfolio_value_history) < 2:
            return None
        
        values = np.array([value for _, value in self.portfolio_value_history])
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)
        return max_drawdown

    def generate_performance_report(self):
        report = "Performance Report\n"
        total_profit = 0
        total_trades = 0
        profitable_trades = 0

        for asset in self.assets:
            asset_trades = len(self.trade_history[asset])
            asset_profitable_trades = sum(1 for trade in self.trade_history[asset] if trade.get('profit', 0) > 0)
            asset_total_profit = sum(trade.get('profit', 0) for trade in self.trade_history[asset])
            
            report += f"\nAsset: {asset}\n"
            report += f"Total Trades: {asset_trades}\n"
            report += f"Profitable Trades: {asset_profitable_trades}\n"
            
            if asset_trades > 0:
                win_rate = (asset_profitable_trades / asset_trades * 100)
                report += f"Win Rate: {win_rate:.2f}%\n"
            else:
                report += "Win Rate: N/A (No trades executed)\n"
            
            report += f"Total Profit: ${asset_total_profit:.2f}\n"

            total_profit += asset_total_profit
            total_trades += asset_trades
            profitable_trades += asset_profitable_trades

        report += f"\nOverall Performance\n"
        report += f"Total Trades: {total_trades}\n"
        report += f"Profitable Trades: {profitable_trades}\n"
        
        if total_trades > 0:
            overall_win_rate = (profitable_trades / total_trades * 100)
            report += f"Overall Win Rate: {overall_win_rate:.2f}%\n"
        else:
            report += "Overall Win Rate: N/A (No trades executed)\n"
        
        report += f"Total Profit: ${total_profit:.2f}\n"
        
        if self.portfolio_value_history:
            initial_value = self.portfolio_value_history[0][1]
            final_value = self.portfolio_value_history[-1][1]
            total_return = (final_value - initial_value) / initial_value * 100
            report += f"Total Return: {total_return:.2f}%\n"

        # Calculate and add new metrics
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio()
        max_drawdown = self.calculate_max_drawdown()

        report += f"Sharpe Ratio: {sharpe_ratio:.4f}\n" if sharpe_ratio is not None else "Sharpe Ratio: N/A\n"
        report += f"Sortino Ratio: {sortino_ratio:.4f}\n" if sortino_ratio is not None else "Sortino Ratio: N/A\n"
        report += f"Maximum Drawdown: {max_drawdown:.2%}\n" if max_drawdown is not None else "Maximum Drawdown: N/A\n"

        return report
    
    def plot_portfolio_value(self):
        if not self.portfolio_value_history:
            logger.warning("No portfolio value history to plot.")
            return
        
        dates, values = zip(*self.portfolio_value_history)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('portfolio_value.png')
        plt.close()
        
    def plot_trade_history(self):
        plt.figure(figsize=(12, 6))
        for asset in self.assets:
            buy_trades = [trade for trade in self.trade_history[asset] if trade['action'] == 'BUY']
            sell_trades = [trade for trade in self.trade_history[asset] if trade['action'] == 'SELL']

            plt.scatter([trade['date'] for trade in buy_trades], [trade['price'] for trade in buy_trades], 
                        label=f'{asset} Buy', marker='^')
            plt.scatter([trade['date'] for trade in sell_trades], [trade['price'] for trade in sell_trades], 
                        label=f'{asset} Sell', marker='v')
        
        plt.title('Trade History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('trade_history.png')
        plt.close()

def send_discord_message(message_type, content, mention_content=""):
    time.sleep(2)  # 2-second cooldown
    try:
        webhook = DiscordWebhook(url=discordwebhookurl)
        embed = DiscordEmbed(title=f"{message_type} Alert", description=content, color='03b2f8')
        
        if mention_content:
            webhook.content = mention_content
        
        webhook.add_embed(embed)
        response = webhook.execute()
        
        if response.status_code == 200:
            logger.info(f"Discord message sent successfully: {message_type}")
        else:
            logger.error(f"Failed to send Discord message. Status code: {response.status_code}")
    
    except Exception as e:
        logger.exception(f"An error occurred while sending Discord message: {e}")

def send_discord_graph(title1, url1):
    webhook = DiscordWebhook(url=discordwebhookurl)
    embed1 = DiscordEmbed(title=title1,rate_limit_retry=True,color="03b2f8")
    embed1.set_image(url1)
    webhook.add_embed(embed1)  
    response = webhook.execute()
    logger.info(f"Discord webhook response: {response}")

if __name__ == "__main__":
    try:
        start = pd.Timestamp('2018-01-11', tz='UTC')
        end = pd.Timestamp.now(tz='UTC')

        assets = ['BTC-CAD', 'ETH-CAD', 'XRP-CAD']  # Add more assets as needed
        data = yf.download(assets, start=start, end=end)['Close']  # Download closing prices for all assets
        
        # Rename columns to remove hyphens
        data.columns = [col.replace('-', '') for col in data.columns]

        # Create instances
        portfolio = Portfolio(initial_cash=10000) # Set this to your desired initial cash amount
        
        # Create a new TradingBot instance with the updated asset names
        bot = TradingBot([asset.replace('-', '') for asset in assets])
        
        # Perform hyperparameter tuning
        bot.tune_hyperparameters(data, portfolio)
        
        # Train the ML models
        for asset in bot.assets:
            asset_data = data[asset.replace('-', '')]
            bot.ml_models[asset].train(asset_data)

        for i in range(90, len(data)):
            daily_data = data.iloc[i-90:i+1]
            bot.handle_data(daily_data, portfolio)

        performance_report = bot.generate_performance_report()
        logger.info(performance_report)

        # Split the performance report into chunks if it's too long for a single Discord message
        max_discord_length = 2000
        report_chunks = [performance_report[i:i+max_discord_length] for i in range(0, len(performance_report), max_discord_length)]
        
        for i, chunk in enumerate(report_chunks):
            send_discord_message(f"Performance Report (Part {i+1})", chunk, "")
        
        bot.plot_portfolio_value()
        bot.plot_trade_history()

        # Upload and send the graphs
        im = pyimgur.Imgur(CLIENT_ID)
        portfolio_value_image = im.upload_image('portfolio_value.png', title="Portfolio Value Over Time")
        trade_history_image = im.upload_image('trade_history.png', title="Trade History")

        send_discord_graph("Portfolio Value Graph", portfolio_value_image.link)
        send_discord_graph("Trade History Graph", trade_history_image.link)

        logger.info(f"Portfolio Value Graph: {portfolio_value_image.link}")
        logger.info(f"Trade History Graph: {trade_history_image.link}")

        logger.info("Backtest completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during the backtesting: {e}")
        
        pass
