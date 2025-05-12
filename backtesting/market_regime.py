"""
Market Regime Detection Module for Advanced Backtesting Engine

This module provides market regime detection capabilities for the backtesting engine,
allowing for adaptive strategy selection based on market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Try to import optional dependencies
try:
    import hmmlearn.hmm as hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logging.warning("hmmlearn not installed. Install with 'pip install hmmlearn' for HMM-based regime detection.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("tensorflow not installed. Install with 'pip install tensorflow' for deep learning-based regime detection.")

# Configure logging
logger = logging.getLogger('market_regime')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MarketRegimeDetector:
    """
    Market regime detector for identifying different market conditions
    and enabling adaptive strategy selection.
    """
    
    def __init__(self, market_data: Dict[str, Any] = None, 
                features: Optional[List[str]] = None,
                n_regimes: int = 3,
                lookback_window: int = 20):
        """
        Initialize the market regime detector.
        
        Args:
            market_data: Dictionary of market data by instrument
            features: List of features to use for regime detection
            n_regimes: Number of regimes to detect
            lookback_window: Window size for feature calculation
        """
        self.market_data = market_data
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.regime_labels = None
        self.feature_data = None
        
        # Set default features if not provided
        if features is None:
            self.features = [
                'returns', 'volatility', 'rsi', 'ema_ratio', 
                'bb_width', 'volume_ratio', 'macd'
            ]
        else:
            self.features = features
    
    def prepare_data(self, primary_instrument: str = None) -> pd.DataFrame:
        """
        Prepare data for regime detection by calculating features.
        
        Args:
            primary_instrument: Primary instrument to use for regime detection
            
        Returns:
            DataFrame with calculated features
        """
        if self.market_data is None or not self.market_data:
            logger.error("No market data available")
            return None
        
        # If primary instrument is not specified, use the first one
        if primary_instrument is None:
            primary_instrument = list(self.market_data.keys())[0]
        
        if primary_instrument not in self.market_data:
            logger.error(f"Primary instrument {primary_instrument} not found in market data")
            return None
        
        # Get data for primary instrument
        data = self.market_data[primary_instrument]
        
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and data and isinstance(data[0], dict):
                data = pd.DataFrame(data)
            else:
                logger.error(f"Unsupported data format for {primary_instrument}")
                return None
        
        # Ensure we have the necessary columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Calculate features
        feature_data = pd.DataFrame(index=data.index)
        
        # Returns
        if 'returns' in self.features:
            feature_data['returns'] = data['close'].pct_change()
        
        # Volatility (standard deviation of returns)
        if 'volatility' in self.features:
            feature_data['volatility'] = data['close'].pct_change().rolling(self.lookback_window).std()
        
        # RSI (Relative Strength Index)
        if 'rsi' in self.features:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(self.lookback_window).mean()
            loss = -delta.where(delta < 0, 0).rolling(self.lookback_window).mean()
            rs = gain / loss
            feature_data['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA ratio (short-term EMA / long-term EMA)
        if 'ema_ratio' in self.features:
            short_ema = data['close'].ewm(span=self.lookback_window // 2).mean()
            long_ema = data['close'].ewm(span=self.lookback_window).mean()
            feature_data['ema_ratio'] = short_ema / long_ema
        
        # Bollinger Band width
        if 'bb_width' in self.features:
            rolling_mean = data['close'].rolling(self.lookback_window).mean()
            rolling_std = data['close'].rolling(self.lookback_window).std()
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            feature_data['bb_width'] = (upper_band - lower_band) / rolling_mean
        
        # Volume ratio (current volume / average volume)
        if 'volume_ratio' in self.features:
            feature_data['volume_ratio'] = data['volume'] / data['volume'].rolling(self.lookback_window).mean()
        
        # MACD (Moving Average Convergence Divergence)
        if 'macd' in self.features:
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            feature_data['macd'] = ema12 - ema26
        
        # Add cross-asset features if available
        if len(self.market_data) > 1:
            # Correlation with other assets
            if 'correlation' in self.features:
                correlations = []
                primary_returns = data['close'].pct_change()
                
                for instrument, inst_data in self.market_data.items():
                    if instrument == primary_instrument:
                        continue
                    
                    if isinstance(inst_data, pd.DataFrame) and 'close' in inst_data.columns:
                        other_returns = inst_data['close'].pct_change()
                        
                        # Align the series
                        aligned_returns = pd.concat([primary_returns, other_returns], axis=1).dropna()
                        
                        if len(aligned_returns) > self.lookback_window:
                            # Calculate rolling correlation
                            rolling_corr = aligned_returns.iloc[:, 0].rolling(self.lookback_window).corr(aligned_returns.iloc[:, 1])
                            correlations.append(rolling_corr)
                
                if correlations:
                    # Average correlation with other assets
                    feature_data['correlation'] = pd.concat(correlations, axis=1).mean(axis=1)
        
        # Drop rows with NaN values
        feature_data = feature_data.dropna()
        
        # Store the feature data
        self.feature_data = feature_data
        
        logger.info(f"Prepared {len(feature_data)} data points with {len(feature_data.columns)} features")
        
        return feature_data
    
    def train(self, method: str = 'kmeans', n_components: int = 2, random_state: int = 42) -> Any:
        """
        Train regime detection model using specified method.
        
        Args:
            method: Method to use ('kmeans', 'gmm', 'hmm', 'pca_kmeans')
            n_components: Number of PCA components to use (for dimensionality reduction)
            random_state: Random seed for reproducibility
            
        Returns:
            Trained model
        """
        if self.feature_data is None or len(self.feature_data) == 0:
            logger.error("No feature data available. Call prepare_data() first.")
            return None
        
        # Scale the features
        X = self.scaler.fit_transform(self.feature_data)
        
        # Apply PCA for dimensionality reduction if needed
        if method == 'pca_kmeans' or (len(self.feature_data.columns) > n_components and n_components > 0):
            self.pca = PCA(n_components=n_components, random_state=random_state)
            X = self.pca.fit_transform(X)
            logger.info(f"Applied PCA, explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        # Train the model based on the selected method
        if method == 'kmeans' or method == 'pca_kmeans':
            model = KMeans(n_clusters=self.n_regimes, random_state=random_state)
            self.regime_labels = model.fit_predict(X)
            self.model = model
            
            logger.info(f"Trained KMeans model with {self.n_regimes} regimes")
        
        elif method == 'gmm':
            model = GaussianMixture(n_components=self.n_regimes, random_state=random_state)
            self.regime_labels = model.fit_predict(X)
            self.model = model
            
            logger.info(f"Trained Gaussian Mixture Model with {self.n_regimes} regimes")
        
        elif method == 'hmm':
            if not HMMLEARN_AVAILABLE:
                logger.error("hmmlearn is required for HMM-based regime detection")
                return None
            
            # HMM requires 2D data with shape (n_samples, n_features)
            model = hmm.GaussianHMM(n_components=self.n_regimes, random_state=random_state)
            model.fit(X)
            self.regime_labels = model.predict(X)
            self.model = model
            
            logger.info(f"Trained Hidden Markov Model with {self.n_regimes} regimes")
        
        else:
            logger.error(f"Unknown method: {method}")
            return None
        
        # Analyze the regimes
        self._analyze_regimes()
        
        return self.model
    
    def _analyze_regimes(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze the characteristics of each detected regime.
        
        Returns:
            Dictionary with regime characteristics
        """
        if self.regime_labels is None or self.feature_data is None:
            logger.error("No regime labels or feature data available")
            return {}
        
        # Create a DataFrame with features and regime labels
        regime_df = self.feature_data.copy()
        regime_df['regime'] = self.regime_labels
        
        # Calculate statistics for each regime
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            regime_data = regime_df[regime_df['regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate statistics
            stats = {}
            for feature in self.features:
                if feature in regime_data.columns:
                    stats[f"{feature}_mean"] = regime_data[feature].mean()
                    stats[f"{feature}_std"] = regime_data[feature].std()
            
            # Count occurrences and percentage
            stats['count'] = len(regime_data)
            stats['percentage'] = len(regime_data) / len(regime_df) * 100
            
            regime_stats[regime] = stats
            
            logger.info(f"Regime {regime}: {stats['count']} samples ({stats['percentage']:.2f}%)")
            
            # Log key characteristics
            if 'returns_mean' in stats:
                logger.info(f"  Average returns: {stats['returns_mean']:.4f}")
            if 'volatility_mean' in stats:
                logger.info(f"  Average volatility: {stats['volatility_mean']:.4f}")
        
        self.regime_stats = regime_stats
        return regime_stats
    
    def predict_regime(self, data: Union[pd.DataFrame, Dict[str, Any], np.ndarray]) -> int:
        """
        Predict market regime for given data.
        
        Args:
            data: Data to predict regime for (DataFrame, dict, or numpy array)
            
        Returns:
            Predicted regime label
        """
        if self.model is None:
            logger.error("No trained model available")
            return -1
        
        # Convert data to the right format
        if isinstance(data, pd.DataFrame):
            # Use as is
            features = data
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            features = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            # Convert numpy array to DataFrame
            if data.ndim == 1:
                features = pd.DataFrame([data], columns=self.features[:len(data)])
            else:
                features = pd.DataFrame(data, columns=self.features[:data.shape[1]])
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            return -1
        
        # Ensure we have the right features
        missing_features = [f for f in self.features if f not in features.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return -1
        
        # Scale the features
        X = self.scaler.transform(features[self.features])
        
        # Apply PCA if used during training
        if self.pca is not None:
            X = self.pca.transform(X)
        
        # Predict regime
        if hasattr(self.model, 'predict'):
            regime = self.model.predict(X)[0]
        else:
            logger.error("Model does not have predict method")
            return -1
        
        return regime
    
    def backtest_with_regime_adaptation(self, backtest_engine, 
                                       strategy_mapping: Dict[int, str],
                                       start_time: Optional[Union[int, str]] = None,
                                       end_time: Optional[Union[int, str]] = None,
                                       params_mapping: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run backtest with adaptive strategy selection based on regime.
        
        Args:
            backtest_engine: The backtesting engine to use
            strategy_mapping: Dictionary mapping regime labels to strategy names
            start_time: Start time for backtest
            end_time: End time for backtest
            params_mapping: Optional dictionary mapping regime labels to strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        if self.model is None or self.regime_labels is None:
            logger.error("No trained model or regime labels available")
            return None
        
        if not strategy_mapping:
            logger.error("No strategy mapping provided")
            return None
        
        logger.info(f"Running adaptive backtest with {len(strategy_mapping)} strategies")
        
        # Get market data from backtest engine
        if hasattr(backtest_engine, 'market_data'):
            market_data = backtest_engine.market_data
        else:
            logger.error("Backtest engine does not have market_data attribute")
            return None
        
        # Get signals from backtest engine
        if hasattr(backtest_engine, 'signals'):
            signals = backtest_engine.signals
        else:
            logger.error("Backtest engine does not have signals attribute")
            return None
        
        # Create a copy of the backtest engine to avoid modifying the original
        engine_copy = copy.deepcopy(backtest_engine)
        
        # Prepare data for regime detection if not already done
        if self.feature_data is None:
            self.prepare_data()
        
        # Get regime labels for each timestamp
        regime_df = pd.DataFrame({'regime': self.regime_labels}, index=self.feature_data.index)
        
        # Initialize results
        results = {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0,
            'equity_curve': [],
            'drawdown_curve': [],
            'regime_changes': [],
            'regime_performance': {}
        }
        
        # Initialize tracking variables
        current_equity = 100.0  # Start with 100 units
        equity_curve = [current_equity]
        max_equity = current_equity
        drawdown_curve = [0.0]
        current_regime = None
        current_strategy = None
        regime_changes = []
        
        # Track performance by regime
        regime_performance = {regime: {'trades': 0, 'wins': 0, 'losses': 0, 'total_return': 0.0} 
                             for regime in strategy_mapping.keys()}
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.get('timestamp', 0))
        
        # Process each signal
        for signal in sorted_signals:
            timestamp = signal.get('timestamp', 0)
            
            # Skip signals outside the specified time range
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            
            # Determine the current regime
            regime = self._get_regime_at_timestamp(timestamp, regime_df)
            
            if regime is None:
                # No regime information available for this timestamp
                continue
            
            # Check if regime has changed
            if regime != current_regime:
                if current_regime is not None:
                    # Log regime change
                    regime_change = {
                        'timestamp': timestamp,
                        'from_regime': current_regime,
                        'to_regime': regime,
                        'from_strategy': current_strategy,
                        'to_strategy': strategy_mapping.get(regime, 'unknown')
                    }
                    regime_changes.append(regime_change)
                    
                    logger.info(f"Regime change at {timestamp}: {current_regime} -> {regime}")
                
                current_regime = regime
                current_strategy = strategy_mapping.get(regime, None)
                
                if current_strategy is None:
                    # No strategy mapped to this regime
                    logger.warning(f"No strategy mapped to regime {regime}")
                    continue
            
            # Skip if no strategy for current regime
            if current_strategy is None:
                continue
            
            # Get parameters for current regime if available
            params = None
            if params_mapping is not None and regime in params_mapping:
                params = params_mapping[regime]
            
            # Execute the strategy for this signal
            trade_result = self._execute_strategy(engine_copy, current_strategy, signal, params)
            
            if trade_result:
                # Update equity
                pnl = trade_result.get('pnl', 0.0)
                current_equity *= (1 + pnl / 100.0)
                equity_curve.append(current_equity)
                
                # Update max equity and drawdown
                max_equity = max(max_equity, current_equity)
                drawdown = (max_equity - current_equity) / max_equity
                drawdown_curve.append(drawdown)
                
                # Update regime performance
                regime_performance[regime]['trades'] += 1
                if pnl > 0:
                    regime_performance[regime]['wins'] += 1
                else:
                    regime_performance[regime]['losses'] += 1
                regime_performance[regime]['total_return'] += pnl
        
        # Calculate final results
        if len(equity_curve) > 1:
            results['total_return'] = (equity_curve[-1] / equity_curve[0] - 1) * 100
            results['max_drawdown'] = max(drawdown_curve) * 100
            results['equity_curve'] = equity_curve
            results['drawdown_curve'] = drawdown_curve
            results['regime_changes'] = regime_changes
            
            # Calculate Sharpe and Sortino ratios
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            if len(returns) > 0:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    results['sharpe_ratio'] = avg_return / std_return * np.sqrt(252)  # Annualized
                
                # Sortino ratio uses only negative returns for denominator
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    std_negative = np.std(negative_returns)
                    if std_negative > 0:
                        results['sortino_ratio'] = avg_return / std_negative * np.sqrt(252)  # Annualized
            
            # Calculate win rate and profit factor
            total_trades = sum(perf['trades'] for perf in regime_performance.values())
            total_wins = sum(perf['wins'] for perf in regime_performance.values())
            
            if total_trades > 0:
                results['win_rate'] = total_wins / total_trades * 100
                results['num_trades'] = total_trades
            
            # Calculate profit factor
            total_gains = sum(max(0, perf['total_return']) for perf in regime_performance.values())
            total_losses = sum(max(0, -perf['total_return']) for perf in regime_performance.values())
            
            if total_losses > 0:
                results['profit_factor'] = total_gains / total_losses
            
            # Add regime performance
            results['regime_performance'] = regime_performance
            
            logger.info(f"Adaptive backtest completed with {total_trades} trades")
            logger.info(f"Total return: {results['total_return']:.2f}%")
            logger.info(f"Max drawdown: {results['max_drawdown']:.2f}%")
            logger.info(f"Win rate: {results['win_rate']:.2f}%")
            logger.info(f"Regime changes: {len(regime_changes)}")
        
        return results
    
    def _get_regime_at_timestamp(self, timestamp, regime_df):
        """
        Get the regime label for a specific timestamp.
        
        Args:
            timestamp: Timestamp to get regime for
            regime_df: DataFrame with regime labels
            
        Returns:
            Regime label or None if not available
        """
        # Convert timestamp to datetime if it's a number
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
            except ValueError:
                logger.error(f"Invalid timestamp format: {timestamp}")
                return None
        else:
            logger.error(f"Unsupported timestamp type: {type(timestamp)}")
            return None
        
        # Find the closest timestamp in the regime_df
        try:
            # If index is datetime
            if isinstance(regime_df.index, pd.DatetimeIndex):
                closest_idx = regime_df.index[regime_df.index.get_indexer([dt], method='nearest')[0]]
            else:
                # If index is numeric, find the closest value
                closest_idx = regime_df.index[np.abs(regime_df.index - timestamp).argmin()]
            
            return regime_df.loc[closest_idx, 'regime']
        except:
            return None
    
    def _execute_strategy(self, engine, strategy_name, signal, params=None):
        """
        Execute a strategy for a specific signal.
        
        Args:
            engine: Backtest engine
            strategy_name: Name of the strategy to execute
            signal: Signal to process
            params: Optional strategy parameters
            
        Returns:
            Trade result or None if execution failed
        """
        # This is a simplified implementation
        # In a real system, you would use the backtest engine's API to execute the strategy
        
        # Check if the strategy exists
        if not hasattr(engine, 'strategies') or strategy_name not in engine.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return None
        
        # Get the strategy
        strategy = engine.strategies[strategy_name]
        
        # Execute the strategy
        try:
            # This assumes the strategy has a process_signal method
            if hasattr(strategy, 'process_signal'):
                return strategy.process_signal(signal, params)
            else:
                logger.error(f"Strategy {strategy_name} does not have process_signal method")
                return None
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
            return None
    
    def plot_regimes(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot detected regimes over time.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.regime_labels is None or self.feature_data is None:
            logger.error("No regime labels or feature data available")
            return None
        
        # Create a DataFrame with regime labels
        regime_df = pd.DataFrame({'regime': self.regime_labels}, index=self.feature_data.index)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot a representative feature (e.g., returns or close price)
        if 'returns' in self.feature_data.columns:
            # Calculate cumulative returns
            cum_returns = (1 + self.feature_data['returns']).cumprod()
            ax1.plot(cum_returns, color='black', linewidth=1)
            ax1.set_ylabel('Cumulative Returns')
        elif 'close' in self.feature_data.columns:
            ax1.plot(self.feature_data['close'], color='black', linewidth=1)
            ax1.set_ylabel('Close Price')
        else:
            # Use the first available feature
            feature = self.feature_data.columns[0]
            ax1.plot(self.feature_data[feature], color='black', linewidth=1)
            ax1.set_ylabel(feature)
        
        # Color the background based on regimes
        for regime in range(self.n_regimes):
            regime_periods = regime_df[regime_df['regime'] == regime]
            
            if len(regime_periods) == 0:
                continue
            
            # Find contiguous periods
            regime_periods['date'] = regime_periods.index
            regime_periods['group'] = (regime_periods['date'].diff() > pd.Timedelta(days=1)).cumsum()
            
            for _, group in regime_periods.groupby('group'):
                if len(group) > 0:
                    start = group.index[0]
                    end = group.index[-1]
                    
                    # Add colored background
                    ax1.axvspan(start, end, alpha=0.2, color=f'C{regime}')
        
        # Add regime labels at the bottom
        ax2.scatter(regime_df.index, regime_df['regime'], c=regime_df['regime'], cmap='viridis', 
                   marker='s', s=10)
        
        # Set y-axis limits and labels
        ax2.set_ylim(-0.5, self.n_regimes - 0.5)
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_yticklabels([f'Regime {i}' for i in range(self.n_regimes)])
        
        # Add labels and title
        ax1.set_title('Market Regimes Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Regime')
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regime_characteristics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot characteristics of each regime.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, 'regime_stats') or not self.regime_stats:
            logger.error("No regime statistics available")
            return None
        
        # Select key features to plot
        key_features = ['returns_mean', 'volatility_mean', 'rsi_mean', 'volume_ratio_mean']
        available_features = []
        
        for feature in key_features:
            if all(feature in stats for stats in self.regime_stats.values()):
                available_features.append(feature)
        
        if not available_features:
            logger.error("No common features available across regimes")
            return None
        
        # Create the plot
        fig, axes = plt.subplots(len(available_features), 1, figsize=figsize)
        
        if len(available_features) == 1:
            axes = [axes]
        
        # Plot each feature
        for i, feature in enumerate(available_features):
            ax = axes[i]
            
            # Extract values for each regime
            regimes = list(self.regime_stats.keys())
            values = [self.regime_stats[r][feature] for r in regimes]
            
            # Create bar chart
            bars = ax.bar(regimes, values, color=[f'C{r}' for r in regimes])
            
            # Add labels
            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_xticks(regimes)
            ax.set_xticklabels([f'Regime {r}' for r in regimes])
            
            # Add values on top of bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{value:.4f}', ha='center', va='bottom')
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        # Add title
        fig.suptitle('Regime Characteristics')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and related data.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            logger.error("No trained model available to save")
            return
        
        # Prepare data for saving
        data = {
            'n_regimes': self.n_regimes,
            'features': self.features,
            'lookback_window': self.lookback_window,
            'regime_labels': self.regime_labels.tolist() if self.regime_labels is not None else None,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        }
        
        # Add PCA data if available
        if self.pca is not None:
            data['pca_components'] = self.pca.components_.tolist()
            data['pca_mean'] = self.pca.mean_.tolist()
            data['pca_explained_variance'] = self.pca.explained_variance_.tolist()
            data['pca_explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
        
        # Add regime statistics if available
        if hasattr(self, 'regime_stats'):
            data['regime_stats'] = self.regime_stats
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved model data to {filepath}")
        
        # For some models, we need to save them separately
        if hasattr(self.model, 'save'):
            model_path = filepath + '.model'
            self.model.save(model_path)
            logger.info(f"Saved model to {model_path}")
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model and related data.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update instance variables
            self.n_regimes = data.get('n_regimes', self.n_regimes)
            self.features = data.get('features', self.features)
            self.lookback_window = data.get('lookback_window', self.lookback_window)
            
            if 'regime_labels' in data and data['regime_labels'] is not None:
                self.regime_labels = np.array(data['regime_labels'])
            
            # Restore scaler
            if 'scaler_mean' in data and 'scaler_scale' in data:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(data['scaler_mean'])
                self.scaler.scale_ = np.array(data['scaler_scale'])
            
            # Restore PCA if available
            if all(k in data for k in ['pca_components', 'pca_mean', 'pca_explained_variance', 'pca_explained_variance_ratio']):
                n_components = len(data['pca_explained_variance'])
                self.pca = PCA(n_components=n_components)
                self.pca.components_ = np.array(data['pca_components'])
                self.pca.mean_ = np.array(data['pca_mean'])
                self.pca.explained_variance_ = np.array(data['pca_explained_variance'])
                self.pca.explained_variance_ratio_ = np.array(data['pca_explained_variance_ratio'])
            
            # Restore regime statistics if available
            if 'regime_stats' in data:
                self.regime_stats = data['regime_stats']
            
            logger.info(f"Loaded model data from {filepath}")
            
            # For some models, we need to load them separately
            model_path = filepath + '.model'
            if os.path.exists(model_path):
                if 'kmeans' in filepath:
                    self.model = KMeans(n_clusters=self.n_regimes)
                    # KMeans doesn't have a load method, so we need to initialize it
                    # and set its attributes manually
                elif 'gmm' in filepath:
                    self.model = GaussianMixture(n_components=self.n_regimes)
                    # Similar to KMeans
                elif 'hmm' in filepath and HMMLEARN_AVAILABLE:
                    self.model = hmm.GaussianHMM(n_components=self.n_regimes)
                    self.model = self.model.load(model_path)
                
                logger.info(f"Loaded model from {model_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False