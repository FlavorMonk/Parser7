#!/usr/bin/env python3
"""
Advanced Backtesting Engine

This module implements a robust backtesting engine for forex trading strategies
with a focus on prop firm challenge simulation, realistic execution modeling,
and comprehensive performance analysis.
"""

import logging
import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import local modules
from backtesting.rules_engine import PropFirmRulesEngine
from backtesting.execution_simulator import ExecutionSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger("backtest_engine")

class BacktestEngine:
    """
    Advanced backtesting engine for forex trading strategies.
    
    Features:
    - Prop firm rules enforcement
    - Realistic execution simulation
    - Rolling window backtesting
    - Comprehensive performance analysis
    - Multi-strategy support
    - Regime detection
    """
    
    def __init__(self, 
                 signals_file: Optional[str] = None,
                 market_data_dir: Optional[str] = None,
                 output_dir: str = "backtest_results",
                 prop_firm: str = "TFT",
                 account_size: float = 100000.0,
                 challenge_phase: str = "phase1",
                 execution_config: Optional[Dict] = None,
                 rules_config: Optional[Dict] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            signals_file: Path to signals JSON file
            market_data_dir: Directory containing market data files
            output_dir: Directory to save backtest results
            prop_firm: Prop firm type ("TFT", "FTMO", "MFF")
            account_size: Account size in base currency
            challenge_phase: Challenge phase ("phase1", "phase2", "verification", "funded")
            execution_config: Execution simulator configuration
            rules_config: Prop firm rules configuration
        """
        self.signals_file = signals_file
        self.market_data_dir = market_data_dir
        self.output_dir = output_dir
        self.prop_firm = prop_firm
        self.account_size = account_size
        self.challenge_phase = challenge_phase
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize rules engine
        self.rules_engine = PropFirmRulesEngine(
            firm_type=prop_firm,
            account_size=account_size,
            challenge_phase=challenge_phase,
            custom_rules=rules_config
        )
        
        # Initialize execution simulator
        self.execution_simulator = ExecutionSimulator(
            slippage_model="volatility",
            spread_model="dynamic",
            partial_fills_enabled=True,
            requotes_enabled=True,
            latency_model="random",
            custom_config=execution_config
        )
        
        # Initialize data containers
        self.signals = []
        self.market_data = {}
        self.strategies = {}
        self.current_positions = []
        self.closed_positions = []
        self.performance_metrics = {}
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Load data if provided
        if signals_file:
            self.load_signals(signals_file)
        
        if market_data_dir:
            self.load_market_data(market_data_dir)
        
        logger.info(f"Initialized backtest engine for {prop_firm} {challenge_phase} with ${account_size:.2f} account")
    
    def load_signals(self, signals_file: str) -> None:
        """
        Load trading signals from a JSON file.
        
        Args:
            signals_file: Path to signals JSON file
        """
        try:
            with open(signals_file, 'r') as f:
                data = json.load(f)
            
            # Handle different signal formats
            if isinstance(data, dict) and 'signals' in data:
                self.signals = data['signals']
            else:
                self.signals = data
            
            # Sort signals by timestamp
            if self.signals and 'timestamp' in self.signals[0]:
                self.signals.sort(key=lambda x: x.get('timestamp', 0))
            
            logger.info(f"Loaded {len(self.signals)} signals from {signals_file}")
        except Exception as e:
            logger.error(f"Error loading signals from {signals_file}: {str(e)}")
            self.signals = []
    
    def load_market_data(self, market_data_dir: str) -> None:
        """
        Load market data from directory.
        
        Args:
            market_data_dir: Directory containing market data files
        """
        try:
            # Get list of market data files
            files = [f for f in os.listdir(market_data_dir) if f.endswith('.json')]
            
            for file in files:
                # Extract instrument from filename
                instrument = os.path.splitext(file)[0].upper()
                
                # Load market data
                with open(os.path.join(market_data_dir, file), 'r') as f:
                    data = json.load(f)
                
                self.market_data[instrument] = data
            
            logger.info(f"Loaded market data for {len(self.market_data)} instruments from {market_data_dir}")
        except Exception as e:
            logger.error(f"Error loading market data from {market_data_dir}: {str(e)}")
            self.market_data = {}
    
    def register_strategy(self, name: str, strategy_class: Any, params: Optional[Dict] = None) -> None:
        """
        Register a trading strategy.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
            params: Strategy parameters
        """
        try:
            # Initialize strategy with parameters
            strategy = strategy_class(**(params or {}))
            
            # Register strategy
            self.strategies[name] = {
                'class': strategy_class,
                'instance': strategy,
                'params': params or {}
            }
            
            logger.info(f"Registered strategy: {name}")
        except Exception as e:
            logger.error(f"Error registering strategy {name}: {str(e)}")
    
    def prepare_market_data(self, instrument: str, start_time: float, end_time: float) -> Dict:
        """
        Prepare market data for a specific instrument and time range.
        
        Args:
            instrument: Instrument symbol
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Dictionary with prepared market data
        """
        # Normalize instrument format
        normalized_instrument = instrument.replace('/', '').upper()
        
        # Check if we have market data for this instrument
        if normalized_instrument not in self.market_data:
            logger.warning(f"No market data available for {normalized_instrument}")
            return None
        
        # Get market data
        data = self.market_data[normalized_instrument]
        
        # Extract relevant data points
        if 'data' in data:
            # Filter data by time range
            filtered_data = [
                point for point in data['data']
                if start_time <= point.get('timestamp', 0) <= end_time
            ]
            
            if not filtered_data:
                logger.warning(f"No market data available for {normalized_instrument} in the specified time range")
                return None
            
            # Extract OHLCV data
            timestamps = [point.get('timestamp', 0) for point in filtered_data]
            opens = [point.get('open', 0) for point in filtered_data]
            highs = [point.get('high', 0) for point in filtered_data]
            lows = [point.get('low', 0) for point in filtered_data]
            closes = [point.get('close', 0) for point in filtered_data]
            volumes = [point.get('volume', 0) for point in filtered_data]
            
            # Calculate additional metrics
            if len(closes) > 1:
                returns = [0] + [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            else:
                returns = [0]
                volatility = 0
            
            # Determine market direction
            if len(closes) > 20:
                short_ma = np.mean(closes[-10:])
                long_ma = np.mean(closes[-20:])
                
                if short_ma > long_ma:
                    market_direction = "up"
                elif short_ma < long_ma:
                    market_direction = "down"
                else:
                    market_direction = None
            else:
                market_direction = None
            
            # Create prepared market data
            prepared_data = {
                'instrument': normalized_instrument,
                'timestamps': timestamps,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes,
                'returns': returns,
                'volatility': volatility * 10000,  # Convert to pips
                'market_direction': market_direction,
                'current_price': closes[-1] if closes else None
            }
            
            return prepared_data
        
        logger.warning(f"Invalid market data format for {normalized_instrument}")
        return None
    
    def execute_trade(self, 
                     signal: Dict, 
                     strategy_name: str, 
                     timestamp: float) -> Dict:
        """
        Execute a trade based on a signal and strategy.
        
        Args:
            signal: Trading signal
            strategy_name: Strategy name
            timestamp: Current timestamp
            
        Returns:
            Dictionary with trade details
        """
        # Check if strategy exists
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return None
        
        # Get strategy instance
        strategy = self.strategies[strategy_name]['instance']
        
        # Extract signal details
        instrument = signal.get('asset') or signal.get('pair', '')
        direction = signal.get('direction', '').upper()
        entry_price = signal.get('entry')
        stop_loss = signal.get('stop_loss') or signal.get('sl')
        take_profit = signal.get('take_profit') or signal.get('tp')
        
        # Skip if missing required fields
        if not all([instrument, direction, entry_price, stop_loss, take_profit]):
            logger.warning(f"Skipping signal due to missing required fields: {signal}")
            return None
        
        # Normalize instrument format
        normalized_instrument = instrument.replace('/', '').upper()
        
        # Prepare market data
        market_data = self.prepare_market_data(
            normalized_instrument,
            timestamp - 86400,  # 24 hours before
            timestamp + 86400   # 24 hours after
        )
        
        if not market_data:
            logger.warning(f"Skipping signal due to missing market data: {signal}")
            return None
        
        # Check if we should enter trade (strategy-specific logic)
        if hasattr(strategy, 'should_enter_trade') and not strategy.should_enter_trade(signal, market_data):
            logger.info(f"Strategy {strategy_name} decided not to enter trade: {signal}")
            return None
        
        # Calculate position size (strategy-specific logic)
        position_size = 1.0  # Default
        if hasattr(strategy, 'calculate_position_size'):
            position_size = strategy.calculate_position_size(
                entry_price, stop_loss, normalized_instrument, signal.get('quality_score', 0.5)
            )
        
        # Create order
        order = {
            "order_id": f"{strategy_name}_{signal.get('id', random.randint(10000, 99999))}",
            "instrument": normalized_instrument,
            "type": "market",
            "direction": direction.lower(),
            "price": entry_price,
            "size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signal_id": signal.get('id', ''),
            "strategy": strategy_name,
            "timestamp": timestamp
        }
        
        # Simulate order execution
        execution_result = self.execution_simulator.simulate_fill(
            order,
            {
                "current_price": market_data['current_price'],
                "volatility": market_data['volatility'],
                "market_direction": market_data['market_direction']
            },
            datetime.fromtimestamp(timestamp)
        )
        
        # Check if order was filled
        if execution_result['status'] != 'filled':
            logger.info(f"Order not filled: {execution_result['status']} - {execution_result.get('rejection_reason', '')}")
            return None
        
        # Create position
        position = {
            "position_id": f"POS_{order['order_id']}",
            "instrument": normalized_instrument,
            "direction": direction.lower(),
            "entry_price": execution_result['fill_price'],
            "entry_time": timestamp,
            "size": execution_result['fill_size'],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signal_id": signal.get('id', ''),
            "strategy": strategy_name,
            "status": "open",
            "exit_price": None,
            "exit_time": None,
            "profit_loss": 0.0,
            "profit_loss_pips": 0.0,
            "exit_reason": None,
            "metadata": {
                "quality_score": signal.get('quality_score', 0.5),
                "slippage_pips": execution_result['slippage_pips'],
                "spread_pips": execution_result['spread_pips'],
                "latency_ms": execution_result['latency_ms'],
                "partially_filled": execution_result['partially_filled']
            }
        }
        
        # Add to current positions
        self.current_positions.append(position)
        
        logger.info(f"Opened position: {normalized_instrument} {direction} at {execution_result['fill_price']:.5f}")
        
        return position
    
    def update_positions(self, timestamp: float) -> None:
        """
        Update open positions based on market data at the given timestamp.
        
        Args:
            timestamp: Current timestamp
        """
        # Process each open position
        for position in list(self.current_positions):  # Use list() to allow removal during iteration
            # Get instrument and prepare market data
            instrument = position['instrument']
            market_data = self.prepare_market_data(
                instrument,
                timestamp - 3600,  # 1 hour before
                timestamp
            )
            
            if not market_data or not market_data['current_price']:
                logger.warning(f"Skipping position update due to missing market data: {position}")
                continue
            
            # Get current price
            current_price = market_data['current_price']
            
            # Check if stop loss or take profit hit
            if position['direction'] == 'buy':
                # Check stop loss
                if current_price <= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], timestamp, "stop_loss")
                    continue
                
                # Check take profit
                if current_price >= position['take_profit']:
                    self._close_position(position, position['take_profit'], timestamp, "take_profit")
                    continue
            else:  # sell
                # Check stop loss
                if current_price >= position['stop_loss']:
                    self._close_position(position, position['stop_loss'], timestamp, "stop_loss")
                    continue
                
                # Check take profit
                if current_price <= position['take_profit']:
                    self._close_position(position, position['take_profit'], timestamp, "take_profit")
                    continue
            
            # Check for dynamic exit (strategy-specific logic)
            if position['strategy'] in self.strategies:
                strategy = self.strategies[position['strategy']]['instance']
                
                if hasattr(strategy, 'should_exit_position'):
                    should_exit, exit_reason = strategy.should_exit_position(
                        position, market_data, timestamp
                    )
                    
                    if should_exit:
                        self._close_position(position, current_price, timestamp, f"dynamic_exit_{exit_reason}")
                        continue
                
                # Check for trailing stop (strategy-specific logic)
                if hasattr(strategy, 'update_trailing_stop'):
                    new_stop_loss = strategy.update_trailing_stop(
                        position, market_data, timestamp
                    )
                    
                    if new_stop_loss != position['stop_loss']:
                        logger.info(f"Updated trailing stop for {position['instrument']} {position['direction']}: {position['stop_loss']:.5f} -> {new_stop_loss:.5f}")
                        position['stop_loss'] = new_stop_loss
    
    def _close_position(self, 
                       position: Dict, 
                       exit_price: float, 
                       exit_time: float, 
                       exit_reason: str) -> None:
        """
        Close a position and record the result.
        
        Args:
            position: Position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Exit reason
        """
        # Calculate profit/loss
        if position['direction'] == 'buy':
            profit_loss = (exit_price - position['entry_price']) * position['size']
            profit_loss_pips = (exit_price - position['entry_price']) * 10000
        else:  # sell
            profit_loss = (position['entry_price'] - exit_price) * position['size']
            profit_loss_pips = (position['entry_price'] - exit_price) * 10000
        
        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = exit_time
        position['profit_loss'] = profit_loss
        position['profit_loss_pips'] = profit_loss_pips
        position['exit_reason'] = exit_reason
        position['status'] = 'closed'
        
        # Move to closed positions
        self.current_positions.remove(position)
        self.closed_positions.append(position)
        
        # Update rules engine
        self.rules_engine.record_trade(
            trade_id=position['position_id'],
            instrument=position['instrument'],
            direction=position['direction'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            position_size=position['size'],
            entry_time=datetime.fromtimestamp(position['entry_time']),
            exit_time=datetime.fromtimestamp(exit_time),
            profit_loss=profit_loss,
            profit_loss_pips=profit_loss_pips,
            trade_duration_minutes=(exit_time - position['entry_time']) / 60
        )
        
        # Check rules
        rules_violated, violation_reason = self.rules_engine.check_rules(datetime.fromtimestamp(exit_time))
        
        logger.info(f"Closed position: {position['instrument']} {position['direction']} at {exit_price:.5f} - P/L: ${profit_loss:.2f} ({profit_loss_pips:.1f} pips) - Reason: {exit_reason}")
        
        if rules_violated:
            logger.warning(f"Rules violated: {violation_reason}")
    
    def run_backtest(self, 
                    strategy_name: str, 
                    start_time: Optional[float] = None, 
                    end_time: Optional[float] = None,
                    max_positions: int = 5) -> Dict:
        """
        Run a backtest for a specific strategy.
        
        Args:
            strategy_name: Strategy name
            start_time: Start timestamp (default: first signal timestamp)
            end_time: End timestamp (default: last signal timestamp)
            max_positions: Maximum number of concurrent positions
            
        Returns:
            Dictionary with backtest results
        """
        # Check if strategy exists
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return None
        
        # Reset state
        self.rules_engine.reset()
        self.execution_simulator.reset_statistics()
        self.current_positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Determine time range
        if not self.signals:
            logger.error("No signals available for backtesting")
            return None
        
        if start_time is None:
            start_time = self.signals[0].get('timestamp', 0)
            # Convert string timestamp to float if needed
            if isinstance(start_time, str):
                try:
                    start_time = float(start_time)
                except ValueError:
                    # Try to parse as ISO format
                    try:
                        start_time = datetime.fromisoformat(start_time).timestamp()
                    except ValueError:
                        logger.error(f"Invalid timestamp format: {start_time}")
                        start_time = 0
        
        if end_time is None:
            end_time = self.signals[-1].get('timestamp', 0)
            # Convert string timestamp to float if needed
            if isinstance(end_time, str):
                try:
                    end_time = float(end_time)
                except ValueError:
                    # Try to parse as ISO format
                    try:
                        end_time = datetime.fromisoformat(end_time).timestamp()
                    except ValueError:
                        logger.error(f"Invalid timestamp format: {end_time}")
                        end_time = float(datetime.now().timestamp())
        
        # Filter signals by time range
        filtered_signals = []
        for signal in self.signals:
            timestamp = signal.get('timestamp', 0)
            # Convert string timestamp to float if needed
            if isinstance(timestamp, str):
                try:
                    timestamp = float(timestamp)
                except ValueError:
                    # Try to parse as ISO format
                    try:
                        timestamp = datetime.fromisoformat(timestamp).timestamp()
                    except ValueError:
                        logger.error(f"Invalid timestamp format: {timestamp}")
                        timestamp = 0
            
            if start_time <= timestamp <= end_time:
                # Update the timestamp in the signal
                signal_copy = signal.copy()
                signal_copy['timestamp'] = timestamp
                filtered_signals.append(signal_copy)
        
        if not filtered_signals:
            logger.error("No signals available in the specified time range")
            return None
        
        start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Running backtest for strategy {strategy_name} from {start_time_str} to {end_time_str} with {len(filtered_signals)} signals")
        
        # Process signals in chronological order
        for signal in filtered_signals:
            # Get signal timestamp
            timestamp = signal.get('timestamp', 0)
            
            # Update existing positions
            self.update_positions(timestamp)
            
            # Check if we can open a new position
            if len(self.current_positions) < max_positions and not self.rules_engine.state['challenge_failed']:
                # Execute trade
                self.execute_trade(signal, strategy_name, timestamp)
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.rules_engine.state['current_balance']
            })
            
            # Update drawdown curve
            self.drawdown_curve.append({
                'timestamp': timestamp,
                'drawdown': self.rules_engine.state['current_drawdown_pct']
            })
            
            # Check if challenge failed or passed
            if self.rules_engine.state['challenge_failed'] or self.rules_engine.state['challenge_passed']:
                break
        
        # Close any remaining positions at the end of the backtest
        for position in list(self.current_positions):
            # Get final market data
            market_data = self.prepare_market_data(
                position['instrument'],
                end_time - 3600,  # 1 hour before
                end_time
            )
            
            if market_data and market_data['current_price']:
                self._close_position(position, market_data['current_price'], end_time, "backtest_end")
            else:
                # Use last known price if market data not available
                self._close_position(position, position['entry_price'], end_time, "backtest_end")
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        # Get challenge status
        challenge_status = self.rules_engine.get_challenge_status()
        
        # Create backtest results
        backtest_results = {
            'strategy': strategy_name,
            'start_time': start_time,
            'end_time': end_time,
            'signals_count': len(filtered_signals),
            'trades_count': len(self.closed_positions),
            'performance_metrics': self.performance_metrics,
            'challenge_status': challenge_status,
            'equity_curve': [point['equity'] for point in self.equity_curve],
            'drawdown_curve': [point['drawdown'] for point in self.drawdown_curve],
            'execution_stats': self.execution_simulator.get_statistics()
        }
        
        # Save results
        self._save_backtest_results(backtest_results, strategy_name)
        
        logger.info(f"Backtest completed for strategy {strategy_name}")
        logger.info(f"Total Return: {self.performance_metrics['total_return'] * 100:.2f}%")
        logger.info(f"Max Drawdown: {self.performance_metrics['max_drawdown'] * 100:.2f}%")
        logger.info(f"Win Rate: {self.performance_metrics['win_rate'] * 100:.2f}%")
        logger.info(f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        logger.info(f"Challenge Result: {'PASSED' if challenge_status['challenge_passed'] else 'FAILED' if challenge_status['challenge_failed'] else 'INCOMPLETE'}")
        
        return backtest_results
    
    def run_rolling_window_backtest(self, 
                                   strategy_name: str, 
                                   window_size: int = 30, 
                                   step_size: int = 5,
                                   max_windows: int = 10,
                                   max_positions: int = 5) -> Dict:
        """
        Run a rolling window backtest for a specific strategy.
        
        Args:
            strategy_name: Strategy name
            window_size: Window size in days
            step_size: Step size in days
            max_windows: Maximum number of windows to test
            max_positions: Maximum number of concurrent positions
            
        Returns:
            Dictionary with rolling window backtest results
        """
        # Check if strategy exists
        if strategy_name not in self.strategies:
            logger.error(f"Strategy {strategy_name} not found")
            return None
        
        # Check if we have signals
        if not self.signals:
            logger.error("No signals available for backtesting")
            return None
        
        # Determine overall time range
        start_time = self.signals[0].get('timestamp', 0)
        end_time = self.signals[-1].get('timestamp', 0)
        
        # Calculate total duration in days
        total_days = (datetime.fromtimestamp(end_time) - datetime.fromtimestamp(start_time)).days
        
        # Check if we have enough data
        if total_days < window_size:
            logger.error(f"Not enough data for {window_size}-day window backtest (only {total_days} days available)")
            return None
        
        # Calculate number of windows
        num_windows = min(max_windows, (total_days - window_size) // step_size + 1)
        
        if num_windows <= 0:
            logger.error("No valid windows available for backtesting")
            return None
        
        logger.info(f"Running rolling window backtest for strategy {strategy_name} with {num_windows} windows of {window_size} days each")
        
        # Run backtest for each window
        window_results = []
        
        for i in range(num_windows):
            # Calculate window start and end times
            window_start = datetime.fromtimestamp(start_time) + timedelta(days=i * step_size)
            window_end = window_start + timedelta(days=window_size)
            
            window_start_time = window_start.timestamp()
            window_end_time = window_end.timestamp()
            
            logger.info(f"Window {i+1}/{num_windows}: {window_start.date()} to {window_end.date()}")
            
            # Run backtest for this window
            result = self.run_backtest(
                strategy_name,
                start_time=window_start_time,
                end_time=window_end_time,
                max_positions=max_positions
            )
            
            if result:
                # Add window information
                result['window'] = {
                    'index': i,
                    'start_time': window_start_time,
                    'end_time': window_end_time,
                    'start_date': window_start.date().isoformat(),
                    'end_date': window_end.date().isoformat()
                }
                
                window_results.append(result)
        
        # Calculate aggregate metrics
        if window_results:
            # Extract key metrics
            returns = [result['performance_metrics']['total_return'] for result in window_results]
            drawdowns = [result['performance_metrics']['max_drawdown'] for result in window_results]
            win_rates = [result['performance_metrics']['win_rate'] for result in window_results]
            profit_factors = [result['performance_metrics']['profit_factor'] for result in window_results]
            sharpe_ratios = [result['performance_metrics']['sharpe_ratio'] for result in window_results]
            sortino_ratios = [result['performance_metrics']['sortino_ratio'] for result in window_results]
            
            # Calculate pass rate
            pass_count = sum(1 for result in window_results if result['challenge_status']['challenge_passed'])
            pass_rate = pass_count / len(window_results)
            
            # Calculate failure reasons
            failure_reasons = {}
            for result in window_results:
                if result['challenge_status']['challenge_failed']:
                    reason = result['challenge_status']['failure_reason']
                    if reason:
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            # Create aggregate results
            aggregate_results = {
                'strategy': strategy_name,
                'window_size': window_size,
                'step_size': step_size,
                'num_windows': len(window_results),
                'pass_rate': pass_rate,
                'pass_count': pass_count,
                'failure_reasons': failure_reasons,
                'returns': {
                    'mean': np.mean(returns),
                    'median': np.median(returns),
                    'min': min(returns),
                    'max': max(returns),
                    'std': np.std(returns)
                },
                'drawdowns': {
                    'mean': np.mean(drawdowns),
                    'median': np.median(drawdowns),
                    'min': min(drawdowns),
                    'max': max(drawdowns),
                    'std': np.std(drawdowns)
                },
                'win_rates': {
                    'mean': np.mean(win_rates),
                    'median': np.median(win_rates),
                    'min': min(win_rates),
                    'max': max(win_rates),
                    'std': np.std(win_rates)
                },
                'profit_factors': {
                    'mean': np.mean(profit_factors),
                    'median': np.median(profit_factors),
                    'min': min(profit_factors),
                    'max': max(profit_factors),
                    'std': np.std(profit_factors)
                },
                'sharpe_ratios': {
                    'mean': np.mean(sharpe_ratios),
                    'median': np.median(sharpe_ratios),
                    'min': min(sharpe_ratios),
                    'max': max(sharpe_ratios),
                    'std': np.std(sharpe_ratios)
                },
                'sortino_ratios': {
                    'mean': np.mean(sortino_ratios),
                    'median': np.median(sortino_ratios),
                    'min': min(sortino_ratios),
                    'max': max(sortino_ratios),
                    'std': np.std(sortino_ratios)
                },
                'window_results': window_results
            }
            
            # Save results
            self._save_rolling_window_results(aggregate_results, strategy_name)
            
            logger.info(f"Rolling window backtest completed for strategy {strategy_name}")
            logger.info(f"Pass Rate: {pass_rate * 100:.2f}% ({pass_count}/{len(window_results)})")
            logger.info(f"Mean Return: {aggregate_results['returns']['mean'] * 100:.2f}%")
            logger.info(f"Mean Drawdown: {aggregate_results['drawdowns']['mean'] * 100:.2f}%")
            logger.info(f"Mean Win Rate: {aggregate_results['win_rates']['mean'] * 100:.2f}%")
            
            return aggregate_results
        
        logger.error("No valid window results available")
        return None
    
    def run_multi_strategy_backtest(self, 
                                   strategy_names: List[str], 
                                   start_time: Optional[float] = None, 
                                   end_time: Optional[float] = None,
                                   max_positions_per_strategy: int = 3) -> Dict:
        """
        Run a backtest with multiple strategies.
        
        Args:
            strategy_names: List of strategy names
            start_time: Start timestamp (default: first signal timestamp)
            end_time: End timestamp (default: last signal timestamp)
            max_positions_per_strategy: Maximum number of concurrent positions per strategy
            
        Returns:
            Dictionary with multi-strategy backtest results
        """
        # Check if strategies exist
        for strategy_name in strategy_names:
            if strategy_name not in self.strategies:
                logger.error(f"Strategy {strategy_name} not found")
                return None
        
        # Reset state
        self.rules_engine.reset()
        self.execution_simulator.reset_statistics()
        self.current_positions = []
        self.closed_positions = []
        self.equity_curve = []
        self.drawdown_curve = []
        
        # Determine time range
        if not self.signals:
            logger.error("No signals available for backtesting")
            return None
        
        if start_time is None:
            start_time = self.signals[0].get('timestamp', 0)
        
        if end_time is None:
            end_time = self.signals[-1].get('timestamp', 0)
        
        # Filter signals by time range
        filtered_signals = [
            signal for signal in self.signals
            if start_time <= signal.get('timestamp', 0) <= end_time
        ]
        
        if not filtered_signals:
            logger.error("No signals available in the specified time range")
            return None
        
        logger.info(f"Running multi-strategy backtest with {len(strategy_names)} strategies from {datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)} with {len(filtered_signals)} signals")
        
        # Track positions by strategy
        positions_by_strategy = {strategy_name: 0 for strategy_name in strategy_names}
        
        # Process signals in chronological order
        for signal in filtered_signals:
            # Get signal timestamp
            timestamp = signal.get('timestamp', 0)
            
            # Update existing positions
            self.update_positions(timestamp)
            
            # Update positions count by strategy
            positions_by_strategy = {
                strategy_name: sum(1 for pos in self.current_positions if pos['strategy'] == strategy_name)
                for strategy_name in strategy_names
            }
            
            # Try each strategy
            for strategy_name in strategy_names:
                # Check if we can open a new position with this strategy
                if positions_by_strategy[strategy_name] < max_positions_per_strategy and not self.rules_engine.state['challenge_failed']:
                    # Execute trade
                    position = self.execute_trade(signal, strategy_name, timestamp)
                    
                    # Update positions count if trade executed
                    if position:
                        positions_by_strategy[strategy_name] += 1
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.rules_engine.state['current_balance']
            })
            
            # Update drawdown curve
            self.drawdown_curve.append({
                'timestamp': timestamp,
                'drawdown': self.rules_engine.state['current_drawdown_pct']
            })
            
            # Check if challenge failed or passed
            if self.rules_engine.state['challenge_failed'] or self.rules_engine.state['challenge_passed']:
                break
        
        # Close any remaining positions at the end of the backtest
        for position in list(self.current_positions):
            # Get final market data
            market_data = self.prepare_market_data(
                position['instrument'],
                end_time - 3600,  # 1 hour before
                end_time
            )
            
            if market_data and market_data['current_price']:
                self._close_position(position, market_data['current_price'], end_time, "backtest_end")
            else:
                # Use last known price if market data not available
                self._close_position(position, position['entry_price'], end_time, "backtest_end")
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics()
        
        # Get challenge status
        challenge_status = self.rules_engine.get_challenge_status()
        
        # Calculate metrics by strategy
        metrics_by_strategy = {}
        
        for strategy_name in strategy_names:
            # Filter positions by strategy
            strategy_positions = [pos for pos in self.closed_positions if pos['strategy'] == strategy_name]
            
            if strategy_positions:
                # Calculate strategy-specific metrics
                strategy_metrics = self._calculate_performance_metrics(strategy_positions)
                metrics_by_strategy[strategy_name] = strategy_metrics
        
        # Create backtest results
        backtest_results = {
            'strategies': strategy_names,
            'start_time': start_time,
            'end_time': end_time,
            'signals_count': len(filtered_signals),
            'trades_count': len(self.closed_positions),
            'performance_metrics': self.performance_metrics,
            'metrics_by_strategy': metrics_by_strategy,
            'challenge_status': challenge_status,
            'equity_curve': [point['equity'] for point in self.equity_curve],
            'drawdown_curve': [point['drawdown'] for point in self.drawdown_curve],
            'execution_stats': self.execution_simulator.get_statistics()
        }
        
        # Save results
        self._save_multi_strategy_results(backtest_results, strategy_names)
        
        logger.info(f"Multi-strategy backtest completed with {len(strategy_names)} strategies")
        logger.info(f"Total Return: {self.performance_metrics['total_return'] * 100:.2f}%")
        logger.info(f"Max Drawdown: {self.performance_metrics['max_drawdown'] * 100:.2f}%")
        logger.info(f"Win Rate: {self.performance_metrics['win_rate'] * 100:.2f}%")
        logger.info(f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        logger.info(f"Challenge Result: {'PASSED' if challenge_status['challenge_passed'] else 'FAILED' if challenge_status['challenge_failed'] else 'INCOMPLETE'}")
        
        return backtest_results
    
    def _calculate_performance_metrics(self, positions: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate performance metrics from closed positions.
        
        Args:
            positions: List of closed positions (default: self.closed_positions)
            
        Returns:
            Dictionary with performance metrics
        """
        if positions is None:
            positions = self.closed_positions
        
        if not positions:
            return {
                'total_return': 0.0,
                'absolute_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_profit': 0.0,
                'largest_loss': 0.0,
                'avg_holding_time': 0.0,
                'expectancy': 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(positions)
        winning_trades = sum(1 for pos in positions if pos['profit_loss'] > 0)
        losing_trades = sum(1 for pos in positions if pos['profit_loss'] <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit and loss
        total_profit = sum(pos['profit_loss'] for pos in positions if pos['profit_loss'] > 0)
        total_loss = sum(pos['profit_loss'] for pos in positions if pos['profit_loss'] <= 0)
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate returns
        initial_capital = self.account_size
        final_capital = initial_capital + sum(pos['profit_loss'] for pos in positions)
        
        absolute_return = final_capital - initial_capital
        total_return = absolute_return / initial_capital
        
        # Calculate drawdown
        if self.drawdown_curve:
            max_drawdown = max(point['drawdown'] for point in self.drawdown_curve)
        else:
            max_drawdown = 0.0
        
        # Calculate average profit and loss
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate largest profit and loss
        largest_profit = max((pos['profit_loss'] for pos in positions if pos['profit_loss'] > 0), default=0)
        largest_loss = min((pos['profit_loss'] for pos in positions if pos['profit_loss'] <= 0), default=0)
        
        # Calculate average holding time
        holding_times = [(pos['exit_time'] - pos['entry_time']) / 3600 for pos in positions]  # in hours
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)
        
        # Calculate Sharpe ratio
        if self.equity_curve and len(self.equity_curve) > 1:
            equity_values = [point['equity'] for point in self.equity_curve]
            returns = [(equity_values[i] - equity_values[i-1]) / equity_values[i-1] for i in range(1, len(equity_values))]
            
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # Calculate Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0
            
            sortino_ratio = (avg_return / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'absolute_return': absolute_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'avg_holding_time': avg_holding_time,
            'expectancy': expectancy
        }
        
        return metrics
    
    def _save_backtest_results(self, results: Dict, strategy_name: str) -> None:
        """
        Save backtest results to file.
        
        Args:
            results: Backtest results
            strategy_name: Strategy name
        """
        # Create output directory
        strategy_dir = os.path.join(self.output_dir, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(strategy_dir, 'backtest_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trades
        trades_file = os.path.join(strategy_dir, 'trades.json')
        with open(trades_file, 'w') as f:
            json.dump(self.closed_positions, f, indent=2)
        
        # Save execution statistics
        execution_file = os.path.join(strategy_dir, 'execution_stats.json')
        self.execution_simulator.save_statistics(execution_file)
        
        # Save rules engine report
        rules_file = os.path.join(strategy_dir, 'rules_report.json')
        self.rules_engine.save_report(rules_file)
        
        # Generate plots
        self._generate_performance_plots(results, strategy_dir, strategy_name)
        
        logger.info(f"Saved backtest results to {strategy_dir}")
    
    def _save_rolling_window_results(self, results: Dict, strategy_name: str) -> None:
        """
        Save rolling window backtest results to file.
        
        Args:
            results: Rolling window backtest results
            strategy_name: Strategy name
        """
        # Create output directory
        strategy_dir = os.path.join(self.output_dir, f"{strategy_name}_rolling")
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save aggregate results
        results_file = os.path.join(strategy_dir, 'rolling_window_results.json')
        
        # Remove window_results to avoid huge files
        results_copy = results.copy()
        window_results = results_copy.pop('window_results', [])
        
        with open(results_file, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        # Save individual window results
        for i, window_result in enumerate(window_results):
            window_file = os.path.join(strategy_dir, f'window_{i+1}.json')
            with open(window_file, 'w') as f:
                json.dump(window_result, f, indent=2)
        
        # Generate plots
        self._generate_rolling_window_plots(results, strategy_dir, strategy_name)
        
        logger.info(f"Saved rolling window results to {strategy_dir}")
    
    def _save_multi_strategy_results(self, results: Dict, strategy_names: List[str]) -> None:
        """
        Save multi-strategy backtest results to file.
        
        Args:
            results: Multi-strategy backtest results
            strategy_names: List of strategy names
        """
        # Create output directory
        strategy_str = "_".join(strategy_names)
        if len(strategy_str) > 50:  # Avoid too long filenames
            strategy_str = "multi_strategy"
        
        strategy_dir = os.path.join(self.output_dir, strategy_str)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save results
        results_file = os.path.join(strategy_dir, 'multi_strategy_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trades
        trades_file = os.path.join(strategy_dir, 'trades.json')
        with open(trades_file, 'w') as f:
            json.dump(self.closed_positions, f, indent=2)
        
        # Save execution statistics
        execution_file = os.path.join(strategy_dir, 'execution_stats.json')
        self.execution_simulator.save_statistics(execution_file)
        
        # Save rules engine report
        rules_file = os.path.join(strategy_dir, 'rules_report.json')
        self.rules_engine.save_report(rules_file)
        
        # Generate plots
        self._generate_multi_strategy_plots(results, strategy_dir, strategy_names)
        
        logger.info(f"Saved multi-strategy results to {strategy_dir}")
    
    def _generate_performance_plots(self, results: Dict, output_dir: str, strategy_name: str) -> None:
        """
        Generate performance plots.
        
        Args:
            results: Backtest results
            output_dir: Output directory
            strategy_name: Strategy name
        """
        try:
            # Equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(results['equity_curve'])
            plt.title(f'Equity Curve - {strategy_name}')
            plt.xlabel('Trade Number')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
            plt.close()
            
            # Drawdown curve
            plt.figure(figsize=(12, 6))
            plt.plot([d * 100 for d in results['drawdown_curve']])
            plt.title(f'Drawdown Curve - {strategy_name}')
            plt.xlabel('Trade Number')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'drawdown_curve.png'))
            plt.close()
            
            # Profit distribution
            if self.closed_positions:
                profits = [pos['profit_loss'] for pos in self.closed_positions]
                
                plt.figure(figsize=(12, 6))
                plt.hist(profits, bins=20)
                plt.title(f'Profit Distribution - {strategy_name}')
                plt.xlabel('Profit/Loss')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, 'profit_distribution.png'))
                plt.close()
            
            # Win/loss by instrument
            if self.closed_positions:
                # Group by instrument
                instruments = {}
                for pos in self.closed_positions:
                    instrument = pos['instrument']
                    if instrument not in instruments:
                        instruments[instrument] = {'wins': 0, 'losses': 0}
                    
                    if pos['profit_loss'] > 0:
                        instruments[instrument]['wins'] += 1
                    else:
                        instruments[instrument]['losses'] += 1
                
                # Create plot
                if instruments:
                    instruments_list = list(instruments.keys())
                    wins = [instruments[inst]['wins'] for inst in instruments_list]
                    losses = [instruments[inst]['losses'] for inst in instruments_list]
                    
                    plt.figure(figsize=(12, 6))
                    width = 0.35
                    x = np.arange(len(instruments_list))
                    
                    plt.bar(x - width/2, wins, width, label='Wins')
                    plt.bar(x + width/2, losses, width, label='Losses')
                    
                    plt.title(f'Win/Loss by Instrument - {strategy_name}')
                    plt.xlabel('Instrument')
                    plt.ylabel('Count')
                    plt.xticks(x, instruments_list, rotation=45)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'win_loss_by_instrument.png'))
                    plt.close()
        except Exception as e:
            logger.error(f"Error generating performance plots: {str(e)}")
    
    def _generate_rolling_window_plots(self, results: Dict, output_dir: str, strategy_name: str) -> None:
        """
        Generate rolling window plots.
        
        Args:
            results: Rolling window backtest results
            output_dir: Output directory
            strategy_name: Strategy name
        """
        try:
            # Returns distribution
            plt.figure(figsize=(12, 6))
            returns = [window['performance_metrics']['total_return'] * 100 for window in results['window_results']]
            plt.hist(returns, bins=10)
            plt.title(f'Returns Distribution - {strategy_name} (Rolling Windows)')
            plt.xlabel('Return (%)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'returns_distribution.png'))
            plt.close()
            
            # Drawdown distribution
            plt.figure(figsize=(12, 6))
            drawdowns = [window['performance_metrics']['max_drawdown'] * 100 for window in results['window_results']]
            plt.hist(drawdowns, bins=10)
            plt.title(f'Drawdown Distribution - {strategy_name} (Rolling Windows)')
            plt.xlabel('Max Drawdown (%)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'drawdown_distribution.png'))
            plt.close()
            
            # Win rate distribution
            plt.figure(figsize=(12, 6))
            win_rates = [window['performance_metrics']['win_rate'] * 100 for window in results['window_results']]
            plt.hist(win_rates, bins=10)
            plt.title(f'Win Rate Distribution - {strategy_name} (Rolling Windows)')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'win_rate_distribution.png'))
            plt.close()
            
            # Pass/fail by window
            plt.figure(figsize=(12, 6))
            window_indices = [i+1 for i in range(len(results['window_results']))]
            pass_fail = [1 if window['challenge_status']['challenge_passed'] else 0 for window in results['window_results']]
            
            plt.bar(window_indices, pass_fail, color=['green' if p == 1 else 'red' for p in pass_fail])
            plt.title(f'Challenge Pass/Fail by Window - {strategy_name}')
            plt.xlabel('Window')
            plt.ylabel('Pass (1) / Fail (0)')
            plt.yticks([0, 1])
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'pass_fail_by_window.png'))
            plt.close()
            
            # Failure reasons
            if results['failure_reasons']:
                reasons = list(results['failure_reasons'].keys())
                counts = list(results['failure_reasons'].values())
                
                plt.figure(figsize=(12, 6))
                plt.bar(reasons, counts)
                plt.title(f'Failure Reasons - {strategy_name} (Rolling Windows)')
                plt.xlabel('Reason')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'failure_reasons.png'))
                plt.close()
        except Exception as e:
            logger.error(f"Error generating rolling window plots: {str(e)}")
    
    def _generate_multi_strategy_plots(self, results: Dict, output_dir: str, strategy_names: List[str]) -> None:
        """
        Generate multi-strategy plots.
        
        Args:
            results: Multi-strategy backtest results
            output_dir: Output directory
            strategy_names: List of strategy names
        """
        try:
            # Equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(results['equity_curve'])
            plt.title(f'Equity Curve - Multi-Strategy')
            plt.xlabel('Trade Number')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
            plt.close()
            
            # Drawdown curve
            plt.figure(figsize=(12, 6))
            plt.plot([d * 100 for d in results['drawdown_curve']])
            plt.title(f'Drawdown Curve - Multi-Strategy')
            plt.xlabel('Trade Number')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'drawdown_curve.png'))
            plt.close()
            
            # Performance by strategy
            if results['metrics_by_strategy']:
                # Extract metrics
                returns = [results['metrics_by_strategy'][strategy]['total_return'] * 100 for strategy in strategy_names if strategy in results['metrics_by_strategy']]
                drawdowns = [results['metrics_by_strategy'][strategy]['max_drawdown'] * 100 for strategy in strategy_names if strategy in results['metrics_by_strategy']]
                win_rates = [results['metrics_by_strategy'][strategy]['win_rate'] * 100 for strategy in strategy_names if strategy in results['metrics_by_strategy']]
                
                strategies = [strategy for strategy in strategy_names if strategy in results['metrics_by_strategy']]
                
                if strategies:
                    # Returns by strategy
                    plt.figure(figsize=(12, 6))
                    plt.bar(strategies, returns)
                    plt.title('Returns by Strategy')
                    plt.xlabel('Strategy')
                    plt.ylabel('Return (%)')
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'returns_by_strategy.png'))
                    plt.close()
                    
                    # Drawdowns by strategy
                    plt.figure(figsize=(12, 6))
                    plt.bar(strategies, drawdowns)
                    plt.title('Max Drawdown by Strategy')
                    plt.xlabel('Strategy')
                    plt.ylabel('Max Drawdown (%)')
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'drawdowns_by_strategy.png'))
                    plt.close()
                    
                    # Win rates by strategy
                    plt.figure(figsize=(12, 6))
                    plt.bar(strategies, win_rates)
                    plt.title('Win Rate by Strategy')
                    plt.xlabel('Strategy')
                    plt.ylabel('Win Rate (%)')
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'win_rates_by_strategy.png'))
                    plt.close()
            
            # Trades by strategy
            if self.closed_positions:
                # Count trades by strategy
                trades_by_strategy = {}
                for pos in self.closed_positions:
                    strategy = pos['strategy']
                    if strategy not in trades_by_strategy:
                        trades_by_strategy[strategy] = 0
                    trades_by_strategy[strategy] += 1
                
                if trades_by_strategy:
                    strategies = list(trades_by_strategy.keys())
                    counts = list(trades_by_strategy.values())
                    
                    plt.figure(figsize=(12, 6))
                    plt.bar(strategies, counts)
                    plt.title('Trades by Strategy')
                    plt.xlabel('Strategy')
                    plt.ylabel('Number of Trades')
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'trades_by_strategy.png'))
                    plt.close()
        except Exception as e:
            logger.error(f"Error generating multi-strategy plots: {str(e)}")

def main():
    """Run a sample backtest."""
    import argparse
    from strategies.regime_switching_strategy import RegimeSwitchingStrategy
    
    parser = argparse.ArgumentParser(description='Run a backtest')
    parser.add_argument('--signals-file', type=str, required=True,
                        help='Path to signals JSON file')
    parser.add_argument('--market-data-dir', type=str, required=True,
                        help='Directory containing market data files')
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                        help='Directory to save backtest results')
    parser.add_argument('--prop-firm', type=str, default='TFT',
                        choices=['TFT', 'FTMO', 'MFF'],
                        help='Prop firm type')
    parser.add_argument('--account-size', type=float, default=100000.0,
                        help='Account size')
    parser.add_argument('--challenge-phase', type=str, default='phase1',
                        choices=['phase1', 'phase2', 'verification', 'funded'],
                        help='Challenge phase')
    parser.add_argument('--rolling-window', action='store_true',
                        help='Run rolling window backtest')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Window size in days for rolling window backtest')
    parser.add_argument('--step-size', type=int, default=5,
                        help='Step size in days for rolling window backtest')
    parser.add_argument('--max-windows', type=int, default=10,
                        help='Maximum number of windows for rolling window backtest')
    
    args = parser.parse_args()
    
    # Initialize backtest engine
    engine = BacktestEngine(
        signals_file=args.signals_file,
        market_data_dir=args.market_data_dir,
        output_dir=args.output_dir,
        prop_firm=args.prop_firm,
        account_size=args.account_size,
        challenge_phase=args.challenge_phase
    )
    
    # Register strategy
    engine.register_strategy(
        name='RegimeSwitchingStrategy',
        strategy_class=RegimeSwitchingStrategy,
        params={
            'initial_capital': args.account_size,
            'base_risk_per_trade': 0.01,
            'max_drawdown': 0.04,
            'daily_loss_limit': 0.015,
            'profit_target': 0.08
        }
    )
    
    # Run backtest
    if args.rolling_window:
        results = engine.run_rolling_window_backtest(
            strategy_name='RegimeSwitchingStrategy',
            window_size=args.window_size,
            step_size=args.step_size,
            max_windows=args.max_windows
        )
        
        if results:
            print(f"Rolling window backtest completed")
            print(f"Pass Rate: {results['pass_rate'] * 100:.2f}% ({results['pass_count']}/{results['num_windows']})")
            print(f"Mean Return: {results['returns']['mean'] * 100:.2f}%")
            print(f"Mean Drawdown: {results['drawdowns']['mean'] * 100:.2f}%")
            print(f"Mean Win Rate: {results['win_rates']['mean'] * 100:.2f}%")
    else:
        results = engine.run_backtest(strategy_name='RegimeSwitchingStrategy')
        
        if results:
            print(f"Backtest completed")
            print(f"Total Return: {results['performance_metrics']['total_return'] * 100:.2f}%")
            print(f"Max Drawdown: {results['performance_metrics']['max_drawdown'] * 100:.2f}%")
            print(f"Win Rate: {results['performance_metrics']['win_rate'] * 100:.2f}%")
            print(f"Profit Factor: {results['performance_metrics']['profit_factor']:.2f}")
            print(f"Challenge Result: {'PASSED' if results['challenge_status']['challenge_passed'] else 'FAILED' if results['challenge_status']['challenge_failed'] else 'INCOMPLETE'}")

if __name__ == "__main__":
    main()