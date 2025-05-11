#!/usr/bin/env python3
"""
Run backtests on the filtered and enriched signals using the Asymmetric Risk Profile Strategy
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backtest')

class AsymmetricRiskProfileStrategy:
    """
    Asymmetric Risk Profile Strategy for forex trading
    
    Features:
    - Asymmetric risk-reward ratio (1:2 or better)
    - Dynamic position sizing based on volatility
    - Early exit on adverse price movement
    - Partial profit taking at key levels
    - Trailing stop loss after reaching partial profit
    """
    
    def __init__(self, initial_capital=10000, risk_per_trade=0.01, max_drawdown=0.04, 
                 daily_loss_limit=0.015, profit_target=0.08, partial_tp_ratio=0.5,
                 trailing_stop_activation=0.5, trailing_stop_distance=0.5):
        """
        Initialize the strategy
        
        Args:
            initial_capital: Starting capital in USD
            risk_per_trade: Maximum risk per trade as a fraction of capital (0.01 = 1%)
            max_drawdown: Maximum allowed drawdown as a fraction of capital (0.04 = 4%)
            daily_loss_limit: Maximum daily loss as a fraction of capital (0.015 = 1.5%)
            profit_target: Profit target as a fraction of capital (0.08 = 8%)
            partial_tp_ratio: Ratio of position to close at first take profit (0.5 = 50%)
            trailing_stop_activation: When to activate trailing stop (0.5 = 50% of the way to TP)
            trailing_stop_distance: Distance of trailing stop as a fraction of ATR
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.profit_target = profit_target
        self.partial_tp_ratio = partial_tp_ratio
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        self.max_capital = initial_capital
        self.max_drawdown_experienced = 0
        self.current_drawdown = 0
        
        # Current state
        self.open_positions = []
        self.closed_positions = []
    
    def calculate_position_size(self, entry_price, stop_loss, pair):
        """
        Calculate position size based on risk per trade and distance to stop loss
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            pair: Currency pair
            
        Returns:
            Position size in units
        """
        if entry_price is None or stop_loss is None:
            return 0
        
        # Calculate risk amount in USD
        risk_amount = self.capital * self.risk_per_trade
        
        # Calculate pip value (assuming standard lot size of 100,000 units)
        if 'JPY' in pair:
            pip_value = 0.01  # For JPY pairs, a pip is 0.01
        else:
            pip_value = 0.0001  # For other pairs, a pip is 0.0001
        
        # Calculate distance to stop loss in pips
        if 'JPY' in pair:
            distance_to_sl = abs(entry_price - stop_loss) / 0.01
        else:
            distance_to_sl = abs(entry_price - stop_loss) / 0.0001
        
        # Calculate position size in standard lots
        if distance_to_sl > 0:
            position_size = risk_amount / (distance_to_sl * pip_value * 100000)
        else:
            position_size = 0
        
        # Convert to units
        position_size_units = position_size * 100000
        
        return position_size_units
    
    def execute_trade(self, signal, market_data):
        """
        Execute a trade based on the signal and market data
        
        Args:
            signal: Trading signal
            market_data: Market data for the pair
            
        Returns:
            Trade result
        """
        # Extract signal details
        pair = signal.get('asset', signal.get('pair', ''))
        direction = signal.get('direction', '')
        entry_price = signal.get('entry', signal.get('entry_price'))
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        timestamp = float(signal.get('timestamp', 0))
        quality_score = signal.get('quality_score', 0.5)
        
        # Skip if missing required data
        if not pair or not direction or not entry_price or not stop_loss or not take_profit:
            logger.warning(f"Skipping signal {signal.get('id')} due to missing data")
            return None
        
        # Convert to float if needed
        try:
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            take_profit = float(take_profit)
        except (ValueError, TypeError):
            logger.warning(f"Skipping signal {signal.get('id')} due to invalid price data")
            return None
        
        # Find relevant market data
        if not market_data:
            logger.warning(f"No market data found for {pair}")
            return None
        
        # Filter market data to the period after the signal
        market_data_df = pd.DataFrame(market_data)
        market_data_df['timestamp'] = market_data_df['timestamp'].astype(float)
        future_data = market_data_df[market_data_df['timestamp'] >= timestamp]
        
        if future_data.empty:
            logger.warning(f"No future market data found for signal {signal.get('id')}")
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss, pair)
        
        # Adjust position size based on quality score
        position_size *= quality_score
        
        # Check if we have enough capital
        if position_size <= 0:
            logger.warning(f"Invalid position size for signal {signal.get('id')}")
            return None
        
        # Initialize trade
        trade = {
            'id': signal.get('id'),
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': timestamp,
            'quality_score': quality_score,
            'status': 'open',
            'exit_price': None,
            'exit_time': None,
            'pnl': 0,
            'pnl_pct': 0,
            'duration': 0,
            'exit_reason': None
        }
        
        # Simulate trade execution
        current_position_size = position_size
        partial_tp_taken = False
        trailing_stop_activated = False
        current_stop_loss = stop_loss
        
        for _, bar in future_data.iterrows():
            bar_time = bar['timestamp']
            bar_open = bar['open']
            bar_high = bar['high']
            bar_low = bar['low']
            bar_close = bar['close']
            
            # Check if we've hit stop loss
            if direction.upper() == 'LONG' and bar_low <= current_stop_loss:
                trade['exit_price'] = current_stop_loss
                trade['exit_time'] = bar_time
                trade['status'] = 'closed'
                trade['exit_reason'] = 'stop_loss'
                break
            elif direction.upper() == 'SHORT' and bar_high >= current_stop_loss:
                trade['exit_price'] = current_stop_loss
                trade['exit_time'] = bar_time
                trade['status'] = 'closed'
                trade['exit_reason'] = 'stop_loss'
                break
            
            # Check if we've hit take profit
            if direction.upper() == 'LONG' and bar_high >= take_profit:
                if not partial_tp_taken and self.partial_tp_ratio > 0:
                    # Take partial profit
                    partial_position = current_position_size * self.partial_tp_ratio
                    current_position_size -= partial_position
                    partial_tp_taken = True
                    
                    # If we've closed the entire position
                    if current_position_size <= 0:
                        trade['exit_price'] = take_profit
                        trade['exit_time'] = bar_time
                        trade['status'] = 'closed'
                        trade['exit_reason'] = 'take_profit'
                        break
                else:
                    # Take full profit
                    trade['exit_price'] = take_profit
                    trade['exit_time'] = bar_time
                    trade['status'] = 'closed'
                    trade['exit_reason'] = 'take_profit'
                    break
            elif direction.upper() == 'SHORT' and bar_low <= take_profit:
                if not partial_tp_taken and self.partial_tp_ratio > 0:
                    # Take partial profit
                    partial_position = current_position_size * self.partial_tp_ratio
                    current_position_size -= partial_position
                    partial_tp_taken = True
                    
                    # If we've closed the entire position
                    if current_position_size <= 0:
                        trade['exit_price'] = take_profit
                        trade['exit_time'] = bar_time
                        trade['status'] = 'closed'
                        trade['exit_reason'] = 'take_profit'
                        break
                else:
                    # Take full profit
                    trade['exit_price'] = take_profit
                    trade['exit_time'] = bar_time
                    trade['status'] = 'closed'
                    trade['exit_reason'] = 'take_profit'
                    break
            
            # Check if we should activate trailing stop
            if not trailing_stop_activated:
                if direction.upper() == 'LONG':
                    price_progress = (bar_close - entry_price) / (take_profit - entry_price)
                    if price_progress >= self.trailing_stop_activation:
                        trailing_stop_activated = True
                        # Set trailing stop at a percentage of the move
                        current_stop_loss = max(current_stop_loss, entry_price + (bar_close - entry_price) * (1 - self.trailing_stop_distance))
                elif direction.upper() == 'SHORT':
                    price_progress = (entry_price - bar_close) / (entry_price - take_profit)
                    if price_progress >= self.trailing_stop_activation:
                        trailing_stop_activated = True
                        # Set trailing stop at a percentage of the move
                        current_stop_loss = min(current_stop_loss, entry_price - (entry_price - bar_close) * (1 - self.trailing_stop_distance))
            else:
                # Update trailing stop
                if direction.upper() == 'LONG':
                    new_stop = bar_close - (take_profit - entry_price) * self.trailing_stop_distance
                    current_stop_loss = max(current_stop_loss, new_stop)
                elif direction.upper() == 'SHORT':
                    new_stop = bar_close + (entry_price - take_profit) * self.trailing_stop_distance
                    current_stop_loss = min(current_stop_loss, new_stop)
        
        # If trade is still open, close at the last price
        if trade['status'] == 'open':
            last_price = future_data.iloc[-1]['close']
            trade['exit_price'] = last_price
            trade['exit_time'] = future_data.iloc[-1]['timestamp']
            trade['status'] = 'closed'
            trade['exit_reason'] = 'end_of_data'
        
        # Calculate P&L
        if direction.upper() == 'LONG':
            trade['pnl'] = (trade['exit_price'] - entry_price) * position_size
        else:
            trade['pnl'] = (entry_price - trade['exit_price']) * position_size
        
        trade['pnl_pct'] = trade['pnl'] / self.capital
        trade['duration'] = trade['exit_time'] - timestamp
        
        # Update capital
        self.capital += trade['pnl']
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': trade['exit_time'],
            'capital': self.capital
        })
        
        # Update max capital and drawdown
        if self.capital > self.max_capital:
            self.max_capital = self.capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.max_capital - self.capital) / self.max_capital
            if self.current_drawdown > self.max_drawdown_experienced:
                self.max_drawdown_experienced = self.current_drawdown
        
        # Update daily P&L
        exit_date = datetime.fromtimestamp(trade['exit_time']).date().isoformat()
        if exit_date not in self.daily_pnl:
            self.daily_pnl[exit_date] = 0
        self.daily_pnl[exit_date] += trade['pnl']
        
        # Add to trades list
        self.trades.append(trade)
        
        return trade
    
    def run_backtest(self, signals, market_data_dir):
        """
        Run backtest on a list of signals
        
        Args:
            signals: List of trading signals
            market_data_dir: Directory containing market data files
            
        Returns:
            Backtest results
        """
        logger.info(f"Running backtest on {len(signals)} signals")
        
        # Reset state
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [{
            'timestamp': 0,
            'capital': self.initial_capital
        }]
        self.daily_pnl = {}
        self.max_capital = self.initial_capital
        self.max_drawdown_experienced = 0
        self.current_drawdown = 0
        
        # Load market data for each pair
        market_data = {}
        for signal in signals:
            pair = signal.get('asset', signal.get('pair', ''))
            if pair and pair not in market_data:
                market_data_file = os.path.join(market_data_dir, f"{pair}.json")
                if os.path.exists(market_data_file):
                    with open(market_data_file, 'r') as f:
                        market_data[pair] = json.load(f)
                    logger.info(f"Loaded market data for {pair}: {len(market_data[pair])} bars")
                else:
                    logger.warning(f"Market data file not found for {pair}: {market_data_file}")
        
        # Sort signals by timestamp
        signals = sorted(signals, key=lambda x: float(x.get('timestamp', 0)))
        
        # Execute trades
        for signal in tqdm(signals, desc="Executing trades"):
            pair = signal.get('asset', signal.get('pair', ''))
            if pair in market_data:
                self.execute_trade(signal, market_data[pair])
            else:
                logger.warning(f"No market data for {pair}, skipping signal {signal.get('id')}")
        
        # Calculate performance metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0
        
        total_profit = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        total_loss = sum([t['pnl'] for t in self.trades if t['pnl'] < 0])
        
        if losing_trades > 0 and winning_trades > 0:
            profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
            avg_win = total_profit / winning_trades
            avg_loss = total_loss / losing_trades
            avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            profit_factor = 0 if winning_trades == 0 else float('inf')
            avg_win = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            avg_win_loss_ratio = 0 if avg_win == 0 or losing_trades == 0 else float('inf')
        
        net_profit = self.capital - self.initial_capital
        net_profit_pct = net_profit / self.initial_capital
        
        # Calculate daily metrics
        daily_returns = pd.Series(self.daily_pnl).sort_index()
        if not daily_returns.empty:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
            sortino_ratio = daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252) if daily_returns[daily_returns < 0].std() != 0 else 0
            max_daily_profit = daily_returns.max()
            max_daily_loss = daily_returns.min()
            profitable_days = (daily_returns > 0).sum()
            losing_days = (daily_returns < 0).sum()
            daily_win_rate = profitable_days / len(daily_returns) if len(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_daily_profit = 0
            max_daily_loss = 0
            profitable_days = 0
            losing_days = 0
            daily_win_rate = 0
        
        # Prepare results
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'max_drawdown': self.max_drawdown_experienced,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_daily_profit': max_daily_profit,
            'max_daily_loss': max_daily_loss,
            'profitable_days': profitable_days,
            'losing_days': losing_days,
            'daily_win_rate': daily_win_rate,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'daily_pnl': self.daily_pnl
        }
        
        return results
    
    def plot_equity_curve(self, results, output_file=None):
        """
        Plot equity curve
        
        Args:
            results: Backtest results
            output_file: Output file path
        """
        equity_curve = pd.DataFrame(results['equity_curve'])
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'], unit='s')
        equity_curve.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['capital'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def plot_drawdown(self, results, output_file=None):
        """
        Plot drawdown
        
        Args:
            results: Backtest results
            output_file: Output file path
        """
        equity_curve = pd.DataFrame(results['equity_curve'])
        equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'], unit='s')
        equity_curve.set_index('timestamp', inplace=True)
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['capital'].cummax()
        equity_curve['drawdown'] = (equity_curve['peak'] - equity_curve['capital']) / equity_curve['peak']
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve.index, equity_curve['drawdown'] * 100)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def plot_monthly_returns(self, results, output_file=None):
        """
        Plot monthly returns
        
        Args:
            results: Backtest results
            output_file: Output file path
        """
        daily_pnl = pd.Series(results['daily_pnl'])
        daily_pnl.index = pd.to_datetime(daily_pnl.index)
        
        # Resample to monthly
        monthly_returns = daily_pnl.resample('M').sum()
        
        plt.figure(figsize=(12, 6))
        monthly_returns.plot(kind='bar')
        plt.title('Monthly Returns')
        plt.xlabel('Month')
        plt.ylabel('Profit/Loss')
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def generate_report(self, results, output_dir):
        """
        Generate backtest report
        
        Args:
            results: Backtest results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
            # Convert to serializable format
            serializable_results = results.copy()
            serializable_results['trades'] = [
                {k: str(v) if isinstance(v, (np.float32, np.float64, np.int64, np.int32)) else v 
                 for k, v in trade.items()}
                for trade in results['trades']
            ]
            serializable_results['equity_curve'] = [
                {k: str(v) if isinstance(v, (np.float32, np.float64, np.int64, np.int32)) else v 
                 for k, v in point.items()}
                for point in results['equity_curve']
            ]
            serializable_results['daily_pnl'] = {
                k: str(v) if isinstance(v, (np.float32, np.float64, np.int64, np.int32)) else v
                for k, v in results['daily_pnl'].items()
            }
            
            # Convert numpy types to Python native types
            for key in serializable_results:
                if isinstance(serializable_results[key], (np.float32, np.float64, np.int64, np.int32)):
                    serializable_results[key] = serializable_results[key].item()
            
            # Remove non-serializable metrics
            for key in ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'avg_win_loss_ratio']:
                if key in serializable_results and not np.isfinite(serializable_results[key]):
                    serializable_results[key] = str(serializable_results[key])
            
            json.dump(serializable_results, f, indent=2)
        
        # Generate plots
        self.plot_equity_curve(results, os.path.join(output_dir, 'equity_curve.png'))
        self.plot_drawdown(results, os.path.join(output_dir, 'drawdown.png'))
        self.plot_monthly_returns(results, os.path.join(output_dir, 'monthly_returns.png'))
        
        # Generate summary report
        with open(os.path.join(output_dir, 'backtest_summary.txt'), 'w') as f:
            f.write("Asymmetric Risk Profile Strategy Backtest Results\n")
            f.write("==============================================\n\n")
            
            f.write(f"Initial Capital: ${results['initial_capital']:.2f}\n")
            f.write(f"Final Capital: ${results['final_capital']:.2f}\n")
            f.write(f"Net Profit: ${results['net_profit']:.2f} ({results['net_profit_pct'] * 100:.2f}%)\n")
            f.write(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%\n\n")
            
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Winning Trades: {results['winning_trades']} ({results['win_rate'] * 100:.2f}%)\n")
            f.write(f"Losing Trades: {results['losing_trades']} ({(1 - results['win_rate']) * 100:.2f}%)\n")
            f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
            f.write(f"Average Win: ${results['avg_win']:.2f}\n")
            f.write(f"Average Loss: ${results['avg_loss']:.2f}\n")
            f.write(f"Average Win/Loss Ratio: {results['avg_win_loss_ratio']:.2f}\n\n")
            
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Sortino Ratio: {results['sortino_ratio']:.2f}\n")
            f.write(f"Max Daily Profit: ${results['max_daily_profit']:.2f}\n")
            f.write(f"Max Daily Loss: ${results['max_daily_loss']:.2f}\n")
            f.write(f"Profitable Days: {results['profitable_days']} ({results['daily_win_rate'] * 100:.2f}%)\n")
            f.write(f"Losing Days: {results['losing_days']} ({(1 - results['daily_win_rate']) * 100:.2f}%)\n")
        
        logger.info(f"Backtest report generated in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run backtest on filtered and enriched signals')
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/enriched/enriched_signals.json',
                        help='Path to the filtered and enriched signals JSON file')
    parser.add_argument('--market-data-dir', type=str, default='data/market_data',
                        help='Directory containing market data JSON files')
    parser.add_argument('--output-dir', type=str, default='data/backtest_results',
                        help='Directory to save backtest results')
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtest')
    parser.add_argument('--risk-per-trade', type=float, default=0.01,
                        help='Risk per trade as a fraction of capital')
    parser.add_argument('--max-drawdown', type=float, default=0.04,
                        help='Maximum drawdown allowed (TFT: 4%)')
    parser.add_argument('--daily-loss-limit', type=float, default=0.015,
                        help='Maximum daily loss allowed (TFT: 1.5%)')
    parser.add_argument('--profit-target', type=float, default=0.08,
                        help='Profit target (TFT: 8%)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load signals
    logger.info(f"Loading signals from {args.signals_file}")
    with open(args.signals_file, 'r') as f:
        data = json.load(f)
    
    # Extract signals from the data
    if isinstance(data, dict) and 'signals' in data:
        signals = data['signals']
    else:
        signals = data
    
    logger.info(f"Loaded {len(signals)} signals")
    
    # Initialize strategy
    logger.info("Initializing Asymmetric Risk Profile Strategy")
    strategy = AsymmetricRiskProfileStrategy(
        initial_capital=args.initial_capital,
        risk_per_trade=args.risk_per_trade,
        max_drawdown=args.max_drawdown,
        daily_loss_limit=args.daily_loss_limit,
        profit_target=args.profit_target,
        partial_tp_ratio=0.5,
        trailing_stop_activation=0.5,
        trailing_stop_distance=0.5
    )
    
    # Run backtest
    logger.info("Running backtest")
    results = strategy.run_backtest(signals, args.market_data_dir)
    logger.info("Backtest complete")
    
    # Generate report
    logger.info("Generating report")
    strategy.generate_report(results, args.output_dir)
    
    # Print summary
    print("\nBacktest Summary:")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    print(f"Net Profit: ${results['net_profit']:.2f} ({results['net_profit_pct'] * 100:.2f}%)")
    print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate'] * 100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")

if __name__ == "__main__":
    main()