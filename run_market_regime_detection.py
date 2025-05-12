#!/usr/bin/env python3
"""
Market Regime Detection Script for Advanced Backtesting Engine

This script provides a command-line interface for detecting market regimes
and running adaptive strategy selection using the advanced backtesting engine.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backtesting modules
from backtesting.backtest_engine import BacktestEngine
from backtesting.rules_engine import PropFirmRulesEngine
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.market_regime import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_regime.log')
    ]
)

logger = logging.getLogger('market_regime_detection')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run market regime detection and adaptive strategy selection')
    
    # Required arguments
    parser.add_argument('--market-data-dir', type=str, required=True,
                       help='Directory containing market data files')
    
    # Optional arguments
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/filtered/filtered_signals.json',
                       help='Path to the signals file')
    parser.add_argument('--output-dir', type=str, default='data/regime_detection_results',
                       help='Directory to save detection results')
    parser.add_argument('--prop-firm', type=str, choices=['TFT', 'FTMO', 'MFF'], default='TFT',
                       help='Prop firm rules to apply')
    parser.add_argument('--account-size', type=float, default=100000.0,
                       help='Account size for backtesting')
    parser.add_argument('--challenge-phase', type=str, 
                       choices=['phase1', 'phase2', 'verification', 'funded'], default='phase1',
                       help='Challenge phase for prop firm rules')
    parser.add_argument('--risk-per-trade', type=float, default=1.0,
                       help='Risk percentage per trade')
    parser.add_argument('--max-positions', type=int, default=3,
                       help='Maximum number of simultaneous positions')
    parser.add_argument('--n-regimes', type=int, default=3,
                       help='Number of regimes to detect')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'gmm', 'hmm', 'pca_kmeans'],
                       help='Regime detection method')
    parser.add_argument('--lookback-window', type=int, default=20,
                       help='Lookback window for feature calculation')
    parser.add_argument('--primary-instrument', type=str, default=None,
                       help='Primary instrument for regime detection')
    parser.add_argument('--features', type=str, nargs='+',
                       default=['returns', 'volatility', 'rsi', 'ema_ratio', 'bb_width', 'volume_ratio', 'macd'],
                       help='Features to use for regime detection')
    parser.add_argument('--run-adaptive-backtest', action='store_true',
                       help='Run backtest with adaptive strategy selection')
    parser.add_argument('--start-time', type=str, default=None,
                       help='Start time for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-time', type=str, default=None,
                       help='End time for backtest (YYYY-MM-DD)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def setup_backtest_engine(args):
    """Set up the backtest engine with the specified configuration."""
    # Initialize rules engine
    rules_engine = PropFirmRulesEngine(
        prop_firm=args.prop_firm,
        account_size=args.account_size,
        challenge_phase=args.challenge_phase
    )
    
    # Initialize execution simulator
    execution_simulator = ExecutionSimulator(
        slippage_model='volatility',
        spread_model='dynamic'
    )
    
    # Initialize backtest engine
    engine = BacktestEngine(
        rules_engine=rules_engine,
        execution_simulator=execution_simulator,
        signals_file=args.signals_file,
        market_data_dir=args.market_data_dir,
        risk_per_trade=args.risk_per_trade,
        max_positions=args.max_positions
    )
    
    # Register strategies
    engine.register_strategy('AsymmetricRiskProfile')
    engine.register_strategy('RegimeSwitching')
    engine.register_strategy('MeanReversion')
    engine.register_strategy('TrendFollowing')
    
    return engine

def load_market_data(market_data_dir):
    """Load market data from directory."""
    market_data = {}
    
    if not os.path.exists(market_data_dir):
        logger.error(f"Market data directory not found: {market_data_dir}")
        return market_data
    
    # Look for CSV or JSON files
    for filename in os.listdir(market_data_dir):
        if filename.endswith('.csv'):
            try:
                filepath = os.path.join(market_data_dir, filename)
                instrument = os.path.splitext(filename)[0]
                data = pd.read_csv(filepath)
                
                # Convert timestamp to datetime if it exists
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True)
                
                market_data[instrument] = data
                logger.info(f"Loaded {len(data)} data points for {instrument}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
        
        elif filename.endswith('.json'):
            try:
                filepath = os.path.join(market_data_dir, filename)
                instrument = os.path.splitext(filename)[0]
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame if it's a list
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                    
                    # Convert timestamp to datetime if it exists
                    if 'timestamp' in data.columns:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data.set_index('timestamp', inplace=True)
                
                market_data[instrument] = data
                logger.info(f"Loaded {len(data)} data points for {instrument}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
    
    return market_data

def run_regime_detection(args, market_data):
    """Run market regime detection with the specified configuration."""
    # Initialize regime detector
    detector = MarketRegimeDetector(
        market_data=market_data,
        features=args.features,
        n_regimes=args.n_regimes,
        lookback_window=args.lookback_window
    )
    
    # Prepare data
    feature_data = detector.prepare_data(primary_instrument=args.primary_instrument)
    
    if feature_data is None or len(feature_data) == 0:
        logger.error("Failed to prepare feature data")
        return None
    
    # Train regime detection model
    model = detector.train(
        method=args.method,
        random_state=args.random_state
    )
    
    if model is None:
        logger.error("Failed to train regime detection model")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate and save plots
    regimes_plot = detector.plot_regimes()
    if regimes_plot:
        regimes_plot.savefig(os.path.join(args.output_dir, 'market_regimes.png'))
    
    characteristics_plot = detector.plot_regime_characteristics()
    if characteristics_plot:
        characteristics_plot.savefig(os.path.join(args.output_dir, 'regime_characteristics.png'))
    
    # Save model
    detector.save_model(os.path.join(args.output_dir, f'regime_model_{args.method}.json'))
    
    return detector

def run_adaptive_backtest(args, detector, engine):
    """Run backtest with adaptive strategy selection based on detected regimes."""
    if detector is None or engine is None:
        logger.error("Detector or engine not available")
        return None
    
    # Define strategy mapping based on regimes
    strategy_mapping = {}
    
    # Map regimes to strategies based on their characteristics
    regime_stats = detector.regime_stats if hasattr(detector, 'regime_stats') else {}
    
    for regime, stats in regime_stats.items():
        # Simple mapping based on volatility and returns
        volatility = stats.get('volatility_mean', 0)
        returns = stats.get('returns_mean', 0)
        
        if volatility > 0.015:  # High volatility
            if returns > 0:  # Positive returns
                strategy_mapping[regime] = 'TrendFollowing'
            else:  # Negative returns
                strategy_mapping[regime] = 'MeanReversion'
        else:  # Low volatility
            if returns > 0:  # Positive returns
                strategy_mapping[regime] = 'AsymmetricRiskProfile'
            else:  # Negative returns
                strategy_mapping[regime] = 'RegimeSwitching'
    
    # Fill in any missing regimes
    for regime in range(detector.n_regimes):
        if regime not in strategy_mapping:
            strategy_mapping[regime] = 'AsymmetricRiskProfile'
    
    # Define parameters for each strategy
    params_mapping = {
        0: {'risk_per_trade': 1.0, 'max_positions': 3},
        1: {'risk_per_trade': 0.8, 'max_positions': 2},
        2: {'risk_per_trade': 1.2, 'max_positions': 4}
    }
    
    # Convert date strings to timestamps if provided
    start_time = None
    if args.start_time:
        start_time = int(datetime.fromisoformat(args.start_time).timestamp())
    
    end_time = None
    if args.end_time:
        end_time = int(datetime.fromisoformat(args.end_time).timestamp())
    
    # Run adaptive backtest
    results = detector.backtest_with_regime_adaptation(
        backtest_engine=engine,
        strategy_mapping=strategy_mapping,
        params_mapping=params_mapping,
        start_time=start_time,
        end_time=end_time
    )
    
    if results is None:
        logger.error("Adaptive backtest failed")
        return None
    
    # Save results
    with open(os.path.join(args.output_dir, 'adaptive_backtest_results.json'), 'w') as f:
        # Remove complex objects that can't be easily serialized
        results_copy = results.copy()
        for key in ['equity_curve', 'drawdown_curve']:
            if key in results_copy:
                del results_copy[key]
        
        json.dump(results_copy, f, indent=2)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    equity_curve = results.get('equity_curve', [])
    if equity_curve:
        ax1.plot(equity_curve, linewidth=2)
        ax1.set_title('Adaptive Strategy Equity Curve')
        ax1.set_ylabel('Equity')
        ax1.grid(True, alpha=0.3)
    
    # Plot drawdown
    drawdown_curve = results.get('drawdown_curve', [])
    if drawdown_curve:
        ax2.fill_between(range(len(drawdown_curve)), 0, drawdown_curve, color='red', alpha=0.3)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'adaptive_backtest_performance.png'))
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Starting market regime detection with {args.n_regimes} regimes")
    logger.info(f"Method: {args.method}")
    
    # Load market data
    market_data = load_market_data(args.market_data_dir)
    
    if not market_data:
        logger.error("No market data loaded")
        sys.exit(1)
    
    # Run regime detection
    detector = run_regime_detection(args, market_data)
    
    if detector is None:
        logger.error("Regime detection failed")
        sys.exit(1)
    
    # Run adaptive backtest if requested
    if args.run_adaptive_backtest:
        logger.info("Running adaptive backtest")
        
        # Set up backtest engine
        engine = setup_backtest_engine(args)
        
        # Run adaptive backtest
        results = run_adaptive_backtest(args, detector, engine)
        
        if results:
            logger.info("Adaptive backtest completed")
            logger.info(f"Total return: {results.get('total_return', 0):.2f}%")
            logger.info(f"Max drawdown: {results.get('max_drawdown', 0):.2f}%")
            logger.info(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Win rate: {results.get('win_rate', 0):.2f}%")
            logger.info(f"Number of trades: {results.get('num_trades', 0)}")
            logger.info(f"Number of regime changes: {len(results.get('regime_changes', []))}")
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()