#!/usr/bin/env python3
"""
Multi-Account Simulation Script for Advanced Backtesting Engine

This script provides a command-line interface for running multi-account simulations
using the advanced backtesting engine.
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
from backtesting.multi_account import MultiAccountSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_account.log')
    ]
)

logger = logging.getLogger('multi_account_simulation')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-account simulations for backtesting')
    
    # Required arguments
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to the configuration file')
    
    # Optional arguments
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/filtered/filtered_signals.json',
                       help='Path to the signals file')
    parser.add_argument('--market-data-dir', type=str, default='data/market_data',
                       help='Directory containing market data files')
    parser.add_argument('--output-dir', type=str, default='data/multi_account_results',
                       help='Directory to save simulation results')
    parser.add_argument('--synchronize-data', action='store_true', default=True,
                       help='Synchronize market data across accounts')
    parser.add_argument('--start-time', type=str, default=None,
                       help='Start time for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-time', type=str, default=None,
                       help='End time for backtest (YYYY-MM-DD)')
    
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from file."""
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def setup_backtest_engines(config, args):
    """Set up backtest engines for each account."""
    engines = []
    
    for account_config in config['accounts']:
        # Initialize rules engine
        rules_engine = PropFirmRulesEngine(
            prop_firm=account_config.get('prop_firm', 'TFT'),
            account_size=account_config.get('account_size', 100000.0),
            challenge_phase=account_config.get('challenge_phase', 'phase1')
        )
        
        # Initialize execution simulator
        execution_simulator = ExecutionSimulator(
            slippage_model=account_config.get('slippage_model', 'volatility'),
            spread_model=account_config.get('spread_model', 'dynamic')
        )
        
        # Initialize backtest engine
        engine = BacktestEngine(
            rules_engine=rules_engine,
            execution_simulator=execution_simulator,
            signals_file=args.signals_file,
            market_data_dir=args.market_data_dir,
            risk_per_trade=account_config.get('risk_per_trade', 1.0),
            max_positions=account_config.get('max_positions', 3)
        )
        
        # Register strategies
        for strategy in account_config.get('strategies', []):
            engine.register_strategy(strategy)
        
        engines.append(engine)
    
    return engines

def run_multi_account_simulation(args, engines, config):
    """Run multi-account simulation with the specified configuration."""
    # Get portfolio weights
    portfolio_weights = [account.get('weight', 1.0) for account in config['accounts']]
    
    # Initialize multi-account simulator
    simulator = MultiAccountSimulator(
        backtest_engines=engines,
        portfolio_weights=portfolio_weights
    )
    
    # Prepare strategy mapping
    strategy_mapping = []
    
    for i, account in enumerate(config['accounts']):
        strategy_name = account.get('primary_strategy')
        if not strategy_name:
            # Use the first strategy if primary is not specified
            strategy_name = account.get('strategies', ['AsymmetricRiskProfile'])[0]
        
        # Get strategy parameters
        params = account.get('strategy_params', {})
        
        strategy_mapping.append({
            'account_id': i,
            'strategy': strategy_name,
            'params': params
        })
    
    # Convert date strings to timestamps if provided
    start_time = None
    if args.start_time:
        start_time = int(datetime.fromisoformat(args.start_time).timestamp())
    
    end_time = None
    if args.end_time:
        end_time = int(datetime.fromisoformat(args.end_time).timestamp())
    
    # Run portfolio backtest
    results = simulator.run_portfolio_backtest(
        strategy_mapping=strategy_mapping,
        start_time=start_time,
        end_time=end_time,
        synchronize_data=args.synchronize_data
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    simulator.save_results(os.path.join(args.output_dir, 'portfolio_results.json'))
    
    # Generate and save plots
    performance_plot = simulator.plot_portfolio_performance()
    if performance_plot:
        performance_plot.savefig(os.path.join(args.output_dir, 'portfolio_performance.png'))
    
    correlation_plot = simulator.plot_correlation_matrix()
    if correlation_plot:
        correlation_plot.savefig(os.path.join(args.output_dir, 'correlation_matrix.png'))
    
    risk_plot = simulator.plot_risk_contribution()
    if risk_plot:
        risk_plot.savefig(os.path.join(args.output_dir, 'risk_contribution.png'))
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    logger.info(f"Starting multi-account simulation with {len(config['accounts'])} accounts")
    
    # Set up backtest engines
    engines = setup_backtest_engines(config, args)
    
    # Run multi-account simulation
    results = run_multi_account_simulation(args, engines, config)
    
    # Print summary
    logger.info("Multi-account simulation completed")
    logger.info(f"Portfolio total return: {results.get('total_return', 0):.2f}%")
    logger.info(f"Portfolio max drawdown: {results.get('max_drawdown', 0):.2f}%")
    logger.info(f"Diversification benefit: {results.get('diversification_benefit', 0):.2f}%")
    
    # Print account results
    for i, account_return in enumerate(results.get('account_returns', [])):
        logger.info(f"Account {i} return: {account_return:.2f}%")
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()