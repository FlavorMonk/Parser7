#!/usr/bin/env python3
"""
Bayesian Optimization Script for Advanced Backtesting Engine

This script provides a command-line interface for running Bayesian optimization
to find optimal strategy parameters using the advanced backtesting engine.
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
from backtesting.bayesian_optimizer import BayesianOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bayesian_optimization.log')
    ]
)

logger = logging.getLogger('bayesian_optimization')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Bayesian optimization for strategy parameters')
    
    # Required arguments
    parser.add_argument('--strategy', type=str, required=True,
                       help='Strategy to optimize (e.g., asymmetric, regime)')
    
    # Optional arguments
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/filtered/filtered_signals.json',
                       help='Path to the signals file')
    parser.add_argument('--market-data-dir', type=str, default='data/market_data',
                       help='Directory containing market data files')
    parser.add_argument('--output-dir', type=str, default='data/optimization_results',
                       help='Directory to save optimization results')
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
    parser.add_argument('--objective-metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'sortino_ratio', 'total_return', 'profit_factor', 'win_rate'],
                       help='Metric to optimize')
    parser.add_argument('--maximize', action='store_true', default=True,
                       help='Maximize the objective metric (default: True)')
    parser.add_argument('--method', type=str, default='tpe',
                       choices=['tpe', 'gp', 'forest'],
                       help='Optimization method')
    parser.add_argument('--max-evals', type=int, default=100,
                       help='Maximum number of evaluations')
    parser.add_argument('--early-stopping', action='store_true', default=True,
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=20,
                       help='Number of evaluations without improvement before stopping')
    parser.add_argument('--random-state', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--start-time', type=str, default=None,
                       help='Start time for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-time', type=str, default=None,
                       help='End time for backtest (YYYY-MM-DD)')
    
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
    if args.strategy.lower() == 'asymmetric':
        engine.register_strategy('AsymmetricRiskProfile')
    elif args.strategy.lower() == 'regime':
        engine.register_strategy('RegimeSwitching')
    else:
        # Try to register the strategy by name
        engine.register_strategy(args.strategy)
    
    return engine

def get_parameter_space(strategy):
    """Define parameter space for the specified strategy."""
    if strategy.lower() == 'asymmetric' or strategy == 'AsymmetricRiskProfile':
        return {
            'tp_sl_ratio_min': {'type': 'float', 'low': 1.0, 'high': 3.0},
            'tp_sl_ratio_max': {'type': 'float', 'low': 3.0, 'high': 6.0},
            'risk_per_trade': {'type': 'float', 'low': 0.5, 'high': 2.0},
            'max_daily_trades': {'type': 'int', 'low': 1, 'high': 5},
            'volatility_factor': {'type': 'float', 'low': 0.5, 'high': 2.0, 'log': True}
        }
    elif strategy.lower() == 'regime' or strategy == 'RegimeSwitching':
        return {
            'volatility_threshold': {'type': 'float', 'low': 10.0, 'high': 30.0},
            'trend_strength_threshold': {'type': 'float', 'low': 0.3, 'high': 0.7},
            'risk_multiplier_trending': {'type': 'float', 'low': 0.8, 'high': 1.5},
            'risk_multiplier_ranging': {'type': 'float', 'low': 0.5, 'high': 1.2},
            'lookback_period': {'type': 'int', 'low': 10, 'high': 50}
        }
    else:
        # Default parameter space
        return {
            'risk_per_trade': {'type': 'float', 'low': 0.5, 'high': 2.0},
            'max_positions': {'type': 'int', 'low': 1, 'high': 5},
            'entry_threshold': {'type': 'float', 'low': 0.3, 'high': 0.8},
            'exit_threshold': {'type': 'float', 'low': 0.2, 'high': 0.7}
        }

def run_bayesian_optimization(args, engine, param_space):
    """Run Bayesian optimization with the specified configuration."""
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimizer(
        backtest_engine=engine,
        param_space=param_space,
        objective_metric=args.objective_metric,
        method=args.method,
        maximize=args.maximize
    )
    
    # Convert date strings to timestamps if provided
    start_time = None
    if args.start_time:
        start_time = int(datetime.fromisoformat(args.start_time).timestamp())
    
    end_time = None
    if args.end_time:
        end_time = int(datetime.fromisoformat(args.end_time).timestamp())
    
    # Run optimization
    results = optimizer.optimize(
        strategy_name=args.strategy,
        max_evals=args.max_evals,
        early_stopping=args.early_stopping,
        patience=args.patience,
        start_time=start_time,
        end_time=end_time,
        random_state=args.random_state
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    optimizer.save_results(os.path.join(args.output_dir, f'{args.strategy}_optimization_results.json'))
    
    # Generate and save plots
    progress_plot = optimizer.plot_optimization_progress()
    if progress_plot:
        progress_plot.savefig(os.path.join(args.output_dir, f'{args.strategy}_optimization_progress.png'))
    
    importance_plot = optimizer.plot_parameter_importance()
    if importance_plot:
        importance_plot.savefig(os.path.join(args.output_dir, f'{args.strategy}_parameter_importance.png'))
    
    # Generate parameter vs objective plots for each parameter
    for param in param_space.keys():
        param_plot = optimizer.plot_parameter_vs_objective(param)
        if param_plot:
            param_plot.savefig(os.path.join(args.output_dir, f'{args.strategy}_{param}_vs_objective.png'))
    
    # Save optimal parameters
    optimal_params = optimizer.get_optimal_parameters()
    
    with open(os.path.join(args.output_dir, f'{args.strategy}_optimal_parameters.json'), 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    return results, optimal_params

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Starting Bayesian optimization for strategy: {args.strategy}")
    logger.info(f"Objective metric: {args.objective_metric} (maximize: {args.maximize})")
    logger.info(f"Maximum evaluations: {args.max_evals}")
    
    # Set up backtest engine
    engine = setup_backtest_engine(args)
    
    # Get parameter space for the strategy
    param_space = get_parameter_space(args.strategy)
    
    # Run Bayesian optimization
    results, optimal_params = run_bayesian_optimization(args, engine, param_space)
    
    # Print summary
    logger.info("Bayesian optimization completed")
    logger.info(f"Best {args.objective_metric}: {results['best_value']:.4f}")
    logger.info("Optimal parameters:")
    for param, value in optimal_params.items():
        logger.info(f"  {param}: {value}")
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()