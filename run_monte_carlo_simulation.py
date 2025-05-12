#!/usr/bin/env python3
"""
Monte Carlo Simulation Script for Advanced Backtesting Engine

This script provides a command-line interface for running Monte Carlo simulations
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
from backtesting.monte_carlo import MonteCarloSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monte_carlo.log')
    ]
)

logger = logging.getLogger('monte_carlo_simulation')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulations for backtesting')
    
    # Required arguments
    parser.add_argument('--strategy', type=str, required=True,
                       help='Strategy to simulate (e.g., asymmetric, regime)')
    
    # Optional arguments
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/filtered/filtered_signals.json',
                       help='Path to the signals file')
    parser.add_argument('--market-data-dir', type=str, default='data/market_data',
                       help='Directory containing market data files')
    parser.add_argument('--output-dir', type=str, default='data/monte_carlo_results',
                       help='Directory to save simulation results')
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
    parser.add_argument('--num-simulations', type=int, default=1000,
                       help='Number of Monte Carlo simulations to run')
    parser.add_argument('--randomize-params', action='store_true',
                       help='Randomize strategy parameters')
    parser.add_argument('--randomize-data', action='store_true',
                       help='Randomize market data (bootstrap)')
    parser.add_argument('--randomize-execution', action='store_true',
                       help='Randomize execution conditions')
    parser.add_argument('--param-variation', type=float, default=0.1,
                       help='Parameter variation percentage (0.1 = 10%)')
    parser.add_argument('--random-seed', type=int, default=None,
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
    elif args.strategy.lower() == 'all':
        engine.register_strategy('AsymmetricRiskProfile')
        engine.register_strategy('RegimeSwitching')
    else:
        # Try to register the strategy by name
        engine.register_strategy(args.strategy)
    
    return engine

def run_monte_carlo_simulations(args, engine):
    """Run Monte Carlo simulations with the specified configuration."""
    # Initialize Monte Carlo simulator
    simulator = MonteCarloSimulator(
        backtest_engine=engine,
        num_simulations=args.num_simulations,
        random_seed=args.random_seed
    )
    
    # Convert date strings to timestamps if provided
    start_time = None
    if args.start_time:
        start_time = int(datetime.fromisoformat(args.start_time).timestamp())
    
    end_time = None
    if args.end_time:
        end_time = int(datetime.fromisoformat(args.end_time).timestamp())
    
    # Run simulations
    results = simulator.run_simulations(
        strategy_name=args.strategy,
        start_time=start_time,
        end_time=end_time,
        randomize_params=args.randomize_params,
        randomize_data=args.randomize_data,
        randomize_execution=args.randomize_execution,
        param_variation=args.param_variation
    )
    
    # Generate confidence intervals
    simulator.generate_confidence_intervals(metric='total_return', confidence=0.95)
    simulator.generate_confidence_intervals(metric='max_drawdown', confidence=0.95)
    simulator.generate_confidence_intervals(metric='sharpe_ratio', confidence=0.95)
    
    # Get worst-case scenario
    worst_case = simulator.get_worst_case_scenario(risk_percentile=5.0)
    
    # Calculate success probability
    success_prob = simulator.get_success_probability(target_return=0.0)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    simulator.save_results(os.path.join(args.output_dir, f'{args.strategy}_monte_carlo_results.csv'))
    
    # Generate and save plots
    distribution_plot = simulator.plot_distribution(metric='total_return')
    if distribution_plot:
        distribution_plot.savefig(os.path.join(args.output_dir, f'{args.strategy}_return_distribution.png'))
    
    equity_plot = simulator.plot_equity_curves()
    if equity_plot:
        equity_plot.savefig(os.path.join(args.output_dir, f'{args.strategy}_equity_curves.png'))
    
    # Save summary
    summary = {
        'strategy': args.strategy,
        'num_simulations': args.num_simulations,
        'prop_firm': args.prop_firm,
        'account_size': args.account_size,
        'challenge_phase': args.challenge_phase,
        'risk_per_trade': args.risk_per_trade,
        'max_positions': args.max_positions,
        'randomize_params': args.randomize_params,
        'randomize_data': args.randomize_data,
        'randomize_execution': args.randomize_execution,
        'confidence_intervals': {
            'total_return': simulator.confidence_intervals.get('total_return', {}),
            'max_drawdown': simulator.confidence_intervals.get('max_drawdown', {}),
            'sharpe_ratio': simulator.confidence_intervals.get('sharpe_ratio', {})
        },
        'worst_case': worst_case,
        'success_probability': success_prob,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, f'{args.strategy}_monte_carlo_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results, summary

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Starting Monte Carlo simulation for strategy: {args.strategy}")
    logger.info(f"Number of simulations: {args.num_simulations}")
    
    # Set up backtest engine
    engine = setup_backtest_engine(args)
    
    # Run Monte Carlo simulations
    results, summary = run_monte_carlo_simulations(args, engine)
    
    # Print summary
    logger.info("Monte Carlo simulation completed")
    logger.info(f"Total simulations: {len(results)}")
    
    if 'total_return' in summary['confidence_intervals']:
        ci = summary['confidence_intervals']['total_return']
        logger.info(f"95% CI for Total Return: [{ci.get('lower_bound', 0):.2f}%, {ci.get('upper_bound', 0):.2f}%]")
    
    if 'max_drawdown' in summary['confidence_intervals']:
        ci = summary['confidence_intervals']['max_drawdown']
        logger.info(f"95% CI for Max Drawdown: [{ci.get('lower_bound', 0):.2f}%, {ci.get('upper_bound', 0):.2f}%]")
    
    if 'success_probability' in summary:
        logger.info(f"Probability of positive return: {summary['success_probability']:.2f}")
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()