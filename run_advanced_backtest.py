#!/usr/bin/env python3
"""
Advanced Backtesting Script

This script runs advanced backtests with the new backtesting engine,
focusing on prop firm challenge simulation and realistic execution.
"""

import argparse
import os
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Import local modules
from backtesting.backtest_engine import BacktestEngine
from backtesting.rules_engine import PropFirmRulesEngine
from backtesting.execution_simulator import ExecutionSimulator

# Import strategies
from strategies.regime_switching_strategy import RegimeSwitchingStrategy
from run_backtest_v2 import AsymmetricRiskProfileStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('advanced_backtest')

def run_single_backtest(args):
    """Run a single backtest with specified parameters."""
    logger.info(f"Running single backtest with {args.strategy} strategy")
    
    # Initialize backtest engine
    engine = BacktestEngine(
        signals_file=args.signals_file,
        market_data_dir=args.market_data_dir,
        output_dir=args.output_dir,
        prop_firm=args.prop_firm,
        account_size=args.account_size,
        challenge_phase=args.challenge_phase
    )
    
    # Register strategies
    if args.strategy == 'asymmetric' or args.strategy == 'all':
        engine.register_strategy(
            name='AsymmetricRiskProfile',
            strategy_class=AsymmetricRiskProfileStrategy,
            params={
                'initial_capital': args.account_size,
                'risk_per_trade': args.risk_per_trade,
                'trailing_stop_activation': 0.5,
                'partial_tp_ratio': 0.5
            }
        )
    
    if args.strategy == 'regime' or args.strategy == 'all':
        engine.register_strategy(
            name='RegimeSwitching',
            strategy_class=RegimeSwitchingStrategy,
            params={
                'initial_capital': args.account_size,
                'base_risk_per_trade': args.risk_per_trade,
                'max_drawdown': 0.04,
                'daily_loss_limit': 0.015,
                'profit_target': 0.08,
                'regime_detection_window': 20,
                'trend_threshold': 0.6,
                'volatility_threshold': 1.5
            }
        )
    
    # Run backtest
    if args.strategy == 'all':
        # Run multi-strategy backtest
        results = engine.run_multi_strategy_backtest(
            strategy_names=['AsymmetricRiskProfile', 'RegimeSwitching'],
            max_positions_per_strategy=args.max_positions
        )
    else:
        # Run single strategy backtest
        strategy_name = 'AsymmetricRiskProfile' if args.strategy == 'asymmetric' else 'RegimeSwitching'
        results = engine.run_backtest(
            strategy_name=strategy_name,
            max_positions=args.max_positions
        )
    
    if results:
        logger.info("Backtest completed successfully")
        
        # Print summary
        if args.strategy == 'all':
            print("\nMulti-Strategy Backtest Results:")
        else:
            print(f"\n{args.strategy.title()} Strategy Backtest Results:")
        
        print(f"Total Return: {results['performance_metrics']['total_return'] * 100:.2f}%")
        print(f"Max Drawdown: {results['performance_metrics']['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate: {results['performance_metrics']['win_rate'] * 100:.2f}%")
        print(f"Profit Factor: {results['performance_metrics']['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['performance_metrics']['sortino_ratio']:.2f}")
        print(f"Total Trades: {results['performance_metrics']['total_trades']}")
        print(f"Challenge Result: {'PASSED' if results['challenge_status']['challenge_passed'] else 'FAILED' if results['challenge_status']['challenge_failed'] else 'INCOMPLETE'}")
        
        if args.strategy == 'all' and 'metrics_by_strategy' in results:
            print("\nPerformance by Strategy:")
            for strategy, metrics in results['metrics_by_strategy'].items():
                print(f"\n{strategy}:")
                print(f"  Return: {metrics['total_return'] * 100:.2f}%")
                print(f"  Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
                print(f"  Win Rate: {metrics['win_rate'] * 100:.2f}%")
                print(f"  Trades: {metrics['total_trades']}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
    else:
        logger.error("Backtest failed")

def run_rolling_window_backtest(args):
    """Run a rolling window backtest with specified parameters."""
    logger.info(f"Running rolling window backtest with {args.strategy} strategy")
    
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
    if args.strategy == 'asymmetric':
        engine.register_strategy(
            name='AsymmetricRiskProfile',
            strategy_class=AsymmetricRiskProfileStrategy,
            params={
                'initial_capital': args.account_size,
                'risk_per_trade': args.risk_per_trade,
                'trailing_stop_activation': 0.5,
                'partial_tp_ratio': 0.5
            }
        )
        strategy_name = 'AsymmetricRiskProfile'
    else:  # regime
        engine.register_strategy(
            name='RegimeSwitching',
            strategy_class=RegimeSwitchingStrategy,
            params={
                'initial_capital': args.account_size,
                'base_risk_per_trade': args.risk_per_trade,
                'max_drawdown': 0.04,
                'daily_loss_limit': 0.015,
                'profit_target': 0.08,
                'regime_detection_window': 20,
                'trend_threshold': 0.6,
                'volatility_threshold': 1.5
            }
        )
        strategy_name = 'RegimeSwitching'
    
    # Run rolling window backtest
    results = engine.run_rolling_window_backtest(
        strategy_name=strategy_name,
        window_size=args.window_size,
        step_size=args.step_size,
        max_windows=args.max_windows,
        max_positions=args.max_positions
    )
    
    if results:
        logger.info("Rolling window backtest completed successfully")
        
        # Print summary
        print(f"\n{args.strategy.title()} Strategy Rolling Window Backtest Results:")
        print(f"Window Size: {args.window_size} days, Step Size: {args.step_size} days")
        print(f"Number of Windows: {results['num_windows']}")
        print(f"Pass Rate: {results['pass_rate'] * 100:.2f}% ({results['pass_count']}/{results['num_windows']})")
        print("\nReturns:")
        print(f"  Mean: {results['returns']['mean'] * 100:.2f}%")
        print(f"  Median: {results['returns']['median'] * 100:.2f}%")
        print(f"  Min: {results['returns']['min'] * 100:.2f}%")
        print(f"  Max: {results['returns']['max'] * 100:.2f}%")
        print(f"  Std Dev: {results['returns']['std'] * 100:.2f}%")
        
        print("\nDrawdowns:")
        print(f"  Mean: {results['drawdowns']['mean'] * 100:.2f}%")
        print(f"  Median: {results['drawdowns']['median'] * 100:.2f}%")
        print(f"  Min: {results['drawdowns']['min'] * 100:.2f}%")
        print(f"  Max: {results['drawdowns']['max'] * 100:.2f}%")
        print(f"  Std Dev: {results['drawdowns']['std'] * 100:.2f}%")
        
        print("\nWin Rates:")
        print(f"  Mean: {results['win_rates']['mean'] * 100:.2f}%")
        print(f"  Median: {results['win_rates']['median'] * 100:.2f}%")
        print(f"  Min: {results['win_rates']['min'] * 100:.2f}%")
        print(f"  Max: {results['win_rates']['max'] * 100:.2f}%")
        print(f"  Std Dev: {results['win_rates']['std'] * 100:.2f}%")
        
        if results['failure_reasons']:
            print("\nFailure Reasons:")
            for reason, count in results['failure_reasons'].items():
                print(f"  {reason}: {count} ({count / (results['num_windows'] - results['pass_count']) * 100:.2f}%)")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
    else:
        logger.error("Rolling window backtest failed")

def run_parameter_optimization(args):
    """Run parameter optimization for a strategy."""
    logger.info(f"Running parameter optimization for {args.strategy} strategy")
    
    # Define parameter grid
    if args.strategy == 'asymmetric':
        param_grid = {
            'risk_per_trade': [0.005, 0.01, 0.015, 0.02],
            'trailing_stop_activation': [0.3, 0.5, 0.7],
            'partial_tp_ratio': [0.3, 0.5, 0.7]
        }
    else:  # regime
        param_grid = {
            'base_risk_per_trade': [0.005, 0.01, 0.015],
            'trend_threshold': [0.5, 0.6, 0.7],
            'volatility_threshold': [1.2, 1.5, 1.8]
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    
    # Generate all parameter combinations
    import itertools
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    # Test each parameter combination
    for i, combination in enumerate(param_combinations):
        params = dict(zip(param_keys, combination))
        
        logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            signals_file=args.signals_file,
            market_data_dir=args.market_data_dir,
            output_dir=os.path.join(args.output_dir, f"params_{i+1}"),
            prop_firm=args.prop_firm,
            account_size=args.account_size,
            challenge_phase=args.challenge_phase
        )
        
        # Register strategy with parameters
        if args.strategy == 'asymmetric':
            engine.register_strategy(
                name='AsymmetricRiskProfile',
                strategy_class=AsymmetricRiskProfileStrategy,
                params={
                    'initial_capital': args.account_size,
                    'risk_per_trade': params['risk_per_trade'],
                    'trailing_stop_activation': params['trailing_stop_activation'],
                    'partial_tp_ratio': params['partial_tp_ratio']
                }
            )
            strategy_name = 'AsymmetricRiskProfile'
        else:  # regime
            engine.register_strategy(
                name='RegimeSwitching',
                strategy_class=RegimeSwitchingStrategy,
                params={
                    'initial_capital': args.account_size,
                    'base_risk_per_trade': params['base_risk_per_trade'],
                    'max_drawdown': 0.04,
                    'daily_loss_limit': 0.015,
                    'profit_target': 0.08,
                    'regime_detection_window': 20,
                    'trend_threshold': params['trend_threshold'],
                    'volatility_threshold': params['volatility_threshold']
                }
            )
            strategy_name = 'RegimeSwitching'
        
        # Run backtest
        results = engine.run_backtest(
            strategy_name=strategy_name,
            max_positions=args.max_positions
        )
        
        if results:
            # Extract key metrics
            metrics = {
                'params': params,
                'total_return': results['performance_metrics']['total_return'],
                'max_drawdown': results['performance_metrics']['max_drawdown'],
                'win_rate': results['performance_metrics']['win_rate'],
                'profit_factor': results['performance_metrics']['profit_factor'],
                'sharpe_ratio': results['performance_metrics']['sharpe_ratio'],
                'sortino_ratio': results['performance_metrics']['sortino_ratio'],
                'total_trades': results['performance_metrics']['total_trades'],
                'challenge_passed': results['challenge_status']['challenge_passed'],
                'challenge_failed': results['challenge_status']['challenge_failed'],
                'failure_reason': results['challenge_status']['failure_reason']
            }
            
            all_results.append(metrics)
            
            logger.info(f"Combination {i+1} results: Return={metrics['total_return']:.2%}, Drawdown={metrics['max_drawdown']:.2%}, Win Rate={metrics['win_rate']:.2%}")
    
    # Save all results
    with open(os.path.join(args.output_dir, 'optimization_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Find best parameters
    if all_results:
        # Sort by different metrics
        by_return = sorted(all_results, key=lambda x: x['total_return'], reverse=True)
        by_sharpe = sorted(all_results, key=lambda x: x['sharpe_ratio'], reverse=True)
        by_sortino = sorted(all_results, key=lambda x: x['sortino_ratio'], reverse=True)
        
        # Filter for passing challenges
        passing_results = [r for r in all_results if r['challenge_passed']]
        if passing_results:
            by_return_passing = sorted(passing_results, key=lambda x: x['total_return'], reverse=True)
        else:
            by_return_passing = []
        
        # Print summary
        print(f"\n{args.strategy.title()} Strategy Parameter Optimization Results:")
        print(f"Tested {len(param_combinations)} parameter combinations")
        
        print("\nTop 5 by Return:")
        for i, result in enumerate(by_return[:5]):
            print(f"{i+1}. Params: {result['params']}")
            print(f"   Return: {result['total_return']:.2%}, Drawdown: {result['max_drawdown']:.2%}, Win Rate: {result['win_rate']:.2%}")
            print(f"   Sharpe: {result['sharpe_ratio']:.2f}, Sortino: {result['sortino_ratio']:.2f}, Trades: {result['total_trades']}")
            print(f"   Challenge: {'PASSED' if result['challenge_passed'] else 'FAILED' if result['challenge_failed'] else 'INCOMPLETE'}")
        
        print("\nTop 5 by Sharpe Ratio:")
        for i, result in enumerate(by_sharpe[:5]):
            print(f"{i+1}. Params: {result['params']}")
            print(f"   Return: {result['total_return']:.2%}, Drawdown: {result['max_drawdown']:.2%}, Win Rate: {result['win_rate']:.2%}")
            print(f"   Sharpe: {result['sharpe_ratio']:.2f}, Sortino: {result['sortino_ratio']:.2f}, Trades: {result['total_trades']}")
            print(f"   Challenge: {'PASSED' if result['challenge_passed'] else 'FAILED' if result['challenge_failed'] else 'INCOMPLETE'}")
        
        if passing_results:
            print("\nTop 5 Passing Challenges by Return:")
            for i, result in enumerate(by_return_passing[:5]):
                print(f"{i+1}. Params: {result['params']}")
                print(f"   Return: {result['total_return']:.2%}, Drawdown: {result['max_drawdown']:.2%}, Win Rate: {result['win_rate']:.2%}")
                print(f"   Sharpe: {result['sharpe_ratio']:.2f}, Sortino: {result['sortino_ratio']:.2f}, Trades: {result['total_trades']}")
        else:
            print("\nNo parameter combinations passed the challenge")
        
        # Generate parameter heatmaps
        try:
            # Create DataFrame from results
            df = pd.DataFrame(all_results)
            
            # Get parameter names
            param_names = list(param_grid.keys())
            
            if len(param_names) >= 2:
                # Create heatmaps for pairs of parameters
                for i in range(len(param_names)):
                    for j in range(i+1, len(param_names)):
                        param1 = param_names[i]
                        param2 = param_names[j]
                        
                        # Create pivot table
                        pivot = df.pivot_table(
                            values='total_return',
                            index=f'params.{param1}',
                            columns=f'params.{param2}',
                            aggfunc='mean'
                        )
                        
                        # Plot heatmap
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='viridis')
                        plt.title(f'Return by {param1} and {param2}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.output_dir, f'heatmap_return_{param1}_{param2}.png'))
                        plt.close()
                        
                        # Repeat for drawdown
                        pivot = df.pivot_table(
                            values='max_drawdown',
                            index=f'params.{param1}',
                            columns=f'params.{param2}',
                            aggfunc='mean'
                        )
                        
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(pivot, annot=True, fmt='.2%', cmap='rocket_r')
                        plt.title(f'Drawdown by {param1} and {param2}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.output_dir, f'heatmap_drawdown_{param1}_{param2}.png'))
                        plt.close()
        except Exception as e:
            logger.error(f"Error generating parameter heatmaps: {str(e)}")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
    else:
        logger.error("No valid results from parameter optimization")

def main():
    parser = argparse.ArgumentParser(description='Run advanced backtests')
    
    # Common arguments
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/filtered/filtered_signals.json',
                        help='Path to filtered signals JSON file')
    parser.add_argument('--market-data-dir', type=str, default='data/market_data',
                        help='Directory containing market data files')
    parser.add_argument('--output-dir', type=str, default='data/advanced_backtest_results',
                        help='Directory to save backtest results')
    parser.add_argument('--prop-firm', type=str, default='TFT',
                        choices=['TFT', 'FTMO', 'MFF'],
                        help='Prop firm type')
    parser.add_argument('--account-size', type=float, default=100000.0,
                        help='Account size')
    parser.add_argument('--challenge-phase', type=str, default='phase1',
                        choices=['phase1', 'phase2', 'verification', 'funded'],
                        help='Challenge phase')
    parser.add_argument('--strategy', type=str, default='regime',
                        choices=['asymmetric', 'regime', 'all'],
                        help='Strategy to backtest')
    parser.add_argument('--risk-per-trade', type=float, default=0.01,
                        help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--max-positions', type=int, default=5,
                        help='Maximum number of concurrent positions')
    
    # Subparsers for different backtest modes
    subparsers = parser.add_subparsers(dest='mode', help='Backtest mode')
    
    # Single backtest
    single_parser = subparsers.add_parser('single', help='Run a single backtest')
    
    # Rolling window backtest
    rolling_parser = subparsers.add_parser('rolling', help='Run a rolling window backtest')
    rolling_parser.add_argument('--window-size', type=int, default=30,
                              help='Window size in days')
    rolling_parser.add_argument('--step-size', type=int, default=5,
                              help='Step size in days')
    rolling_parser.add_argument('--max-windows', type=int, default=10,
                              help='Maximum number of windows')
    
    # Parameter optimization
    optimize_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run appropriate backtest mode
    if args.mode == 'rolling':
        run_rolling_window_backtest(args)
    elif args.mode == 'optimize':
        run_parameter_optimization(args)
    else:  # single or default
        run_single_backtest(args)

if __name__ == "__main__":
    main()