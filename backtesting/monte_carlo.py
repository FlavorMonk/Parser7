"""
Monte Carlo Simulation Module for Advanced Backtesting Engine

This module provides Monte Carlo simulation capabilities for the backtesting engine,
allowing for robust assessment of strategy performance under various conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger('monte_carlo')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtesting strategies with randomization
    to assess robustness and generate confidence intervals.
    """
    
    def __init__(self, backtest_engine, num_simulations: int = 1000, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            backtest_engine: The backtesting engine to use for simulations
            num_simulations: Number of Monte Carlo simulations to run
            random_seed: Optional seed for reproducibility
        """
        self.backtest_engine = backtest_engine
        self.num_simulations = num_simulations
        self.simulation_results = []
        self.confidence_intervals = {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def run_simulations(self, 
                        strategy_name: str, 
                        params: Dict[str, Any] = None,
                        start_time: Optional[Union[int, str]] = None,
                        end_time: Optional[Union[int, str]] = None,
                        randomize_params: bool = False,
                        randomize_data: bool = True,
                        randomize_execution: bool = True,
                        param_variation: float = 0.1,
                        max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Run Monte Carlo simulations with various randomization options.
        
        Args:
            strategy_name: Name of the strategy to simulate
            params: Strategy parameters
            start_time: Start time for backtest
            end_time: End time for backtest
            randomize_params: Whether to randomize strategy parameters
            randomize_data: Whether to randomize market data (bootstrap)
            randomize_execution: Whether to randomize execution conditions
            param_variation: Percentage variation for parameter randomization
            max_workers: Maximum number of parallel workers (None = auto)
            
        Returns:
            List of simulation results
        """
        logger.info(f"Starting {self.num_simulations} Monte Carlo simulations for {strategy_name}")
        
        self.simulation_results = []
        
        # Create simulation configurations
        simulation_configs = []
        for i in range(self.num_simulations):
            sim_config = {
                'sim_id': i,
                'strategy_name': strategy_name,
                'params': self._randomize_params(params, param_variation) if randomize_params else params,
                'start_time': start_time,
                'end_time': end_time,
                'randomize_data': randomize_data,
                'randomize_execution': randomize_execution
            }
            simulation_configs.append(sim_config)
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._run_single_simulation, config) 
                      for config in simulation_configs]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        self.simulation_results.append(result)
                    
                    if (i + 1) % 100 == 0 or (i + 1) == self.num_simulations:
                        logger.info(f"Completed {i + 1}/{self.num_simulations} simulations")
                except Exception as e:
                    logger.error(f"Error in simulation: {str(e)}")
        
        logger.info(f"Completed {len(self.simulation_results)}/{self.num_simulations} successful simulations")
        return self.simulation_results
    
    def _run_single_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single Monte Carlo simulation with the given configuration.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Simulation result
        """
        # Create a deep copy of the backtest engine to avoid state interference
        engine_copy = copy.deepcopy(self.backtest_engine)
        
        # Apply randomizations
        if config['randomize_data']:
            self._randomize_market_data(engine_copy)
        
        if config['randomize_execution']:
            self._randomize_execution_conditions(engine_copy)
        
        # Run backtest
        result = engine_copy.run_backtest(
            strategy_name=config['strategy_name'],
            params=config['params'],
            start_time=config['start_time'],
            end_time=config['end_time']
        )
        
        if result:
            result['sim_id'] = config['sim_id']
            return result
        
        return None
    
    def _randomize_params(self, params: Dict[str, Any], variation: float) -> Dict[str, Any]:
        """
        Randomize strategy parameters within a specified variation range.
        
        Args:
            params: Original parameters
            variation: Percentage variation
            
        Returns:
            Randomized parameters
        """
        if not params:
            return {}
        
        randomized = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Apply random variation within the specified percentage
                min_val = value * (1 - variation)
                max_val = value * (1 + variation)
                
                if isinstance(value, int):
                    randomized[key] = int(np.random.uniform(min_val, max_val))
                else:
                    randomized[key] = np.random.uniform(min_val, max_val)
            else:
                # Non-numeric parameters are kept as is
                randomized[key] = value
                
        return randomized
    
    def _randomize_market_data(self, engine):
        """
        Randomize market data using bootstrap resampling or other techniques.
        
        Args:
            engine: Backtest engine instance
        """
        # Implementation depends on how market data is stored in the engine
        # This is a placeholder for the actual implementation
        if hasattr(engine, 'market_data') and engine.market_data:
            for symbol, data in engine.market_data.items():
                if isinstance(data, pd.DataFrame):
                    # Bootstrap resampling with replacement
                    indices = np.random.choice(len(data), size=len(data), replace=True)
                    engine.market_data[symbol] = data.iloc[indices].sort_index()
    
    def _randomize_execution_conditions(self, engine):
        """
        Randomize execution conditions like slippage, spread, etc.
        
        Args:
            engine: Backtest engine instance
        """
        # Implementation depends on the execution simulator in the engine
        if hasattr(engine, 'execution_simulator'):
            # Randomize slippage model parameters
            if hasattr(engine.execution_simulator, 'slippage_factor'):
                engine.execution_simulator.slippage_factor *= np.random.uniform(0.8, 1.2)
            
            # Randomize spread model parameters
            if hasattr(engine.execution_simulator, 'spread_factor'):
                engine.execution_simulator.spread_factor *= np.random.uniform(0.8, 1.2)
    
    def generate_confidence_intervals(self, metric: str = 'total_return', confidence: float = 0.95) -> Dict[str, float]:
        """
        Generate confidence intervals for performance metrics.
        
        Args:
            metric: Performance metric to analyze
            confidence: Confidence level (0-1)
            
        Returns:
            Dictionary with lower and upper bounds
        """
        if not self.simulation_results:
            logger.error("No simulation results available")
            return {}
        
        # Extract the metric from all simulation results
        metric_values = []
        for result in self.simulation_results:
            if metric in result:
                metric_values.append(result[metric])
        
        if not metric_values:
            logger.error(f"Metric '{metric}' not found in simulation results")
            return {}
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(metric_values, lower_percentile)
        upper_bound = np.percentile(metric_values, upper_percentile)
        mean_value = np.mean(metric_values)
        median_value = np.median(metric_values)
        
        interval = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'mean': mean_value,
            'median': median_value,
            'confidence': confidence
        }
        
        self.confidence_intervals[metric] = interval
        
        logger.info(f"{confidence*100}% confidence interval for {metric}: [{lower_bound:.2f}, {upper_bound:.2f}]")
        return interval
    
    def plot_distribution(self, metric: str = 'total_return', bins: int = 50, 
                         figsize: Tuple[int, int] = (10, 6), 
                         show_intervals: bool = True) -> plt.Figure:
        """
        Plot distribution of simulation results for a specific metric.
        
        Args:
            metric: Performance metric to plot
            bins: Number of histogram bins
            figsize: Figure size
            show_intervals: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure
        """
        if not self.simulation_results:
            logger.error("No simulation results available")
            return None
        
        # Extract the metric from all simulation results
        metric_values = []
        for result in self.simulation_results:
            if metric in result:
                metric_values.append(result[metric])
        
        if not metric_values:
            logger.error(f"Metric '{metric}' not found in simulation results")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        ax.hist(metric_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add mean and median lines
        mean_value = np.mean(metric_values)
        median_value = np.median(metric_values)
        
        ax.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
        ax.axvline(median_value, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_value:.2f}')
        
        # Add confidence intervals if requested
        if show_intervals and metric in self.confidence_intervals:
            interval = self.confidence_intervals[metric]
            ax.axvline(interval['lower_bound'], color='purple', linestyle=':', linewidth=2, 
                      label=f"{interval['confidence']*100}% CI Lower: {interval['lower_bound']:.2f}")
            ax.axvline(interval['upper_bound'], color='purple', linestyle=':', linewidth=2,
                      label=f"{interval['confidence']*100}% CI Upper: {interval['upper_bound']:.2f}")
        
        # Add labels and title
        ax.set_xlabel(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()} across {len(metric_values)} Simulations')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_equity_curves(self, max_curves: int = 100, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot equity curves from multiple simulations.
        
        Args:
            max_curves: Maximum number of curves to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.simulation_results:
            logger.error("No simulation results available")
            return None
        
        # Check if equity curves are available
        curves_available = [result for result in self.simulation_results if 'equity_curve' in result]
        
        if not curves_available:
            logger.error("No equity curves found in simulation results")
            return None
        
        # Sample curves if there are too many
        if len(curves_available) > max_curves:
            sampled_results = random.sample(curves_available, max_curves)
        else:
            sampled_results = curves_available
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual equity curves with low opacity
        for i, result in enumerate(sampled_results):
            equity_curve = result['equity_curve']
            if isinstance(equity_curve, list) and equity_curve:
                # Convert to numpy array if it's a list
                equity_array = np.array(equity_curve)
                ax.plot(equity_array, alpha=0.1, color='blue')
        
        # Calculate and plot the average equity curve
        all_curves = [np.array(result['equity_curve']) for result in curves_available 
                     if isinstance(result.get('equity_curve'), list) and result['equity_curve']]
        
        # Ensure all curves have the same length for averaging
        min_length = min(len(curve) for curve in all_curves)
        standardized_curves = [curve[:min_length] for curve in all_curves]
        
        if standardized_curves:
            avg_curve = np.mean(standardized_curves, axis=0)
            ax.plot(avg_curve, linewidth=2, color='red', label='Average Equity Curve')
            
            # Calculate percentiles for confidence bands
            lower_band = np.percentile(standardized_curves, 5, axis=0)
            upper_band = np.percentile(standardized_curves, 95, axis=0)
            
            ax.fill_between(range(len(avg_curve)), lower_band, upper_band, 
                           color='red', alpha=0.2, label='90% Confidence Band')
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity')
        ax.set_title(f'Equity Curves from {len(sampled_results)} Monte Carlo Simulations')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_worst_case_scenario(self, risk_percentile: float = 5.0) -> Dict[str, Any]:
        """
        Get the worst-case scenario based on a risk percentile.
        
        Args:
            risk_percentile: Percentile to use for worst-case (lower is more conservative)
            
        Returns:
            Worst-case scenario metrics
        """
        if not self.simulation_results:
            logger.error("No simulation results available")
            return {}
        
        # Define key risk metrics (lower is worse)
        risk_metrics = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']
        
        worst_case = {}
        for metric in risk_metrics:
            values = [result.get(metric, 0) for result in self.simulation_results]
            
            if not values:
                continue
                
            # For drawdown, higher is worse, so we use the opposite percentile
            if metric == 'max_drawdown':
                worst_case[metric] = np.percentile(values, 100 - risk_percentile)
            else:
                worst_case[metric] = np.percentile(values, risk_percentile)
        
        # Find the simulation that most closely matches the worst-case scenario
        if 'total_return' in worst_case:
            worst_return = worst_case['total_return']
            closest_sim = min(self.simulation_results, 
                             key=lambda x: abs(x.get('total_return', 0) - worst_return))
            
            worst_case['closest_simulation'] = closest_sim.get('sim_id')
            
        return worst_case
    
    def get_success_probability(self, target_return: float = 0.0) -> float:
        """
        Calculate the probability of achieving a target return.
        
        Args:
            target_return: Target return threshold
            
        Returns:
            Probability (0-1) of achieving the target return
        """
        if not self.simulation_results:
            logger.error("No simulation results available")
            return 0.0
        
        # Count simulations that achieved the target return
        successful_sims = sum(1 for result in self.simulation_results 
                             if result.get('total_return', 0) >= target_return)
        
        probability = successful_sims / len(self.simulation_results)
        
        logger.info(f"Probability of achieving {target_return:.2f}% return: {probability:.2f}")
        return probability
    
    def analyze_parameter_sensitivity(self, param_name: str) -> Dict[str, Any]:
        """
        Analyze sensitivity of results to a specific parameter.
        
        Args:
            param_name: Name of the parameter to analyze
            
        Returns:
            Sensitivity analysis results
        """
        if not self.simulation_results:
            logger.error("No simulation results available")
            return {}
        
        # Extract parameter values and corresponding returns
        param_values = []
        returns = []
        
        for result in self.simulation_results:
            if 'params' in result and param_name in result['params'] and 'total_return' in result:
                param_values.append(result['params'][param_name])
                returns.append(result['total_return'])
        
        if not param_values:
            logger.error(f"Parameter '{param_name}' not found in simulation results")
            return {}
        
        # Calculate correlation
        correlation = np.corrcoef(param_values, returns)[0, 1]
        
        # Group by parameter value ranges
        param_array = np.array(param_values)
        return_array = np.array(returns)
        
        # Create bins for parameter values
        num_bins = min(10, len(set(param_values)))
        bins = np.linspace(min(param_values), max(param_values), num_bins + 1)
        
        bin_indices = np.digitize(param_array, bins)
        
        # Calculate average return for each bin
        bin_returns = {}
        for i in range(1, num_bins + 1):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                bin_returns[f"{bins[i-1]:.2f}-{bins[i]:.2f}"] = np.mean(return_array[bin_mask])
        
        return {
            'correlation': correlation,
            'bin_returns': bin_returns,
            'param_values': param_values,
            'returns': returns
        }
    
    def save_results(self, filepath: str) -> None:
        """
        Save simulation results to a file.
        
        Args:
            filepath: Path to save the results
        """
        if not self.simulation_results:
            logger.error("No simulation results available to save")
            return
        
        # Convert to DataFrame for easier saving
        results_df = pd.DataFrame(self.simulation_results)
        
        # Remove complex objects that can't be easily serialized
        for col in results_df.columns:
            if col in ['equity_curve', 'drawdown_curve', 'trades']:
                results_df = results_df.drop(columns=[col])
        
        # Save to file
        results_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.simulation_results)} simulation results to {filepath}")