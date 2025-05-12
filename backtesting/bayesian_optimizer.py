"""
Bayesian Optimization Module for Advanced Backtesting Engine

This module provides Bayesian optimization capabilities for parameter tuning
in the backtesting engine, allowing for efficient exploration of parameter spaces.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import copy
import json
import os

# Import optional dependencies with fallbacks
try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
    from hyperopt.pyll.base import scope
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    logging.warning("hyperopt not installed. Install with 'pip install hyperopt' for Bayesian optimization.")

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.plots import plot_convergence, plot_objective
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not installed. Install with 'pip install scikit-optimize' for alternative optimization methods.")

# Configure logging
logger = logging.getLogger('bayesian_optimizer')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class BayesianOptimizer:
    """
    Bayesian optimizer for tuning strategy parameters using efficient
    exploration of parameter spaces.
    """
    
    def __init__(self, backtest_engine, param_space: Dict[str, Any], 
                 objective_metric: str = 'sharpe_ratio', 
                 method: str = 'tpe', maximize: bool = True):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            backtest_engine: The backtesting engine to use for optimization
            param_space: Dictionary defining parameter space for optimization
            objective_metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
            method: Optimization method ('tpe', 'gp', 'forest')
            maximize: Whether to maximize (True) or minimize (False) the objective
        """
        self.backtest_engine = backtest_engine
        self.param_space = param_space
        self.objective_metric = objective_metric
        self.method = method
        self.maximize = maximize
        self.optimization_results = []
        self.best_params = None
        self.best_value = None
        self.trials = None
        
        # Check if required dependencies are available
        if method == 'tpe' and not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for TPE optimization. Install with 'pip install hyperopt'")
        elif method in ['gp', 'forest'] and not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for GP/Forest optimization. Install with 'pip install scikit-optimize'")
    
    def optimize(self, strategy_name: str, max_evals: int = 100, 
                early_stopping: bool = True, patience: int = 20,
                start_time: Optional[Union[int, str]] = None,
                end_time: Optional[Union[int, str]] = None,
                random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find optimal parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            max_evals: Maximum number of evaluations
            early_stopping: Whether to use early stopping
            patience: Number of evaluations without improvement before stopping
            start_time: Start time for backtest
            end_time: End time for backtest
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Bayesian optimization for {strategy_name} with {max_evals} max evaluations")
        
        self.strategy_name = strategy_name
        self.start_time = start_time
        self.end_time = end_time
        self.optimization_results = []
        
        # Define the objective function
        def objective(params):
            # Convert params to the format expected by the backtest engine
            formatted_params = self._format_params(params)
            
            # Run backtest with the parameters
            result = self.backtest_engine.run_backtest(
                strategy_name=strategy_name,
                params=formatted_params,
                start_time=start_time,
                end_time=end_time
            )
            
            if not result or self.objective_metric not in result:
                # If backtest failed or metric is missing, return worst possible value
                logger.warning(f"Backtest failed or missing metric for params: {formatted_params}")
                value = -np.inf if self.maximize else np.inf
            else:
                value = result[self.objective_metric]
                
                # Store the result
                opt_result = {
                    'params': formatted_params,
                    'value': value,
                    'iteration': len(self.optimization_results) + 1
                }
                
                # Add all metrics from the backtest result
                for key, val in result.items():
                    if key not in ['params', 'value', 'iteration']:
                        opt_result[key] = val
                
                self.optimization_results.append(opt_result)
                
                # Update best value if needed
                if self.best_value is None or (self.maximize and value > self.best_value) or (not self.maximize and value < self.best_value):
                    self.best_value = value
                    self.best_params = formatted_params
                    logger.info(f"New best value: {value:.4f} with params: {formatted_params}")
            
            # For hyperopt, we need to return a dict with 'loss' and 'status'
            if self.method == 'tpe':
                return {
                    'loss': -value if self.maximize else value,
                    'status': STATUS_OK,
                    'params': formatted_params
                }
            else:
                # For skopt, we just return the value
                return -value if self.maximize else value
        
        # Run optimization based on the selected method
        if self.method == 'tpe':
            self._run_tpe_optimization(objective, max_evals, early_stopping, patience, random_state)
        elif self.method == 'gp':
            self._run_gp_optimization(objective, max_evals, early_stopping, patience, random_state)
        elif self.method == 'forest':
            self._run_forest_optimization(objective, max_evals, early_stopping, patience, random_state)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
        logger.info(f"Optimization completed. Best {self.objective_metric}: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'results': self.optimization_results,
            'method': self.method,
            'objective_metric': self.objective_metric,
            'maximize': self.maximize
        }
    
    def _run_tpe_optimization(self, objective, max_evals, early_stopping, patience, random_state):
        """
        Run optimization using Tree-structured Parzen Estimator (TPE) from hyperopt.
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for TPE optimization")
        
        # Convert param_space to hyperopt format
        hyperopt_space = self._convert_to_hyperopt_space(self.param_space)
        
        # Create trials object to store results
        self.trials = Trials()
        
        # Define early stopping callback if needed
        if early_stopping:
            def early_stopping_callback(trials):
                # Check if we have enough trials
                if len(trials.trials) < patience:
                    return False
                
                # Get the best trial so far
                if self.maximize:
                    best_trial = min(trials.trials, key=lambda t: t['result']['loss'] if t['result']['loss'] != float('inf') else float('inf'))
                    best_iter = best_trial['tid']
                else:
                    best_trial = max(trials.trials, key=lambda t: t['result']['loss'] if t['result']['loss'] != float('-inf') else float('-inf'))
                    best_iter = best_trial['tid']
                
                # Check if we've gone patience iterations without improvement
                return len(trials.trials) - best_iter >= patience
            
            early_stop = early_stopping_callback
        else:
            early_stop = None
        
        # Run optimization
        best = fmin(
            fn=objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            rstate=np.random.RandomState(random_state) if random_state is not None else None,
            early_stop_fn=early_stop
        )
        
        # Get the best parameters
        self.best_params = space_eval(hyperopt_space, best)
    
    def _run_gp_optimization(self, objective, max_evals, early_stopping, patience, random_state):
        """
        Run optimization using Gaussian Process from scikit-optimize.
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for GP optimization")
        
        # Convert param_space to skopt format
        skopt_space = self._convert_to_skopt_space(self.param_space)
        
        # Define callback for logging and early stopping
        def callback(res):
            if len(self.optimization_results) % 10 == 0:
                logger.info(f"Completed {len(self.optimization_results)} evaluations. Best value: {res.fun:.4f}")
            
            if early_stopping and len(res.x_iters) >= patience:
                # Check if we've gone patience iterations without improvement
                best_idx = np.argmin(res.func_vals)
                if len(res.x_iters) - best_idx >= patience:
                    logger.info(f"Early stopping triggered after {len(res.x_iters)} evaluations")
                    return True
            
            return False
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=skopt_space,
            n_calls=max_evals,
            random_state=random_state,
            callback=callback
        )
        
        # Get the best parameters
        self.best_params = self._format_params(result.x)
        self.best_value = -result.fun if self.maximize else result.fun
    
    def _run_forest_optimization(self, objective, max_evals, early_stopping, patience, random_state):
        """
        Run optimization using Random Forest from scikit-optimize.
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Forest optimization")
        
        # Convert param_space to skopt format
        skopt_space = self._convert_to_skopt_space(self.param_space)
        
        # Define callback for logging and early stopping
        def callback(res):
            if len(self.optimization_results) % 10 == 0:
                logger.info(f"Completed {len(self.optimization_results)} evaluations. Best value: {res.fun:.4f}")
            
            if early_stopping and len(res.x_iters) >= patience:
                # Check if we've gone patience iterations without improvement
                best_idx = np.argmin(res.func_vals)
                if len(res.x_iters) - best_idx >= patience:
                    logger.info(f"Early stopping triggered after {len(res.x_iters)} evaluations")
                    return True
            
            return False
        
        # Run optimization
        result = forest_minimize(
            func=objective,
            dimensions=skopt_space,
            n_calls=max_evals,
            random_state=random_state,
            callback=callback
        )
        
        # Get the best parameters
        self.best_params = self._format_params(result.x)
        self.best_value = -result.fun if self.maximize else result.fun
    
    def _convert_to_hyperopt_space(self, param_space):
        """
        Convert parameter space definition to hyperopt format.
        """
        hyperopt_space = {}
        
        for param_name, param_def in param_space.items():
            if isinstance(param_def, dict):
                if 'type' not in param_def:
                    raise ValueError(f"Parameter definition for {param_name} must include 'type'")
                
                param_type = param_def['type']
                
                if param_type == 'int':
                    low = param_def.get('low', 0)
                    high = param_def.get('high', 100)
                    hyperopt_space[param_name] = scope.int(hp.quniform(param_name, low, high, 1))
                
                elif param_type == 'float':
                    low = param_def.get('low', 0.0)
                    high = param_def.get('high', 1.0)
                    log = param_def.get('log', False)
                    
                    if log:
                        hyperopt_space[param_name] = hp.loguniform(param_name, np.log(low), np.log(high))
                    else:
                        hyperopt_space[param_name] = hp.uniform(param_name, low, high)
                
                elif param_type == 'choice':
                    choices = param_def.get('choices', [])
                    hyperopt_space[param_name] = hp.choice(param_name, choices)
                
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            
            else:
                raise ValueError(f"Parameter definition for {param_name} must be a dictionary")
        
        return hyperopt_space
    
    def _convert_to_skopt_space(self, param_space):
        """
        Convert parameter space definition to scikit-optimize format.
        """
        skopt_space = []
        self.param_names = []
        
        for param_name, param_def in param_space.items():
            self.param_names.append(param_name)
            
            if isinstance(param_def, dict):
                if 'type' not in param_def:
                    raise ValueError(f"Parameter definition for {param_name} must include 'type'")
                
                param_type = param_def['type']
                
                if param_type == 'int':
                    low = param_def.get('low', 0)
                    high = param_def.get('high', 100)
                    skopt_space.append(Integer(low, high, name=param_name))
                
                elif param_type == 'float':
                    low = param_def.get('low', 0.0)
                    high = param_def.get('high', 1.0)
                    log = param_def.get('log', False)
                    
                    skopt_space.append(Real(low, high, prior='log-uniform' if log else 'uniform', name=param_name))
                
                elif param_type == 'choice':
                    choices = param_def.get('choices', [])
                    skopt_space.append(Categorical(choices, name=param_name))
                
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            
            else:
                raise ValueError(f"Parameter definition for {param_name} must be a dictionary")
        
        return skopt_space
    
    def _format_params(self, params):
        """
        Format parameters from optimization format to backtest engine format.
        """
        if self.method == 'tpe':
            # For hyperopt, params is already a dict
            return params
        else:
            # For skopt, params is a list, so we need to convert it to a dict
            return {name: value for name, value in zip(self.param_names, params)}
    
    def plot_optimization_progress(self, figsize: Tuple[int, int] = (10, 6), 
                                  smooth: int = 1) -> plt.Figure:
        """
        Plot optimization progress over iterations.
        
        Args:
            figsize: Figure size
            smooth: Smoothing window size for the curve
            
        Returns:
            Matplotlib figure
        """
        if not self.optimization_results:
            logger.error("No optimization results available")
            return None
        
        # Extract iteration and value from results
        iterations = [result['iteration'] for result in self.optimization_results]
        values = [result['value'] for result in self.optimization_results]
        
        # Calculate best value at each iteration
        if self.maximize:
            best_values = np.maximum.accumulate(values)
        else:
            best_values = np.minimum.accumulate(values)
        
        # Apply smoothing if requested
        if smooth > 1:
            smoothed_values = []
            for i in range(len(values)):
                start = max(0, i - smooth + 1)
                smoothed_values.append(np.mean(values[start:i+1]))
        else:
            smoothed_values = values
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual evaluations
        ax.scatter(iterations, values, alpha=0.5, color='blue', label='Evaluations')
        
        # Plot smoothed curve
        if smooth > 1:
            ax.plot(iterations, smoothed_values, alpha=0.7, color='green', label=f'Smoothed (window={smooth})')
        
        # Plot best value curve
        ax.plot(iterations, best_values, color='red', linewidth=2, label='Best so far')
        
        # Add labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.objective_metric)
        ax.set_title(f'Optimization Progress for {self.objective_metric}')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_importance(self, figsize: Tuple[int, int] = (12, 8), 
                                 top_n: int = 10) -> plt.Figure:
        """
        Plot parameter importance based on correlation with objective value.
        
        Args:
            figsize: Figure size
            top_n: Number of top parameters to show
            
        Returns:
            Matplotlib figure
        """
        if not self.optimization_results:
            logger.error("No optimization results available")
            return None
        
        # Extract parameters and values
        param_names = set()
        for result in self.optimization_results:
            param_names.update(result['params'].keys())
        
        param_names = list(param_names)
        correlations = {}
        
        # Calculate correlation for each parameter
        for param in param_names:
            param_values = []
            objective_values = []
            
            for result in self.optimization_results:
                if param in result['params']:
                    param_val = result['params'][param]
                    
                    # Skip non-numeric parameters
                    if not isinstance(param_val, (int, float)):
                        continue
                    
                    param_values.append(param_val)
                    objective_values.append(result['value'])
            
            if param_values:
                try:
                    corr = np.abs(np.corrcoef(param_values, objective_values)[0, 1])
                    if not np.isnan(corr):
                        correlations[param] = corr
                except:
                    pass
        
        if not correlations:
            logger.error("No valid correlations found")
            return None
        
        # Sort parameters by correlation
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top_n parameters
        if top_n > 0:
            sorted_params = sorted_params[:top_n]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot parameter importance
        params = [p[0] for p in sorted_params]
        importances = [p[1] for p in sorted_params]
        
        y_pos = np.arange(len(params))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()  # Labels read top-to-bottom
        
        # Add labels and title
        ax.set_xlabel('Absolute Correlation')
        ax.set_title('Parameter Importance')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_vs_objective(self, param_name: str, 
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot relationship between a parameter and the objective value.
        
        Args:
            param_name: Name of the parameter to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.optimization_results:
            logger.error("No optimization results available")
            return None
        
        # Extract parameter values and objective values
        param_values = []
        objective_values = []
        
        for result in self.optimization_results:
            if param_name in result['params']:
                param_val = result['params'][param_name]
                
                # Skip non-numeric parameters
                if not isinstance(param_val, (int, float)):
                    continue
                
                param_values.append(param_val)
                objective_values.append(result['value'])
        
        if not param_values:
            logger.error(f"No valid values found for parameter {param_name}")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scatter of parameter vs objective
        ax.scatter(param_values, objective_values, alpha=0.7)
        
        # Try to fit a polynomial regression
        try:
            z = np.polyfit(param_values, objective_values, 2)
            p = np.poly1d(z)
            
            # Generate points for the curve
            x_range = np.linspace(min(param_values), max(param_values), 100)
            ax.plot(x_range, p(x_range), 'r--', linewidth=2)
        except:
            pass
        
        # Add labels and title
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.objective_metric)
        ax.set_title(f'Relationship between {param_name} and {self.objective_metric}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """
        Return the optimal parameters found.
        
        Returns:
            Dictionary with optimal parameters
        """
        if self.best_params is None:
            logger.error("No optimization has been run yet")
            return {}
        
        return self.best_params
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to a file.
        
        Args:
            filepath: Path to save the results
        """
        if not self.optimization_results:
            logger.error("No optimization results available to save")
            return
        
        # Prepare results for saving
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'objective_metric': self.objective_metric,
            'method': self.method,
            'maximize': self.maximize,
            'strategy_name': getattr(self, 'strategy_name', None),
            'timestamp': datetime.now().isoformat(),
            'iterations': []
        }
        
        # Add iteration results
        for result in self.optimization_results:
            # Create a copy to avoid modifying the original
            iter_result = result.copy()
            
            # Remove complex objects that can't be easily serialized
            for key in list(iter_result.keys()):
                if key in ['equity_curve', 'drawdown_curve', 'trades']:
                    del iter_result[key]
            
            results['iterations'].append(iter_result)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved optimization results to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load optimization results from a file.
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            Dictionary with loaded results
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            # Update instance variables
            self.best_params = results.get('best_params')
            self.best_value = results.get('best_value')
            self.objective_metric = results.get('objective_metric', self.objective_metric)
            self.method = results.get('method', self.method)
            self.maximize = results.get('maximize', self.maximize)
            self.strategy_name = results.get('strategy_name')
            self.optimization_results = results.get('iterations', [])
            
            logger.info(f"Loaded optimization results from {filepath}")
            return results
        
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return {}