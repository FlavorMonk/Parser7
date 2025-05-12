"""
Multi-Account Simulation Module for Advanced Backtesting Engine

This module provides multi-account simulation capabilities for the backtesting engine,
allowing for portfolio-level analysis and risk management.
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

# Configure logging
logger = logging.getLogger('multi_account')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MultiAccountSimulator:
    """
    Multi-account simulator for backtesting strategies across multiple accounts
    with portfolio-level analysis and risk management.
    """
    
    def __init__(self, backtest_engines: List[Any], portfolio_weights: Optional[List[float]] = None):
        """
        Initialize the multi-account simulator.
        
        Args:
            backtest_engines: List of backtesting engine instances
            portfolio_weights: Optional list of weights for each account (defaults to equal weights)
        """
        self.backtest_engines = backtest_engines
        
        # Set default equal weights if not provided
        if portfolio_weights is None:
            self.portfolio_weights = [1.0 / len(backtest_engines)] * len(backtest_engines)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(portfolio_weights)
            self.portfolio_weights = [w / total_weight for w in portfolio_weights]
        
        self.portfolio_results = {}
        self.account_results = []
        self.correlation_matrix = None
    
    def run_portfolio_backtest(self, 
                              strategy_mapping: List[Dict[str, Any]],
                              start_time: Optional[Union[int, str]] = None,
                              end_time: Optional[Union[int, str]] = None,
                              synchronize_data: bool = True) -> Dict[str, Any]:
        """
        Run backtest across multiple accounts with different strategies.
        
        Args:
            strategy_mapping: List of dictionaries mapping strategies to accounts
                             [{'account_id': 0, 'strategy': 'Strategy1', 'params': {...}}, ...]
            start_time: Start time for backtest
            end_time: End time for backtest
            synchronize_data: Whether to synchronize market data across accounts
            
        Returns:
            Dictionary with portfolio-level results
        """
        if len(strategy_mapping) != len(self.backtest_engines):
            logger.warning(f"Strategy mapping length ({len(strategy_mapping)}) doesn't match number of accounts ({len(self.backtest_engines)})")
        
        logger.info(f"Running portfolio backtest across {len(self.backtest_engines)} accounts")
        
        # Synchronize market data if requested
        if synchronize_data:
            self._synchronize_market_data()
        
        # Run backtest for each account
        self.account_results = []
        
        for i, engine in enumerate(self.backtest_engines):
            # Find strategy mapping for this account
            strategy_config = next((s for s in strategy_mapping if s.get('account_id') == i), None)
            
            if strategy_config is None:
                logger.warning(f"No strategy mapping found for account {i}, skipping")
                self.account_results.append(None)
                continue
            
            strategy_name = strategy_config.get('strategy')
            params = strategy_config.get('params')
            
            logger.info(f"Running backtest for account {i} with strategy {strategy_name}")
            
            # Run backtest
            result = engine.run_backtest(
                strategy_name=strategy_name,
                params=params,
                start_time=start_time,
                end_time=end_time
            )
            
            if result:
                # Add account info to result
                result['account_id'] = i
                result['account_weight'] = self.portfolio_weights[i]
                result['strategy_name'] = strategy_name
                
                self.account_results.append(result)
            else:
                logger.error(f"Backtest failed for account {i}")
                self.account_results.append(None)
        
        # Calculate portfolio-level metrics
        self.calculate_portfolio_metrics()
        
        # Analyze correlations
        self.analyze_correlations()
        
        return self.portfolio_results
    
    def _synchronize_market_data(self):
        """
        Synchronize market data across all accounts to ensure consistent backtesting.
        """
        # This implementation depends on how market data is stored in the engine
        # Here's a simple approach that assumes market_data is a dictionary attribute
        
        if not self.backtest_engines:
            return
        
        # Use the first engine's market data as reference
        reference_engine = self.backtest_engines[0]
        
        if hasattr(reference_engine, 'market_data') and reference_engine.market_data:
            reference_data = reference_engine.market_data
            
            # Copy to all other engines
            for engine in self.backtest_engines[1:]:
                if hasattr(engine, 'market_data'):
                    engine.market_data = copy.deepcopy(reference_data)
                    logger.info(f"Synchronized market data for engine {id(engine)}")
    
    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculate portfolio-level performance metrics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        if not self.account_results or all(r is None for r in self.account_results):
            logger.error("No valid account results available")
            return {}
        
        # Initialize portfolio results
        self.portfolio_results = {
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0,
            'equity_curve': None,
            'drawdown_curve': None,
            'account_returns': [],
            'account_drawdowns': [],
            'account_sharpes': [],
            'account_sortinos': [],
            'account_win_rates': [],
            'account_profit_factors': [],
            'account_num_trades': [],
            'diversification_benefit': 0.0
        }
        
        # Extract equity curves from each account
        equity_curves = []
        weights = []
        
        for i, result in enumerate(self.account_results):
            if result is None or 'equity_curve' not in result:
                continue
            
            equity_curve = result.get('equity_curve', [])
            
            if equity_curve:
                equity_curves.append(equity_curve)
                weights.append(self.portfolio_weights[i])
                
                # Store individual account metrics
                self.portfolio_results['account_returns'].append(result.get('total_return', 0.0))
                self.portfolio_results['account_drawdowns'].append(result.get('max_drawdown', 0.0))
                self.portfolio_results['account_sharpes'].append(result.get('sharpe_ratio', 0.0))
                self.portfolio_results['account_sortinos'].append(result.get('sortino_ratio', 0.0))
                self.portfolio_results['account_win_rates'].append(result.get('win_rate', 0.0))
                self.portfolio_results['account_profit_factors'].append(result.get('profit_factor', 0.0))
                self.portfolio_results['account_num_trades'].append(result.get('num_trades', 0))
        
        if not equity_curves:
            logger.error("No valid equity curves available")
            return self.portfolio_results
        
        # Standardize equity curves to the same length
        min_length = min(len(curve) for curve in equity_curves)
        standardized_curves = [curve[:min_length] for curve in equity_curves]
        
        # Calculate weighted portfolio equity curve
        portfolio_equity = np.zeros(min_length)
        for i, curve in enumerate(standardized_curves):
            portfolio_equity += np.array(curve) * weights[i]
        
        self.portfolio_results['equity_curve'] = portfolio_equity.tolist()
        
        # Calculate portfolio drawdown curve
        portfolio_drawdown = self._calculate_drawdown(portfolio_equity)
        self.portfolio_results['drawdown_curve'] = portfolio_drawdown.tolist()
        
        # Calculate portfolio metrics
        self.portfolio_results['total_return'] = (portfolio_equity[-1] / portfolio_equity[0] - 1) * 100
        self.portfolio_results['max_drawdown'] = np.min(portfolio_drawdown) * 100
        
        # Calculate Sharpe and Sortino ratios
        returns = np.diff(portfolio_equity) / portfolio_equity[:-1]
        if len(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                self.portfolio_results['sharpe_ratio'] = avg_return / std_return * np.sqrt(252)  # Annualized
            
            # Sortino ratio uses only negative returns for denominator
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                std_negative = np.std(negative_returns)
                if std_negative > 0:
                    self.portfolio_results['sortino_ratio'] = avg_return / std_negative * np.sqrt(252)  # Annualized
        
        # Calculate diversification benefit
        weighted_avg_return = sum(r * w for r, w in zip(self.portfolio_results['account_returns'], weights))
        diversification_benefit = self.portfolio_results['total_return'] - weighted_avg_return
        self.portfolio_results['diversification_benefit'] = diversification_benefit
        
        # Aggregate trade statistics
        total_trades = sum(self.portfolio_results['account_num_trades'])
        self.portfolio_results['num_trades'] = total_trades
        
        # Calculate weighted average win rate and profit factor
        if total_trades > 0:
            weighted_win_rate = sum(wr * nt / total_trades for wr, nt in 
                                   zip(self.portfolio_results['account_win_rates'], 
                                       self.portfolio_results['account_num_trades']))
            self.portfolio_results['win_rate'] = weighted_win_rate
        
        # Profit factor is more complex, would need actual trade data
        # This is a simplified approximation
        if self.portfolio_results['account_profit_factors']:
            self.portfolio_results['profit_factor'] = np.mean(self.portfolio_results['account_profit_factors'])
        
        logger.info(f"Portfolio total return: {self.portfolio_results['total_return']:.2f}%")
        logger.info(f"Portfolio max drawdown: {self.portfolio_results['max_drawdown']:.2f}%")
        logger.info(f"Diversification benefit: {diversification_benefit:.2f}%")
        
        return self.portfolio_results
    
    def _calculate_drawdown(self, equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown curve from equity curve.
        
        Args:
            equity_curve: Numpy array of equity values
            
        Returns:
            Numpy array of drawdown values (negative percentages)
        """
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown in percentage terms
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown
    
    def analyze_correlations(self) -> np.ndarray:
        """
        Analyze correlations between different accounts/strategies.
        
        Returns:
            Correlation matrix as numpy array
        """
        if not self.account_results or all(r is None for r in self.account_results):
            logger.error("No valid account results available")
            return np.array([])
        
        # Extract returns from each account
        return_series = []
        
        for result in self.account_results:
            if result is None or 'equity_curve' not in result:
                continue
            
            equity_curve = result.get('equity_curve', [])
            
            if len(equity_curve) > 1:
                # Calculate returns
                returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
                return_series.append(returns)
        
        if not return_series:
            logger.error("No valid return series available")
            return np.array([])
        
        # Standardize return series to the same length
        min_length = min(len(series) for series in return_series)
        standardized_series = [series[:min_length] for series in return_series]
        
        # Calculate correlation matrix
        returns_matrix = np.vstack(standardized_series)
        correlation_matrix = np.corrcoef(returns_matrix)
        
        self.correlation_matrix = correlation_matrix
        
        logger.info(f"Calculated correlation matrix of shape {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def plot_portfolio_performance(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot portfolio and individual account performance.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.portfolio_results or 'equity_curve' not in self.portfolio_results:
            logger.error("No portfolio results available")
            return None
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio equity curve
        portfolio_equity = self.portfolio_results['equity_curve']
        ax1.plot(portfolio_equity, linewidth=2, color='black', label='Portfolio')
        
        # Plot individual account equity curves
        for i, result in enumerate(self.account_results):
            if result is None or 'equity_curve' not in result:
                continue
            
            equity_curve = result.get('equity_curve', [])
            
            if equity_curve:
                # Truncate to the same length as portfolio curve
                truncated_curve = equity_curve[:len(portfolio_equity)]
                ax1.plot(truncated_curve, alpha=0.5, 
                        label=f"Account {i}: {result.get('strategy_name', 'Unknown')}")
        
        # Add labels and title for equity plot
        ax1.set_title('Portfolio and Account Equity Curves')
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot portfolio drawdown
        portfolio_drawdown = self.portfolio_results['drawdown_curve']
        ax2.fill_between(range(len(portfolio_drawdown)), 0, portfolio_drawdown, 
                        color='red', alpha=0.3)
        ax2.plot(portfolio_drawdown, color='red', linewidth=1)
        
        # Add labels for drawdown plot
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown')
        ax2.set_ylim(min(portfolio_drawdown) * 1.1, 0.01)  # Add some padding
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot correlation matrix between accounts.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.correlation_matrix is None:
            logger.error("No correlation matrix available")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create labels for accounts
        labels = [f"Account {i}" for i in range(len(self.correlation_matrix))]
        
        # Plot heatmap
        im = ax.imshow(self.correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
        
        # Add ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values in each cell
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f"{self.correlation_matrix[i, j]:.2f}",
                              ha="center", va="center", color="black" if abs(self.correlation_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title("Account Return Correlation Matrix")
        
        plt.tight_layout()
        return fig
    
    def plot_risk_contribution(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot risk contribution of each account to the portfolio.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.portfolio_results or not self.correlation_matrix is not None:
            logger.error("No portfolio results or correlation matrix available")
            return None
        
        # Extract account volatilities
        volatilities = []
        
        for result in self.account_results:
            if result is None or 'equity_curve' not in result:
                volatilities.append(0.0)
                continue
            
            equity_curve = result.get('equity_curve', [])
            
            if len(equity_curve) > 1:
                # Calculate returns
                returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                volatilities.append(volatility)
            else:
                volatilities.append(0.0)
        
        # Calculate risk contribution
        weights = np.array(self.portfolio_weights)
        vols = np.array(volatilities)
        
        # Handle zero volatilities
        vols = np.where(vols == 0, 1e-10, vols)
        
        # Calculate portfolio volatility
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 0:
            corr_matrix = self.correlation_matrix
            cov_matrix = np.outer(vols, vols) * corr_matrix
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # Calculate marginal contribution to risk
            mcr = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Calculate risk contribution
            rc = weights * mcr
            
            # Normalize to percentage
            risk_contrib_pct = rc / portfolio_vol * 100
        else:
            # Fallback if correlation matrix is not available
            weighted_vols = weights * vols
            portfolio_vol = np.sum(weighted_vols)
            risk_contrib_pct = weighted_vols / portfolio_vol * 100 if portfolio_vol > 0 else np.zeros_like(weights)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create labels for accounts
        labels = [f"Account {i}" for i in range(len(weights))]
        
        # Plot risk contribution
        ax.bar(labels, risk_contrib_pct)
        
        # Add labels and title
        ax.set_xlabel('Account')
        ax.set_ylabel('Risk Contribution (%)')
        ax.set_title('Risk Contribution by Account')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add weight and volatility annotations
        for i, (w, v) in enumerate(zip(weights, vols)):
            ax.annotate(f"Weight: {w:.2f}\nVol: {v:.2%}", 
                       (i, risk_contrib_pct[i]), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')
        
        plt.tight_layout()
        return fig
    
    def save_results(self, filepath: str) -> None:
        """
        Save portfolio results to a file.
        
        Args:
            filepath: Path to save the results
        """
        if not self.portfolio_results:
            logger.error("No portfolio results available to save")
            return
        
        # Prepare results for saving
        results = {
            'portfolio': self.portfolio_results.copy(),
            'accounts': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Remove complex objects that can't be easily serialized
        for key in ['equity_curve', 'drawdown_curve']:
            if key in results['portfolio']:
                del results['portfolio'][key]
        
        # Add account results
        for i, result in enumerate(self.account_results):
            if result is None:
                continue
            
            # Create a copy to avoid modifying the original
            account_result = result.copy()
            
            # Remove complex objects
            for key in ['equity_curve', 'drawdown_curve', 'trades']:
                if key in account_result:
                    del account_result[key]
            
            results['accounts'].append(account_result)
        
        # Add correlation matrix if available
        if self.correlation_matrix is not None:
            results['correlation_matrix'] = self.correlation_matrix.tolist()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved portfolio results to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load portfolio results from a file.
        
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
            self.portfolio_results = results.get('portfolio', {})
            self.account_results = results.get('accounts', [])
            
            # Convert correlation matrix back to numpy array if available
            if 'correlation_matrix' in results:
                self.correlation_matrix = np.array(results['correlation_matrix'])
            
            logger.info(f"Loaded portfolio results from {filepath}")
            return results
        
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return {}