# Sprint 5: Advanced Backtesting Enhancements

## Overview

Sprint 5 builds on the advanced backtesting engine implemented in Sprint 4, adding sophisticated analysis capabilities, optimization techniques, and machine learning integration for adaptive strategy selection. These enhancements enable more robust strategy development, parameter optimization, and portfolio-level risk management.

## Key Components

### 1. Monte Carlo Simulation

The `MonteCarloSimulator` class in `backtesting/monte_carlo.py` provides robust Monte Carlo simulation capabilities to assess strategy robustness under various conditions:

- **Parameter Randomization**: Stress test strategies by varying parameters within specified ranges
- **Market Data Randomization**: Bootstrap resampling of market data to simulate different market conditions
- **Execution Randomization**: Vary execution conditions like slippage and spread to simulate real-world trading
- **Confidence Intervals**: Generate confidence intervals for key performance metrics
- **Worst-Case Analysis**: Identify worst-case scenarios and calculate success probabilities

```bash
# Example usage
./run_monte_carlo_simulation.py --strategy asymmetric --num-simulations 1000 --randomize-params --randomize-execution
```

### 2. Bayesian Parameter Optimization

The `BayesianOptimizer` class in `backtesting/bayesian_optimizer.py` implements efficient parameter tuning using Bayesian optimization:

- **Multiple Optimization Methods**: Support for TPE (Tree-structured Parzen Estimator), GP (Gaussian Process), and Random Forest
- **Flexible Parameter Space**: Define continuous, discrete, and categorical parameters with constraints
- **Early Stopping**: Intelligent early stopping based on convergence criteria
- **Parameter Importance Analysis**: Identify which parameters have the most impact on performance
- **Visualization**: Plot optimization progress and parameter relationships

```bash
# Example usage
./run_bayesian_optimization.py --strategy regime --objective-metric sharpe_ratio --method tpe --max-evals 100
```

### 3. Multi-Account Simulation

The `MultiAccountSimulator` class in `backtesting/multi_account.py` enables portfolio-level analysis across multiple accounts:

- **Portfolio Weighting**: Specify weights for different accounts in a portfolio
- **Correlation Analysis**: Analyze correlations between different strategies/accounts
- **Diversification Benefit**: Calculate the benefit of diversification across accounts
- **Risk Contribution**: Analyze how each account contributes to portfolio risk
- **Synchronized Data**: Option to synchronize market data across accounts for consistent backtesting

```bash
# Example usage
./run_multi_account_simulation.py --config-file configs/multi_account_config.json
```

### 4. Market Regime Detection

The `MarketRegimeDetector` class in `backtesting/market_regime.py` implements market regime detection and adaptive strategy selection:

- **Multiple Detection Methods**: Support for K-means clustering, Gaussian Mixture Models, and Hidden Markov Models
- **Feature Engineering**: Calculate various technical features for regime detection
- **Regime Visualization**: Plot detected regimes over time and analyze their characteristics
- **Adaptive Strategy Selection**: Run backtests with strategy selection based on detected regimes
- **Cross-Asset Analysis**: Option to include cross-asset features for more robust regime detection

```bash
# Example usage
./run_market_regime_detection.py --market-data-dir data/market_data --n-regimes 3 --method hmm --run-adaptive-backtest
```

## Configuration Files

### Multi-Account Configuration Example

```json
{
  "accounts": [
    {
      "prop_firm": "TFT",
      "account_size": 100000.0,
      "challenge_phase": "phase1",
      "risk_per_trade": 1.0,
      "max_positions": 3,
      "strategies": ["AsymmetricRiskProfile"],
      "primary_strategy": "AsymmetricRiskProfile",
      "strategy_params": {
        "tp_sl_ratio_min": 2.0,
        "tp_sl_ratio_max": 4.0
      },
      "weight": 0.4
    },
    {
      "prop_firm": "FTMO",
      "account_size": 200000.0,
      "challenge_phase": "phase1",
      "risk_per_trade": 0.8,
      "max_positions": 2,
      "strategies": ["RegimeSwitching"],
      "primary_strategy": "RegimeSwitching",
      "strategy_params": {
        "volatility_threshold": 20.0,
        "trend_strength_threshold": 0.5
      },
      "weight": 0.6
    }
  ]
}
```

## Installation Requirements

Additional Python packages required for Sprint 5 features:

```
hyperopt>=0.2.7
scikit-optimize>=0.9.0
hmmlearn>=0.2.8
tensorflow>=2.8.0  # Optional, for deep learning-based regime detection
```

## Usage Examples

### Monte Carlo Simulation

```bash
# Run 1000 simulations with parameter randomization
./run_monte_carlo_simulation.py --strategy asymmetric --num-simulations 1000 --randomize-params

# Run simulations with execution randomization and specific date range
./run_monte_carlo_simulation.py --strategy regime --num-simulations 500 --randomize-execution --start-time 2023-01-01 --end-time 2023-06-30
```

### Bayesian Optimization

```bash
# Optimize Sharpe ratio using TPE method
./run_bayesian_optimization.py --strategy asymmetric --objective-metric sharpe_ratio --method tpe --max-evals 100

# Optimize total return using Gaussian Process with early stopping
./run_bayesian_optimization.py --strategy regime --objective-metric total_return --method gp --max-evals 50 --early-stopping --patience 10
```

### Multi-Account Simulation

```bash
# Run multi-account simulation with configuration file
./run_multi_account_simulation.py --config-file configs/multi_account_config.json

# Run with specific date range
./run_multi_account_simulation.py --config-file configs/multi_account_config.json --start-time 2023-01-01 --end-time 2023-12-31
```

### Market Regime Detection

```bash
# Detect 3 market regimes using K-means
./run_market_regime_detection.py --market-data-dir data/market_data --n-regimes 3 --method kmeans

# Detect regimes and run adaptive backtest
./run_market_regime_detection.py --market-data-dir data/market_data --n-regimes 3 --method hmm --run-adaptive-backtest
```

## Future Enhancements

1. **Monte Carlo Simulation**:
   - Add walk-forward Monte Carlo for more realistic simulation
   - Implement extreme event simulation (black swan events)

2. **Bayesian Optimization**:
   - Add multi-objective optimization capabilities
   - Implement distributed optimization for faster processing

3. **Multi-Account Simulation**:
   - Add dynamic portfolio rebalancing
   - Implement hierarchical risk management

4. **Market Regime Detection**:
   - Add deep learning-based regime detection
   - Implement online/incremental regime detection for real-time applications

## Contributors

- Sprint 5 implementation by the OpenHands team