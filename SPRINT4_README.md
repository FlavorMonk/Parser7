# Forex Signal Trading System - Sprint 4 Advanced Backtesting Engine

This document outlines the advanced backtesting engine implemented in Sprint 4 of the Forex Signal Trading System.

## Overview

Sprint 4 focused on building a robust, modular, and realistic backtesting engine specifically designed for prop firm challenge simulation. The engine includes:

1. **Prop Firm Rules Engine**: Enforces all prop firm rules (daily loss limits, max drawdown, etc.)
2. **Execution Simulator**: Models realistic slippage, spread, partial fills, and requotes
3. **Backtest Engine**: Provides comprehensive backtesting capabilities with multiple strategies
4. **Rolling Window Simulation**: Tests strategies across multiple time windows for robust performance evaluation

## Components

### 1. Prop Firm Rules Engine (`backtesting/rules_engine.py`)

A flexible rules engine that enforces prop firm-specific rules:

- Daily loss limits
- Maximum drawdown
- Minimum trading days
- Profit targets
- Comprehensive rule violation tracking
- Support for different prop firms (TFT, FTMO, MFF)
- Support for different account types and challenge phases

### 2. Execution Simulator (`backtesting/execution_simulator.py`)

Simulates realistic trade execution with:

- Volatility-based slippage modeling
- Dynamic spread calculation
- Partial fills for large orders
- Requotes during volatile conditions
- Realistic latency simulation
- Support for different order types (market, limit, stop)

### 3. Backtest Engine (`backtesting/backtest_engine.py`)

Core backtesting engine with:

- Multi-strategy support
- Rolling window backtesting
- Comprehensive performance metrics
- Detailed trade logging
- Visualization of results
- Parameter optimization capabilities

### 4. Advanced Backtest Runner (`run_advanced_backtest.py`)

Script to run different types of backtests:

- Single strategy backtests
- Multi-strategy backtests
- Rolling window backtests
- Parameter optimization

## Key Features

### True-to-life Simulation

- Models all prop firm rules and real-world execution constraints
- Includes slippage, spread, partial fills, and requotes
- Enforces daily loss limits and maximum drawdown intraday

### Rolling-Window Simulation

- Tests strategies across multiple time windows
- Provides robust pass rate estimates across market conditions
- Identifies failure reasons and performance patterns

### Realistic Execution Modeling

- Volatility-based slippage
- Dynamic spreads based on market conditions
- Partial fills for large orders
- Requotes during volatile conditions

### Comprehensive Analytics

- Detailed performance metrics (Sharpe, Sortino, win rate, etc.)
- Trade-by-trade analysis
- Visualization of equity curves and drawdowns
- Failure mode analysis

## Usage

### Basic Backtest

```bash
python run_advanced_backtest.py single --strategy regime --prop-firm TFT --account-size 100000
```

### Rolling Window Backtest

```bash
python run_advanced_backtest.py rolling --strategy regime --window-size 30 --step-size 5 --max-windows 10
```

### Parameter Optimization

```bash
python run_advanced_backtest.py optimize --strategy regime
```

### Multi-Strategy Backtest

```bash
python run_advanced_backtest.py single --strategy all
```

## Configuration Options

- **Prop Firm**: TFT, FTMO, MFF
- **Challenge Phase**: phase1, phase2, verification, funded
- **Strategy**: asymmetric, regime, all
- **Risk Per Trade**: 0.01 (1% of account)
- **Max Positions**: Maximum concurrent positions
- **Window Size**: Size of rolling windows in days
- **Step Size**: Step between windows in days

## Performance Metrics

- **Pass Rate**: Percentage of windows that pass the challenge
- **Total Return**: Overall return on investment
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit divided by gross loss
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return

## Next Steps

- **Monte Carlo Simulation**: Add synthetic scenario generation
- **Agentic Optimization**: Integrate Bayesian parameter optimization
- **Multi-Account Simulation**: Simulate multiple accounts and cross-strategy risk
- **Machine Learning Integration**: Add ML-based regime detection and strategy selection