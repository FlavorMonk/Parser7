#!/bin/bash
# Run the full pipeline for the Forex Signal Trading System

# Set up environment
echo "Setting up environment..."
mkdir -p data/pipeline_output_real/filtered
mkdir -p data/pipeline_output_real/enriched
mkdir -p data/market_data
mkdir -p data/backtest_results

# Step 1: Filter signals
echo "Step 1: Filtering signals..."
python filter_signals_enhanced.py \
  --input data/signals/signals_10k.json \
  --output data/pipeline_output_real/filtered/filtered_signals.json

# Step 2: Enrich signals
echo "Step 2: Enriching signals..."
python enhanced_multi_source_enricher.py \
  --input data/pipeline_output_real/filtered/filtered_signals.json \
  --output data/pipeline_output_real/enriched/enriched_signals.json

# Step 3: Generate market data for backtesting
echo "Step 3: Generating market data for backtesting..."
python generate_sample_market_data.py

# Step 4: Run backtest
echo "Step 4: Running backtest..."
python run_backtest_v2.py \
  --signals-file data/pipeline_output_real/enriched/enriched_signals.json \
  --market-data-dir data/market_data \
  --output-dir data/backtest_results

# Step 5: Generate performance report
echo "Step 5: Generating performance report..."
python generate_performance_report.py \
  --backtest-results data/backtest_results/backtest_results.json \
  --output data/backtest_results/performance_report.html

# Step 6: Start advisor dashboard
echo "Step 6: Starting advisor dashboard..."
echo "Dashboard will be available at http://localhost:12000"
python advisor_dashboard.py \
  --signals-file data/pipeline_output_real/filtered/filtered_signals.json \
  --backtest-results data/backtest_results/backtest_results.json \
  --port 12000

echo "Pipeline completed successfully!"