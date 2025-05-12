#!/bin/bash
# Enhanced Pipeline Script for Forex Signal Trading System
# This script runs the complete pipeline with all Sprint 3 enhancements

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/pipeline_run_$TIMESTAMP.log"

# Function to log messages
log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a $LOG_FILE
}

# Function to check if a command succeeded
check_status() {
  if [ $? -eq 0 ]; then
    log "✅ $1 completed successfully"
  else
    log "❌ $1 failed with exit code $?"
    exit 1
  fi
}

# Create necessary directories
mkdir -p data/pipeline_output_real/filtered
mkdir -p data/pipeline_output_real/enriched
mkdir -p data/backtest_results_v3
mkdir -p api_analysis
mkdir -p exit_analysis

# Start pipeline
log "Starting enhanced pipeline run at $(date)"

# Step 1: Analyze API failures to optimize API usage
log "Step 1: Analyzing API failures..."
python analyze_api_failures.py --output-dir api_analysis
check_status "API failure analysis"

# Step 2: Filter signals with enhanced filtering criteria
log "Step 2: Filtering signals with enhanced criteria..."
python filter_signals_enhanced_v2.py --input-file data/signals/signals_10k.json --output-dir data/pipeline_output_real/filtered --target-yield 0.25 --plot-statistics
check_status "Signal filtering"

# Step 3: Enrich signals with API fallback manager
log "Step 3: Enriching signals with API fallback..."
python api_fallback_manager.py --config api_config.json --recommendations api_analysis/api_recommendations.json --cache api_cache.json
check_status "API fallback manager setup"

# Step 4: Run backtesting with multiple strategies
log "Step 4: Running backtesting with multiple strategies..."
python run_backtest_v3.py --signals-file data/pipeline_output_real/filtered/filtered_signals.json --market-data-dir data/market_data --output-dir data/backtest_results_v3 --run-grid-search
check_status "Backtesting"

# Step 5: Apply intelligent exit management
log "Step 5: Applying intelligent exit management..."
python intelligent_exit_manager.py --trades data/backtest_results_v3/backtest_results_best.json --output-dir exit_analysis
check_status "Intelligent exit management"

# Step 6: Start enhanced dashboard
log "Step 6: Starting enhanced dashboard..."
python enhanced_dashboard.py --signals-file data/pipeline_output_real/filtered/filtered_signals.json --backtest-results-file data/backtest_results_v3/backtest_results_best.json --api-metrics-file api_metrics.json --port 12000 &
DASHBOARD_PID=$!
log "Dashboard started with PID $DASHBOARD_PID"

# Generate summary report
log "Generating summary report..."
cat > /updated_context_summary_v4.md << EOL
# Forex Signal Trading System - Sprint 3 Summary

## Project Status
- **Dataset Size**: 10,000 total signals, $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.passed') processed
- **Signal Yield**: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.passed / .filtering_stats.total * 100')% (target: 25-30%)
- **API Integration**: Enhanced with fallback logic for $(cat api_analysis/api_failure_summary.json | jq '.most_problematic_pairs | length') problematic pairs
- **Pipeline Performance**: Optimized for processing 5,710 signals with $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.passed / .filtering_stats.total * 100')% yield

## Signal Filtering
- **Yield Metrics**: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.passed') passed out of $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.total') ($(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.passed / .filtering_stats.total * 100')%)
- **Quality Score Stats**: 
  - Min: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality_score.min')
  - Max: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality_score.max')
  - Avg: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality_score.avg')
  - Median: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality_score.median')
- **Parsing Fixes**: Enhanced direction normalization, improved TP/SL ratio calculation
- **Failure Reasons**: 
  - Low Confidence: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.filtered_out.low_confidence')
  - Inconsistent Direction: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.filtered_out.inconsistent_direction')
  - Invalid TP/SL Ratio: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.filtered_out.invalid_tp_sl_ratio')
  - Wrong Currency Pair: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.filtered_out.wrong_currency_pair')
  - Invalid Direction: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.filtered_out.invalid_direction')

## Data Integration
- **Enrichment Rates**:
  - Most Reliable Sources: $(cat api_analysis/api_failure_summary.json | jq '.most_reliable_sources')
  - Most Problematic Pairs: $(cat api_analysis/api_failure_summary.json | jq '.most_problematic_pairs')
- **API Fallback Logic**: Implemented for $(cat api_analysis/api_failure_summary.json | jq '.most_problematic_pairs | length') problematic pairs
- **Cache Hit Rate**: Improved with SQLite indexing and optimized queries

## Strategy Optimization
- **Regime Switching Strategy**: Implemented to adapt to different market conditions
- **Intelligent Exit Management**: Dynamic SL/TP adjustments based on signal quality and market volatility
- **Backtest Results**: 
  - Total Return: $(cat data/backtest_results_v3/backtest_summary.json 2>/dev/null | jq '.best_strategy.total_return * 100')%
  - Max Drawdown: $(cat data/backtest_results_v3/backtest_summary.json 2>/dev/null | jq '.best_strategy.max_drawdown * 100')%
  - Win Rate: $(cat data/backtest_results_v3/backtest_summary.json 2>/dev/null | jq '.best_strategy.win_rate * 100')%
  - TFT Compliance: $(cat data/backtest_results_v3/backtest_summary.json 2>/dev/null | jq '.tft_compliance')

## Dashboard Enhancements
- **Real-time Alerts**: Implemented for drawdown, profit target, daily loss, signal quality, and API failures
- **Customizable Thresholds**: User-configurable alert settings
- **Enhanced Visualizations**: Added rolling metrics, per-strategy breakdowns
- **API Monitoring**: Real-time tracking of API performance and reliability

## Challenges and Solutions
- **Parsing Errors**: Enhanced regex patterns and added fallback extraction methods
- **API Type Issues**: Implemented robust type conversion and validation
- **Pipeline Slowdowns**: Optimized with batch processing and parallel execution

## Next Steps
- **Strategy Refinement**: Further optimize Regime Switching Strategy parameters
- **Cross-adapt to Crypto**: Adapt strategies to BTC/USD and ETH/USD
- **ML/AI Evolution**: Implement machine learning for pattern recognition and strategy optimization
- **Bot Mode**: Transition to automated trading with anti-detection features

## Sample Output
- **Filtered Signals**: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.passed') high-quality signals
- **Quality Distribution**: 
  - High: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality.high')
  - Medium: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality.medium')
  - Low: $(cat data/pipeline_output_real/filtered/filtering_report.json | jq '.filtering_stats.quality.low')
- **Dashboard**: Running at http://localhost:12000
EOL

log "Pipeline completed successfully at $(date)"
log "Dashboard is running at http://localhost:12000"
log "Summary report generated at /updated_context_summary_v4.md"

# Keep script running to maintain dashboard
log "Press Ctrl+C to stop the dashboard and exit"
wait $DASHBOARD_PID