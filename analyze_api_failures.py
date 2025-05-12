#!/usr/bin/env python3
"""
Analyze API Failures

This script analyzes the API failures in the enrichment process and identifies
problematic currency pairs and alternative data sources.
"""

import argparse
import json
import os
import pandas as pd
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_log_file(log_file):
    """Load and parse the log file."""
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    return log_content

def extract_api_failures(log_content):
    """Extract API failures from the log content."""
    # Extract "No data found" warnings
    no_data_pattern = r"WARNING - enhanced_multi_source_enricher - No data found for ([A-Z/]+) from ([A-Za-z]+)"
    no_data_matches = re.findall(no_data_pattern, log_content)
    
    # Extract "Failed to get data" warnings
    failed_pattern = r"WARNING - enhanced_multi_source_enricher - Failed to get data for ([A-Z/]+) from any source"
    failed_matches = re.findall(failed_pattern, log_content)
    
    # Extract "No market data found" warnings
    no_market_pattern = r"WARNING - enhanced_multi_source_enricher - No market data found for ([A-Z/]+) at (\d+\.\d+)"
    no_market_matches = re.findall(no_market_pattern, log_content)
    
    return no_data_matches, failed_matches, no_market_matches

def analyze_failures(no_data_matches, failed_matches, no_market_matches):
    """Analyze the failures and generate statistics."""
    # Count failures by currency pair and source
    pair_source_failures = Counter(no_data_matches)
    pair_failures = Counter([pair for pair, _ in no_data_matches])
    source_failures = Counter([source for _, source in no_data_matches])
    
    # Count complete failures by currency pair
    complete_failures = Counter(failed_matches)
    
    # Count market data failures by currency pair
    market_failures = Counter([pair for pair, _ in no_market_matches])
    
    # Calculate success rate by source for each pair
    pair_source_success = defaultdict(dict)
    for pair in pair_failures:
        for source in ['Yahoo Finance', 'TraderMade', 'Finage', 'Alpha Vantage', 'ForexRate', 'TwelveData']:
            failures = pair_source_failures.get((pair, source), 0)
            total_attempts = sum(1 for p, s in no_data_matches if p == pair)
            success_rate = 1.0 - (failures / total_attempts if total_attempts > 0 else 0)
            pair_source_success[pair][source] = success_rate
    
    return {
        'pair_source_failures': pair_source_failures,
        'pair_failures': pair_failures,
        'source_failures': source_failures,
        'complete_failures': complete_failures,
        'market_failures': market_failures,
        'pair_source_success': pair_source_success
    }

def generate_recommendations(analysis):
    """Generate recommendations for alternative data sources."""
    recommendations = {}
    
    for pair, sources in analysis['pair_source_success'].items():
        # Sort sources by success rate
        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        
        # Get the best sources (success rate > 0)
        best_sources = [source for source, rate in sorted_sources if rate > 0]
        
        if best_sources:
            recommendations[pair] = {
                'recommended_sources': best_sources,
                'success_rates': {source: sources[source] for source in best_sources}
            }
        else:
            recommendations[pair] = {
                'recommended_sources': [],
                'success_rates': {},
                'alternative_suggestion': 'Consider using a paid API service or implementing a custom data source'
            }
    
    return recommendations

def plot_failure_statistics(analysis, output_dir):
    """Plot failure statistics and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot pair failures
    plt.figure(figsize=(12, 8))
    pair_failures = pd.Series(analysis['pair_failures']).sort_values(ascending=False)
    sns.barplot(x=pair_failures.index, y=pair_failures.values)
    plt.title('API Failures by Currency Pair')
    plt.xlabel('Currency Pair')
    plt.ylabel('Number of Failures')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pair_failures.png'))
    
    # Plot source failures
    plt.figure(figsize=(12, 8))
    source_failures = pd.Series(analysis['source_failures']).sort_values(ascending=False)
    sns.barplot(x=source_failures.index, y=source_failures.values)
    plt.title('API Failures by Data Source')
    plt.xlabel('Data Source')
    plt.ylabel('Number of Failures')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'source_failures.png'))
    
    # Plot complete failures
    plt.figure(figsize=(12, 8))
    complete_failures = pd.Series(analysis['complete_failures']).sort_values(ascending=False)
    sns.barplot(x=complete_failures.index, y=complete_failures.values)
    plt.title('Complete API Failures by Currency Pair')
    plt.xlabel('Currency Pair')
    plt.ylabel('Number of Complete Failures')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complete_failures.png'))
    
    # Plot success rates heatmap
    plt.figure(figsize=(14, 10))
    success_df = pd.DataFrame(analysis['pair_source_success']).T.fillna(0)
    sns.heatmap(success_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('API Success Rates by Currency Pair and Source')
    plt.xlabel('Data Source')
    plt.ylabel('Currency Pair')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rates.png'))

def save_results(analysis, recommendations, output_dir):
    """Save analysis results and recommendations to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save analysis results
    with open(os.path.join(output_dir, 'api_failure_analysis.json'), 'w') as f:
        # Convert Counter objects to dictionaries
        serializable_analysis = {
            'pair_source_failures': analysis['pair_source_failures'],
            'pair_failures': dict(analysis['pair_failures']) if isinstance(analysis['pair_failures'], Counter) else analysis['pair_failures'],
            'source_failures': dict(analysis['source_failures']) if isinstance(analysis['source_failures'], Counter) else analysis['source_failures'],
            'complete_failures': dict(analysis['complete_failures']) if isinstance(analysis['complete_failures'], Counter) else analysis['complete_failures'],
            'market_failures': dict(analysis['market_failures']) if isinstance(analysis['market_failures'], Counter) else analysis['market_failures'],
            'pair_source_success': analysis['pair_source_success']
        }
        json.dump(serializable_analysis, f, indent=2)
    
    # Save recommendations
    with open(os.path.join(output_dir, 'api_recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    # Generate summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_pairs_analyzed': len(analysis['pair_failures']),
        'total_sources_analyzed': len(analysis['source_failures']),
        'most_problematic_pairs': [pair for pair, _ in Counter(analysis['pair_failures']).most_common(5)],
        'most_reliable_sources': [source for source, _ in Counter({source: 1 - failures / sum(analysis['source_failures'].values()) 
                                                                for source, failures in analysis['source_failures'].items()}).most_common(3)],
        'pairs_with_no_data': [pair for pair, count in analysis['complete_failures'].items() if count > 0],
        'recommended_alternative_apis': [
            'Oanda API (requires account)',
            'FXCM API (requires account)',
            'Polygon.io (paid, but reliable)',
            'IEX Cloud (paid, but reliable)',
            'Alpha Vantage Premium (paid upgrade)'
        ]
    }
    
    with open(os.path.join(output_dir, 'api_failure_summary.json'), 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Analyze API failures in the enrichment process')
    parser.add_argument('--log-file', type=str, default='logs/enrichment.log',
                        help='Path to the log file containing API failure information')
    parser.add_argument('--output-dir', type=str, default='api_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample data if log file is not available')
    
    args = parser.parse_args()
    
    if args.generate_sample or not os.path.exists(args.log_file):
        print(f"Log file {args.log_file} not found. Generating sample data.")
        # Generate sample data
        sample_data = {
            'pair_source_failures': {
                'XAUUSD_Yahoo Finance': 15, 'XAUUSD_TraderMade': 5, 'XAUUSD_Finage': 8,
                'USDCAD_Yahoo Finance': 12, 'USDCAD_TraderMade': 3, 'USDCAD_Finage': 6,
                'GBPJPY_Yahoo Finance': 10, 'GBPJPY_TraderMade': 2, 'GBPJPY_Finage': 4,
                'GBPCAD_Yahoo Finance': 9, 'GBPCAD_TraderMade': 1, 'GBPCAD_Finage': 3,
                'GOLD_Yahoo Finance': 8, 'GOLD_TraderMade': 4, 'GOLD_Finage': 2
            },
            'pair_failures': {
                'XAUUSD': 28, 'USDCAD': 21, 'GBPJPY': 16, 'GBPCAD': 13, 'GOLD': 14
            },
            'source_failures': {
                'Yahoo Finance': 54, 'TraderMade': 15, 'Finage': 23, 'Alpha Vantage': 12, 'ForexRate': 18, 'TwelveData': 8
            },
            'complete_failures': {
                'XAUUSD': 5, 'USDCAD': 3, 'GBPJPY': 2, 'GBPCAD': 1, 'GOLD': 4
            },
            'market_failures': {
                'XAUUSD': 10, 'USDCAD': 8, 'GBPJPY': 6, 'GBPCAD': 5, 'GOLD': 9
            },
            'pair_source_success': {
                'XAUUSD': {'Yahoo Finance': 0.2, 'TraderMade': 0.8, 'Finage': 0.6, 'Alpha Vantage': 0.7, 'ForexRate': 0.5, 'TwelveData': 0.9},
                'USDCAD': {'Yahoo Finance': 0.3, 'TraderMade': 0.9, 'Finage': 0.7, 'Alpha Vantage': 0.6, 'ForexRate': 0.4, 'TwelveData': 0.8},
                'GBPJPY': {'Yahoo Finance': 0.4, 'TraderMade': 0.9, 'Finage': 0.8, 'Alpha Vantage': 0.5, 'ForexRate': 0.3, 'TwelveData': 0.7},
                'GBPCAD': {'Yahoo Finance': 0.5, 'TraderMade': 0.95, 'Finage': 0.85, 'Alpha Vantage': 0.4, 'ForexRate': 0.2, 'TwelveData': 0.6},
                'GOLD': {'Yahoo Finance': 0.6, 'TraderMade': 0.85, 'Finage': 0.9, 'Alpha Vantage': 0.3, 'ForexRate': 0.1, 'TwelveData': 0.5}
            }
        }
        analysis = sample_data
    else:
        # Load log file
        log_content = load_log_file(args.log_file)
        
        # Extract API failures
        no_data_matches, failed_matches, no_market_matches = extract_api_failures(log_content)
        
        # Analyze failures
        analysis = analyze_failures(no_data_matches, failed_matches, no_market_matches)
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis)
    
    # Plot failure statistics
    plot_failure_statistics(analysis, args.output_dir)
    
    # Save results
    save_results(analysis, recommendations, args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()