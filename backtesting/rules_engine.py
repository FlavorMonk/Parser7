#!/usr/bin/env python3
"""
Prop Firm Rules Engine

This module implements a flexible rules engine for prop firm challenge simulations.
It enforces rules such as daily loss limits, maximum drawdown, minimum trading days,
and other prop firm-specific requirements.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger("prop_firm_rules_engine")

class PropFirmRulesEngine:
    """
    Rules engine for prop firm challenge simulations.
    
    Features:
    - Modular rule enforcement for different prop firms
    - Intraday rule checking
    - Comprehensive logging of rule violations
    - Support for multiple account types and challenge phases
    """
    
    def __init__(self, 
                 firm_type: str = "TFT", 
                 account_type: str = "standard",
                 account_size: float = 100000.0,
                 challenge_phase: str = "phase1",
                 custom_rules: Optional[Dict] = None):
        """
        Initialize the rules engine with specific prop firm rules.
        
        Args:
            firm_type: Prop firm type (e.g., "TFT", "FTMO", "MFF")
            account_type: Account type (e.g., "standard", "aggressive", "conservative")
            account_size: Account size in base currency
            challenge_phase: Challenge phase (e.g., "phase1", "phase2", "verification")
            custom_rules: Custom rules to override defaults
        """
        self.firm_type = firm_type
        self.account_type = account_type
        self.account_size = account_size
        self.challenge_phase = challenge_phase
        
        # Initialize default rules
        self.rules = self._get_default_rules()
        
        # Override with custom rules if provided
        if custom_rules:
            self._update_rules(custom_rules)
        
        # Initialize rule violation tracking
        self.violations = []
        self.warnings = []
        
        # Initialize state tracking
        self.state = {
            "current_balance": account_size,
            "highest_balance": account_size,
            "lowest_balance": account_size,
            "daily_starting_balance": account_size,
            "current_drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_loss_pct": 0.0,
            "trading_days": 0,
            "profitable_trading_days": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "current_day": None,
            "last_trade_time": None,
            "daily_trades": 0,
            "daily_profit_loss": 0.0,
            "total_profit_loss": 0.0,
            "profit_target_reached": False,
            "challenge_passed": False,
            "challenge_failed": False,
            "failure_reason": None,
            "trade_history": [],
            "daily_balance_history": [],
            "equity_curve": []
        }
        
        logger.info(f"Initialized {firm_type} rules engine for {account_type} account (${account_size}) in {challenge_phase}")
    
    def _get_default_rules(self) -> Dict:
        """
        Get default rules based on firm type, account type, and challenge phase.
        
        Returns:
            Dictionary of rules
        """
        # Base rules common to most prop firms
        base_rules = {
            "max_daily_loss_pct": 0.05,  # 5% daily loss limit
            "max_total_loss_pct": 0.10,  # 10% total loss limit (max drawdown)
            "profit_target_pct": 0.08,   # 8% profit target
            "min_trading_days": 10,      # Minimum trading days
            "max_challenge_days": 30,    # Maximum days to complete challenge
            "weekend_trading": False,    # Weekend trading allowed
            "news_trading": True,        # News trading allowed
            "overnight_positions": True, # Overnight positions allowed
            "max_positions": 5,          # Maximum simultaneous positions
            "max_daily_trades": 0,       # Maximum daily trades (0 = unlimited)
            "min_duration_minutes": 0,   # Minimum trade duration in minutes
            "scaling_enabled": False,    # Scaling rules enabled
            "consistency_required": False, # Consistency rules (e.g., trading minimum days)
            "max_leverage": 30.0,        # Maximum leverage
            "max_position_size_pct": 0.05, # Maximum position size as % of account
            "restricted_instruments": [], # Restricted instruments
            "required_instruments": [],  # Required instruments to trade
            "inactivity_limit_days": 5,  # Maximum consecutive days without trading
            "min_win_rate": 0.0,         # Minimum win rate required
            "min_profit_days_pct": 0.0,  # Minimum percentage of profitable days
            "max_daily_drawdown_pct": 0.0, # Maximum intraday drawdown
            "scaling_profit_pct": 0.0,   # Profit percentage for scaling
            "scaling_factor": 0.0,       # Scaling factor for account growth
            "time_restrictions": {}      # Time restrictions for trading
        }
        
        # Firm-specific rule overrides
        if self.firm_type == "TFT":  # The Funded Trader
            tft_rules = {
                "max_daily_loss_pct": 0.02,  # 2% daily loss limit
                "max_total_loss_pct": 0.04,  # 4% total loss limit
                "profit_target_pct": 0.08,   # 8% profit target
                "min_trading_days": 5,       # Minimum trading days
                "max_challenge_days": 30,    # 30-day challenge
                "consistency_required": True, # Must trade minimum days
                "min_profit_days_pct": 0.0,  # No minimum profitable days
                "max_leverage": 30.0,        # 1:30 leverage
                "scaling_enabled": True,     # Scaling program available
                "scaling_profit_pct": 0.05,  # 5% for scaling
                "scaling_factor": 0.25       # 25% account growth per scaling
            }
            base_rules.update(tft_rules)
            
        elif self.firm_type == "FTMO":  # FTMO
            ftmo_rules = {
                "max_daily_loss_pct": 0.05,  # 5% daily loss limit
                "max_total_loss_pct": 0.10,  # 10% total loss limit
                "profit_target_pct": 0.10,   # 10% profit target
                "min_trading_days": 10,      # Minimum trading days
                "max_challenge_days": 30,    # 30-day challenge
                "consistency_required": True, # Must trade minimum days
                "min_profit_days_pct": 0.0,  # No minimum profitable days
                "max_leverage": 100.0,       # 1:100 leverage
                "max_daily_drawdown_pct": 0.05, # 5% max daily drawdown
                "overnight_positions": True,  # Overnight positions allowed
                "weekend_trading": False      # No weekend trading
            }
            base_rules.update(ftmo_rules)
            
        elif self.firm_type == "MFF":  # My Forex Funds
            mff_rules = {
                "max_daily_loss_pct": 0.04,  # 4% daily loss limit
                "max_total_loss_pct": 0.08,  # 8% total loss limit
                "profit_target_pct": 0.10,   # 10% profit target (Rapid)
                "min_trading_days": 5,       # Minimum trading days
                "max_challenge_days": 30,    # 30-day challenge
                "consistency_required": True, # Must trade minimum days
                "min_profit_days_pct": 0.0,  # No minimum profitable days
                "max_leverage": 100.0,       # 1:100 leverage
                "overnight_positions": True,  # Overnight positions allowed
                "weekend_trading": False      # No weekend trading
            }
            base_rules.update(mff_rules)
        
        # Account type specific adjustments
        if self.account_type == "aggressive":
            base_rules["max_daily_loss_pct"] *= 1.2  # 20% higher daily loss limit
            base_rules["max_total_loss_pct"] *= 1.2  # 20% higher total loss limit
            base_rules["profit_target_pct"] *= 1.5   # 50% higher profit target
            
        elif self.account_type == "conservative":
            base_rules["max_daily_loss_pct"] *= 0.8  # 20% lower daily loss limit
            base_rules["max_total_loss_pct"] *= 0.8  # 20% lower total loss limit
            base_rules["profit_target_pct"] *= 0.7   # 30% lower profit target
        
        # Challenge phase specific adjustments
        if self.challenge_phase == "phase2" or self.challenge_phase == "verification":
            # Phase 2 typically has same rules but longer duration
            base_rules["max_challenge_days"] = 60  # Longer duration for phase 2
            
        elif self.challenge_phase == "funded":
            # Funded accounts typically have relaxed rules
            base_rules["min_trading_days"] = 0  # No minimum trading days
            base_rules["profit_target_pct"] = 0.0  # No profit target
            base_rules["max_challenge_days"] = 0  # No time limit
        
        return base_rules
    
    def _update_rules(self, custom_rules: Dict) -> None:
        """
        Update rules with custom overrides.
        
        Args:
            custom_rules: Dictionary of custom rules to override defaults
        """
        self.rules.update(custom_rules)
        logger.info(f"Updated rules with custom overrides: {custom_rules}")
    
    def reset(self) -> None:
        """Reset the rules engine state for a new simulation."""
        initial_balance = self.account_size
        self.state = {
            "current_balance": initial_balance,
            "highest_balance": initial_balance,
            "lowest_balance": initial_balance,
            "daily_starting_balance": initial_balance,
            "current_drawdown_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_loss_pct": 0.0,
            "trading_days": 0,
            "profitable_trading_days": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "current_day": None,
            "last_trade_time": None,
            "daily_trades": 0,
            "daily_profit_loss": 0.0,
            "total_profit_loss": 0.0,
            "profit_target_reached": False,
            "challenge_passed": False,
            "challenge_failed": False,
            "failure_reason": None,
            "trade_history": [],
            "daily_balance_history": [],
            "equity_curve": []
        }
        self.violations = []
        self.warnings = []
        logger.info("Rules engine state reset")
    
    def update_account_state(self, 
                            current_balance: float, 
                            timestamp: datetime,
                            unrealized_pnl: float = 0.0) -> None:
        """
        Update account state with current balance and timestamp.
        
        Args:
            current_balance: Current account balance
            timestamp: Current timestamp
            unrealized_pnl: Unrealized profit/loss from open positions
        """
        # Calculate equity (balance + unrealized P&L)
        current_equity = current_balance + unrealized_pnl
        
        # Check if this is a new day
        current_day = timestamp.date()
        if self.state["current_day"] is None or current_day != self.state["current_day"]:
            # It's a new day
            if self.state["current_day"] is not None:
                # Record previous day's results
                daily_result = {
                    "date": self.state["current_day"].isoformat(),
                    "starting_balance": self.state["daily_starting_balance"],
                    "ending_balance": self.state["current_balance"],
                    "profit_loss": self.state["daily_profit_loss"],
                    "profit_loss_pct": self.state["daily_profit_loss"] / self.state["daily_starting_balance"] if self.state["daily_starting_balance"] > 0 else 0,
                    "trades": self.state["daily_trades"],
                    "profitable": self.state["daily_profit_loss"] > 0
                }
                self.state["daily_balance_history"].append(daily_result)
                
                # Update trading days count
                self.state["trading_days"] += 1
                
                # Update profitable days count
                if self.state["daily_profit_loss"] > 0:
                    self.state["profitable_trading_days"] += 1
            
            # Reset daily tracking
            self.state["current_day"] = current_day
            self.state["daily_starting_balance"] = current_balance
            self.state["daily_profit_loss"] = 0.0
            self.state["daily_trades"] = 0
            self.state["daily_loss_pct"] = 0.0
            
            logger.info(f"New trading day: {current_day.isoformat()}")
        
        # Update balance and related metrics
        previous_balance = self.state["current_balance"]
        self.state["current_balance"] = current_balance
        
        # Update highest/lowest balance
        if current_balance > self.state["highest_balance"]:
            self.state["highest_balance"] = current_balance
        if current_balance < self.state["lowest_balance"]:
            self.state["lowest_balance"] = current_balance
        
        # Calculate drawdown
        if self.state["highest_balance"] > 0:
            self.state["current_drawdown_pct"] = (self.state["highest_balance"] - current_equity) / self.state["highest_balance"]
            if self.state["current_drawdown_pct"] > self.state["max_drawdown_pct"]:
                self.state["max_drawdown_pct"] = self.state["current_drawdown_pct"]
        
        # Calculate daily loss
        if self.state["daily_starting_balance"] > 0:
            daily_change = current_equity - self.state["daily_starting_balance"]
            self.state["daily_profit_loss"] = daily_change
            self.state["daily_loss_pct"] = abs(daily_change) / self.state["daily_starting_balance"] if daily_change < 0 else 0
        
        # Calculate total profit/loss
        self.state["total_profit_loss"] = current_balance - self.account_size
        
        # Check if profit target reached
        if not self.state["profit_target_reached"] and self.state["total_profit_loss"] >= self.account_size * self.rules["profit_target_pct"]:
            self.state["profit_target_reached"] = True
            logger.info(f"Profit target reached: ${self.state['total_profit_loss']:.2f} ({self.state['total_profit_loss'] / self.account_size * 100:.2f}%)")
        
        # Update equity curve
        self.state["equity_curve"].append({
            "timestamp": timestamp.isoformat(),
            "balance": current_balance,
            "equity": current_equity,
            "drawdown_pct": self.state["current_drawdown_pct"],
            "unrealized_pnl": unrealized_pnl
        })
        
        # Log significant changes
        if abs(current_balance - previous_balance) > 0.01 * self.account_size:
            logger.info(f"Significant balance change: ${previous_balance:.2f} -> ${current_balance:.2f} (${current_balance - previous_balance:.2f})")
    
    def record_trade(self, 
                    trade_id: str,
                    instrument: str,
                    direction: str,
                    entry_price: float,
                    exit_price: float,
                    position_size: float,
                    entry_time: datetime,
                    exit_time: datetime,
                    profit_loss: float,
                    profit_loss_pips: float = None,
                    trade_duration_minutes: float = None,
                    metadata: Dict = None) -> None:
        """
        Record a completed trade and update account state.
        
        Args:
            trade_id: Unique trade identifier
            instrument: Traded instrument (e.g., "EURUSD")
            direction: Trade direction ("BUY" or "SELL")
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size in units
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            profit_loss: Profit/loss in account currency
            profit_loss_pips: Profit/loss in pips (optional)
            trade_duration_minutes: Trade duration in minutes (optional)
            metadata: Additional trade metadata (optional)
        """
        # Calculate trade duration if not provided
        if trade_duration_minutes is None and entry_time and exit_time:
            trade_duration_minutes = (exit_time - entry_time).total_seconds() / 60
        
        # Create trade record
        trade = {
            "trade_id": trade_id,
            "instrument": instrument,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "entry_time": entry_time.isoformat() if entry_time else None,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss / self.state["current_balance"] if self.state["current_balance"] > 0 else 0,
            "profit_loss_pips": profit_loss_pips,
            "trade_duration_minutes": trade_duration_minutes,
            "metadata": metadata or {}
        }
        
        # Update trade counts
        self.state["total_trades"] += 1
        self.state["daily_trades"] += 1
        
        if profit_loss > 0:
            self.state["winning_trades"] += 1
        elif profit_loss < 0:
            self.state["losing_trades"] += 1
        
        # Update last trade time
        self.state["last_trade_time"] = exit_time
        
        # Add to trade history
        self.state["trade_history"].append(trade)
        
        # Update account state
        new_balance = self.state["current_balance"] + profit_loss
        self.update_account_state(new_balance, exit_time)
        
        logger.info(f"Recorded trade {trade_id}: {instrument} {direction} - P/L: ${profit_loss:.2f}")
    
    def check_rules(self, timestamp: datetime = None) -> Tuple[bool, Optional[str]]:
        """
        Check if any rules are violated.
        
        Args:
            timestamp: Current timestamp (default: None, uses current time)
            
        Returns:
            Tuple of (rules_violated, violation_reason)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check daily loss limit
        if self.state["daily_loss_pct"] >= self.rules["max_daily_loss_pct"]:
            violation = f"Daily loss limit exceeded: {self.state['daily_loss_pct'] * 100:.2f}% (limit: {self.rules['max_daily_loss_pct'] * 100:.2f}%)"
            self.violations.append({
                "timestamp": timestamp.isoformat(),
                "rule": "max_daily_loss_pct",
                "description": violation,
                "value": self.state["daily_loss_pct"],
                "limit": self.rules["max_daily_loss_pct"]
            })
            self.state["challenge_failed"] = True
            self.state["failure_reason"] = violation
            logger.warning(violation)
            return True, violation
        
        # Check total loss limit (max drawdown)
        if self.state["current_drawdown_pct"] >= self.rules["max_total_loss_pct"]:
            violation = f"Maximum drawdown exceeded: {self.state['current_drawdown_pct'] * 100:.2f}% (limit: {self.rules['max_total_loss_pct'] * 100:.2f}%)"
            self.violations.append({
                "timestamp": timestamp.isoformat(),
                "rule": "max_total_loss_pct",
                "description": violation,
                "value": self.state["current_drawdown_pct"],
                "limit": self.rules["max_total_loss_pct"]
            })
            self.state["challenge_failed"] = True
            self.state["failure_reason"] = violation
            logger.warning(violation)
            return True, violation
        
        # Check maximum daily trades
        if self.rules["max_daily_trades"] > 0 and self.state["daily_trades"] > self.rules["max_daily_trades"]:
            violation = f"Maximum daily trades exceeded: {self.state['daily_trades']} (limit: {self.rules['max_daily_trades']})"
            self.violations.append({
                "timestamp": timestamp.isoformat(),
                "rule": "max_daily_trades",
                "description": violation,
                "value": self.state["daily_trades"],
                "limit": self.rules["max_daily_trades"]
            })
            self.state["challenge_failed"] = True
            self.state["failure_reason"] = violation
            logger.warning(violation)
            return True, violation
        
        # Check challenge duration
        if self.rules["max_challenge_days"] > 0:
            challenge_duration = (timestamp.date() - self.state["equity_curve"][0]["timestamp"].split("T")[0]) if self.state["equity_curve"] else timedelta(days=0)
            if challenge_duration.days > self.rules["max_challenge_days"]:
                violation = f"Challenge duration exceeded: {challenge_duration.days} days (limit: {self.rules['max_challenge_days']} days)"
                self.violations.append({
                    "timestamp": timestamp.isoformat(),
                    "rule": "max_challenge_days",
                    "description": violation,
                    "value": challenge_duration.days,
                    "limit": self.rules["max_challenge_days"]
                })
                self.state["challenge_failed"] = True
                self.state["failure_reason"] = violation
                logger.warning(violation)
                return True, violation
        
        # Check if profit target reached and minimum trading days met
        if self.state["profit_target_reached"] and self.state["trading_days"] >= self.rules["min_trading_days"]:
            self.state["challenge_passed"] = True
            logger.info(f"Challenge passed: Profit target reached ({self.state['total_profit_loss'] / self.account_size * 100:.2f}%) and minimum trading days met ({self.state['trading_days']} days)")
            return False, None
        
        # No rules violated
        return False, None
    
    def check_warnings(self, timestamp: datetime = None) -> List[Dict]:
        """
        Check for warning conditions that don't violate rules but are concerning.
        
        Args:
            timestamp: Current timestamp (default: None, uses current time)
            
        Returns:
            List of warning dictionaries
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        new_warnings = []
        
        # Check approaching daily loss limit
        daily_loss_threshold = self.rules["max_daily_loss_pct"] * 0.8  # 80% of limit
        if self.state["daily_loss_pct"] >= daily_loss_threshold:
            warning = f"Approaching daily loss limit: {self.state['daily_loss_pct'] * 100:.2f}% (limit: {self.rules['max_daily_loss_pct'] * 100:.2f}%)"
            new_warnings.append({
                "timestamp": timestamp.isoformat(),
                "rule": "max_daily_loss_pct",
                "description": warning,
                "value": self.state["daily_loss_pct"],
                "threshold": daily_loss_threshold,
                "limit": self.rules["max_daily_loss_pct"]
            })
            logger.warning(warning)
        
        # Check approaching total loss limit
        total_loss_threshold = self.rules["max_total_loss_pct"] * 0.8  # 80% of limit
        if self.state["current_drawdown_pct"] >= total_loss_threshold:
            warning = f"Approaching maximum drawdown: {self.state['current_drawdown_pct'] * 100:.2f}% (limit: {self.rules['max_total_loss_pct'] * 100:.2f}%)"
            new_warnings.append({
                "timestamp": timestamp.isoformat(),
                "rule": "max_total_loss_pct",
                "description": warning,
                "value": self.state["current_drawdown_pct"],
                "threshold": total_loss_threshold,
                "limit": self.rules["max_total_loss_pct"]
            })
            logger.warning(warning)
        
        # Check minimum trading days as challenge end approaches
        if self.rules["max_challenge_days"] > 0 and self.state["equity_curve"]:
            challenge_start = datetime.fromisoformat(self.state["equity_curve"][0]["timestamp"].split("+")[0])
            challenge_duration = (timestamp - challenge_start).days
            days_remaining = self.rules["max_challenge_days"] - challenge_duration
            
            if days_remaining <= 5 and self.state["trading_days"] < self.rules["min_trading_days"]:
                days_needed = self.rules["min_trading_days"] - self.state["trading_days"]
                warning = f"Challenge ending soon: Need {days_needed} more trading days in {days_remaining} days remaining"
                new_warnings.append({
                    "timestamp": timestamp.isoformat(),
                    "rule": "min_trading_days",
                    "description": warning,
                    "value": self.state["trading_days"],
                    "limit": self.rules["min_trading_days"],
                    "days_remaining": days_remaining
                })
                logger.warning(warning)
        
        # Add new warnings to the list
        self.warnings.extend(new_warnings)
        
        return new_warnings
    
    def get_challenge_status(self) -> Dict:
        """
        Get the current challenge status.
        
        Returns:
            Dictionary with challenge status information
        """
        return {
            "firm_type": self.firm_type,
            "account_type": self.account_type,
            "account_size": self.account_size,
            "challenge_phase": self.challenge_phase,
            "current_balance": self.state["current_balance"],
            "current_equity": self.state["current_balance"],  # Assuming no open positions
            "total_profit_loss": self.state["total_profit_loss"],
            "total_profit_loss_pct": self.state["total_profit_loss"] / self.account_size if self.account_size > 0 else 0,
            "max_drawdown_pct": self.state["max_drawdown_pct"],
            "current_drawdown_pct": self.state["current_drawdown_pct"],
            "daily_loss_pct": self.state["daily_loss_pct"],
            "trading_days": self.state["trading_days"],
            "profitable_trading_days": self.state["profitable_trading_days"],
            "profitable_days_pct": self.state["profitable_trading_days"] / self.state["trading_days"] if self.state["trading_days"] > 0 else 0,
            "total_trades": self.state["total_trades"],
            "winning_trades": self.state["winning_trades"],
            "losing_trades": self.state["losing_trades"],
            "win_rate": self.state["winning_trades"] / self.state["total_trades"] if self.state["total_trades"] > 0 else 0,
            "profit_target_reached": self.state["profit_target_reached"],
            "challenge_passed": self.state["challenge_passed"],
            "challenge_failed": self.state["challenge_failed"],
            "failure_reason": self.state["failure_reason"],
            "violations_count": len(self.violations),
            "warnings_count": len(self.warnings),
            "rules": self.rules
        }
    
    def get_detailed_report(self) -> Dict:
        """
        Get a detailed report of the challenge.
        
        Returns:
            Dictionary with detailed challenge report
        """
        status = self.get_challenge_status()
        
        # Add detailed information
        status.update({
            "violations": self.violations,
            "warnings": self.warnings,
            "trade_history": self.state["trade_history"],
            "daily_balance_history": self.state["daily_balance_history"],
            "equity_curve": self.state["equity_curve"]
        })
        
        return status
    
    def save_report(self, filename: str) -> None:
        """
        Save a detailed report to a JSON file.
        
        Args:
            filename: Output filename
        """
        report = self.get_detailed_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved detailed report to {filename}")

def main():
    """Test the rules engine with a simple simulation."""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description='Test the prop firm rules engine')
    parser.add_argument('--firm', type=str, default='TFT', help='Prop firm type (TFT, FTMO, MFF)')
    parser.add_argument('--account-size', type=float, default=100000.0, help='Account size')
    parser.add_argument('--days', type=int, default=30, help='Number of days to simulate')
    parser.add_argument('--trades-per-day', type=int, default=3, help='Average trades per day')
    parser.add_argument('--win-rate', type=float, default=0.6, help='Win rate (0-1)')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (0-1)')
    parser.add_argument('--reward-risk-ratio', type=float, default=1.5, help='Reward-to-risk ratio')
    parser.add_argument('--output', type=str, default='rules_engine_test.json', help='Output filename')
    
    args = parser.parse_args()
    
    # Initialize rules engine
    rules_engine = PropFirmRulesEngine(
        firm_type=args.firm,
        account_size=args.account_size
    )
    
    # Run simulation
    start_date = datetime.now() - timedelta(days=args.days)
    current_date = start_date
    
    for day in range(args.days):
        # Update date
        current_date = start_date + timedelta(days=day)
        
        # Determine number of trades for the day
        num_trades = max(0, int(random.gauss(args.trades_per_day, args.trades_per_day / 2)))
        
        for trade_idx in range(num_trades):
            # Skip if challenge already failed
            if rules_engine.state["challenge_failed"]:
                break
            
            # Generate random trade
            trade_id = f"TRADE_{day}_{trade_idx}"
            instrument = random.choice(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
            direction = random.choice(["BUY", "SELL"])
            
            # Determine if trade is winner
            is_winner = random.random() < args.win_rate
            
            # Calculate profit/loss
            risk_amount = rules_engine.state["current_balance"] * args.risk_per_trade
            if is_winner:
                profit_loss = risk_amount * args.reward_risk_ratio
            else:
                profit_loss = -risk_amount
            
            # Generate trade timestamps
            entry_time = current_date.replace(
                hour=random.randint(8, 16),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )
            trade_duration = random.randint(15, 240)  # 15 minutes to 4 hours
            exit_time = entry_time + timedelta(minutes=trade_duration)
            
            # Record trade
            rules_engine.record_trade(
                trade_id=trade_id,
                instrument=instrument,
                direction=direction,
                entry_price=100.0,  # Dummy price
                exit_price=101.0,   # Dummy price
                position_size=1.0,  # Dummy size
                entry_time=entry_time,
                exit_time=exit_time,
                profit_loss=profit_loss,
                profit_loss_pips=profit_loss / 10.0,  # Dummy pips
                trade_duration_minutes=trade_duration
            )
            
            # Check rules
            rules_violated, violation_reason = rules_engine.check_rules(exit_time)
            if rules_violated:
                print(f"Challenge failed on day {day+1}: {violation_reason}")
                break
        
        # End of day check
        if not rules_engine.state["challenge_failed"] and not rules_engine.state["challenge_passed"]:
            rules_engine.check_rules(current_date.replace(hour=23, minute=59, second=59))
            rules_engine.check_warnings(current_date.replace(hour=23, minute=59, second=59))
        
        # Check if challenge passed or failed
        if rules_engine.state["challenge_passed"]:
            print(f"Challenge passed on day {day+1}!")
            break
        elif rules_engine.state["challenge_failed"]:
            print(f"Challenge failed on day {day+1}: {rules_engine.state['failure_reason']}")
            break
    
    # Save report
    rules_engine.save_report(args.output)
    
    # Print summary
    status = rules_engine.get_challenge_status()
    print("\nChallenge Summary:")
    print(f"Firm: {status['firm_type']}, Account Size: ${status['account_size']:.2f}")
    print(f"Final Balance: ${status['current_balance']:.2f} (P/L: ${status['total_profit_loss']:.2f}, {status['total_profit_loss_pct'] * 100:.2f}%)")
    print(f"Max Drawdown: {status['max_drawdown_pct'] * 100:.2f}%")
    print(f"Trading Days: {status['trading_days']} (Profitable: {status['profitable_trading_days']}, {status['profitable_days_pct'] * 100:.2f}%)")
    print(f"Total Trades: {status['total_trades']} (Win: {status['winning_trades']}, Lose: {status['losing_trades']}, Win Rate: {status['win_rate'] * 100:.2f}%)")
    print(f"Challenge Result: {'PASSED' if status['challenge_passed'] else 'FAILED' if status['challenge_failed'] else 'INCOMPLETE'}")
    if status['failure_reason']:
        print(f"Failure Reason: {status['failure_reason']}")
    print(f"Detailed report saved to {args.output}")

if __name__ == "__main__":
    main()