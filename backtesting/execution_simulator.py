#!/usr/bin/env python3
"""
Execution Simulator

This module simulates realistic trade execution including:
- Slippage based on volatility and market conditions
- Spread modeling based on historical data
- Partial fills and requotes
- Order types (market, limit, stop)
- Realistic latency
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger("execution_simulator")

class ExecutionSimulator:
    """
    Simulates realistic trade execution with slippage, spread, and partial fills.
    
    Features:
    - Realistic slippage based on volatility and market conditions
    - Spread modeling based on historical data or volatility
    - Partial fills and requotes
    - Support for different order types
    - Latency simulation
    """
    
    def __init__(self, 
                 slippage_model: str = "volatility",
                 spread_model: str = "dynamic",
                 partial_fills_enabled: bool = True,
                 requotes_enabled: bool = True,
                 latency_model: str = "random",
                 spread_data_file: Optional[str] = None,
                 volatility_data_file: Optional[str] = None,
                 custom_config: Optional[Dict] = None):
        """
        Initialize the execution simulator.
        
        Args:
            slippage_model: Slippage model type ("none", "fixed", "volatility", "custom")
            spread_model: Spread model type ("fixed", "dynamic", "historical", "custom")
            partial_fills_enabled: Enable partial fills simulation
            requotes_enabled: Enable requotes simulation
            latency_model: Latency model type ("none", "fixed", "random", "realistic")
            spread_data_file: Path to historical spread data file (JSON)
            volatility_data_file: Path to historical volatility data file (JSON)
            custom_config: Custom configuration parameters
        """
        self.slippage_model = slippage_model
        self.spread_model = spread_model
        self.partial_fills_enabled = partial_fills_enabled
        self.requotes_enabled = requotes_enabled
        self.latency_model = latency_model
        
        # Initialize default configuration
        self.config = {
            "slippage": {
                "fixed_pips": 1.0,  # Fixed slippage in pips
                "volatility_factor": 0.1,  # Slippage as fraction of volatility
                "max_slippage_pips": 5.0,  # Maximum slippage in pips
                "market_impact_factor": 0.2,  # Market impact factor for large orders
                "asymmetric_factor": 0.2,  # Asymmetric slippage factor (market direction bias)
            },
            "spread": {
                "fixed_pips": {
                    "EURUSD": 1.0,
                    "GBPUSD": 1.5,
                    "USDJPY": 1.5,
                    "AUDUSD": 1.8,
                    "USDCAD": 2.0,
                    "NZDUSD": 2.2,
                    "USDCHF": 2.0,
                    "EURGBP": 1.8,
                    "EURJPY": 2.0,
                    "GBPJPY": 2.5,
                    "default": 2.0
                },
                "volatility_factor": 0.15,  # Spread as fraction of volatility
                "min_spread_pips": 0.5,  # Minimum spread in pips
                "max_spread_pips": 10.0,  # Maximum spread in pips
                "spread_increase_factor": 1.5,  # Spread increase during high volatility
            },
            "partial_fills": {
                "probability": 0.2,  # Probability of partial fill
                "min_fill_ratio": 0.5,  # Minimum fill ratio
                "max_fill_ratio": 0.9,  # Maximum fill ratio
                "size_threshold": 5.0,  # Size threshold for partial fills (in lots)
            },
            "requotes": {
                "probability": 0.1,  # Probability of requote
                "max_attempts": 3,  # Maximum requote attempts
                "price_change_factor": 0.5,  # Price change factor for requotes (in pips)
                "volatility_factor": 0.2,  # Requote probability increase with volatility
            },
            "latency": {
                "fixed_ms": 50,  # Fixed latency in milliseconds
                "min_ms": 20,  # Minimum latency in milliseconds
                "max_ms": 200,  # Maximum latency in milliseconds
                "jitter_ms": 30,  # Latency jitter in milliseconds
                "timeout_probability": 0.01,  # Probability of timeout
                "timeout_ms": 5000,  # Timeout in milliseconds
            },
            "order_types": {
                "market_slippage_factor": 1.0,  # Slippage factor for market orders
                "limit_fill_probability": 0.9,  # Probability of limit order fill
                "stop_slippage_factor": 1.5,  # Slippage factor for stop orders
            }
        }
        
        # Override with custom configuration if provided
        if custom_config:
            self._update_config(custom_config)
        
        # Load historical spread data if provided
        self.spread_data = self._load_spread_data(spread_data_file)
        
        # Load historical volatility data if provided
        self.volatility_data = self._load_volatility_data(volatility_data_file)
        
        # Initialize execution statistics
        self.stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_partially_filled": 0,
            "orders_rejected": 0,
            "orders_requoted": 0,
            "orders_timed_out": 0,
            "total_slippage_pips": 0,
            "avg_slippage_pips": 0,
            "max_slippage_pips": 0,
            "avg_spread_pips": 0,
            "avg_latency_ms": 0,
            "instrument_stats": {}
        }
        
        logger.info(f"Initialized execution simulator with slippage model: {slippage_model}, spread model: {spread_model}")
    
    def _update_config(self, custom_config: Dict) -> None:
        """
        Update configuration with custom parameters.
        
        Args:
            custom_config: Custom configuration parameters
        """
        # Recursively update nested dictionaries
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v
        
        update_dict(self.config, custom_config)
        logger.info("Updated execution simulator configuration with custom parameters")
    
    def _load_spread_data(self, spread_data_file: Optional[str]) -> Dict:
        """
        Load historical spread data from file.
        
        Args:
            spread_data_file: Path to historical spread data file (JSON)
            
        Returns:
            Dictionary of historical spread data by instrument
        """
        if not spread_data_file:
            return {}
        
        try:
            with open(spread_data_file, 'r') as f:
                spread_data = json.load(f)
            
            logger.info(f"Loaded historical spread data for {len(spread_data)} instruments")
            return spread_data
        except Exception as e:
            logger.error(f"Error loading spread data from {spread_data_file}: {str(e)}")
            return {}
    
    def _load_volatility_data(self, volatility_data_file: Optional[str]) -> Dict:
        """
        Load historical volatility data from file.
        
        Args:
            volatility_data_file: Path to historical volatility data file (JSON)
            
        Returns:
            Dictionary of historical volatility data by instrument
        """
        if not volatility_data_file:
            return {}
        
        try:
            with open(volatility_data_file, 'r') as f:
                volatility_data = json.load(f)
            
            logger.info(f"Loaded historical volatility data for {len(volatility_data)} instruments")
            return volatility_data
        except Exception as e:
            logger.error(f"Error loading volatility data from {volatility_data_file}: {str(e)}")
            return {}
    
    def _get_instrument_volatility(self, instrument: str, timestamp: Optional[datetime] = None) -> float:
        """
        Get volatility for an instrument at a specific time.
        
        Args:
            instrument: Instrument symbol
            timestamp: Timestamp (default: None, uses current volatility)
            
        Returns:
            Volatility in pips
        """
        # Use historical volatility data if available
        if self.volatility_data and instrument in self.volatility_data:
            if timestamp and isinstance(self.volatility_data[instrument], dict):
                # Find closest timestamp
                closest_ts = min(self.volatility_data[instrument].keys(), 
                                key=lambda ts: abs(datetime.fromisoformat(ts) - timestamp))
                return self.volatility_data[instrument][closest_ts]
            elif isinstance(self.volatility_data[instrument], (int, float)):
                return self.volatility_data[instrument]
        
        # Default volatility values by instrument
        default_volatility = {
            "EURUSD": 50.0,
            "GBPUSD": 70.0,
            "USDJPY": 60.0,
            "AUDUSD": 55.0,
            "USDCAD": 60.0,
            "NZDUSD": 50.0,
            "USDCHF": 55.0,
            "EURGBP": 45.0,
            "EURJPY": 65.0,
            "GBPJPY": 80.0
        }
        
        return default_volatility.get(instrument, 60.0)
    
    def _get_instrument_spread(self, instrument: str, timestamp: Optional[datetime] = None, volatility: Optional[float] = None) -> float:
        """
        Get spread for an instrument at a specific time.
        
        Args:
            instrument: Instrument symbol
            timestamp: Timestamp (default: None, uses current spread)
            volatility: Volatility in pips (default: None, calculated if needed)
            
        Returns:
            Spread in pips
        """
        if self.spread_model == "fixed":
            # Use fixed spread from configuration
            return self.config["spread"]["fixed_pips"].get(instrument, self.config["spread"]["fixed_pips"]["default"])
        
        elif self.spread_model == "historical" and self.spread_data:
            # Use historical spread data if available
            if instrument in self.spread_data:
                if timestamp and isinstance(self.spread_data[instrument], dict):
                    # Find closest timestamp
                    closest_ts = min(self.spread_data[instrument].keys(), 
                                    key=lambda ts: abs(datetime.fromisoformat(ts) - timestamp))
                    return self.spread_data[instrument][closest_ts]
                elif isinstance(self.spread_data[instrument], (int, float)):
                    return self.spread_data[instrument]
        
        elif self.spread_model == "dynamic" or self.spread_model == "custom":
            # Calculate spread based on volatility
            if volatility is None:
                volatility = self._get_instrument_volatility(instrument, timestamp)
            
            # Base spread from configuration
            base_spread = self.config["spread"]["fixed_pips"].get(instrument, self.config["spread"]["fixed_pips"]["default"])
            
            # Adjust spread based on volatility
            volatility_spread = volatility * self.config["spread"]["volatility_factor"]
            
            # Combine base and volatility-based spread
            spread = base_spread + volatility_spread
            
            # Apply limits
            spread = max(self.config["spread"]["min_spread_pips"], 
                        min(self.config["spread"]["max_spread_pips"], spread))
            
            # Add random noise
            spread *= random.uniform(0.9, 1.1)
            
            return spread
        
        # Default to fixed spread if all else fails
        return self.config["spread"]["fixed_pips"].get(instrument, self.config["spread"]["fixed_pips"]["default"])
    
    def _calculate_slippage(self, 
                           instrument: str, 
                           order_type: str, 
                           direction: str, 
                           size: float, 
                           volatility: Optional[float] = None,
                           market_direction: Optional[str] = None) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            instrument: Instrument symbol
            order_type: Order type ("market", "limit", "stop")
            direction: Order direction ("buy" or "sell")
            size: Order size in lots
            volatility: Volatility in pips (default: None, calculated if needed)
            market_direction: Market direction ("up", "down", or None)
            
        Returns:
            Slippage in pips (positive for unfavorable, negative for favorable)
        """
        if self.slippage_model == "none":
            return 0.0
        
        elif self.slippage_model == "fixed":
            # Use fixed slippage from configuration
            base_slippage = self.config["slippage"]["fixed_pips"]
            
            # Adjust for order type
            if order_type == "market":
                base_slippage *= self.config["order_types"]["market_slippage_factor"]
            elif order_type == "stop":
                base_slippage *= self.config["order_types"]["stop_slippage_factor"]
            elif order_type == "limit":
                # Limit orders typically have less slippage
                base_slippage *= 0.5
            
            # Random variation
            slippage = base_slippage * random.uniform(0.5, 1.5)
            
            # Ensure slippage is positive (unfavorable) most of the time
            if random.random() < 0.8:  # 80% chance of unfavorable slippage
                return abs(slippage)
            else:
                return -abs(slippage) * 0.5  # Favorable slippage is typically smaller
        
        elif self.slippage_model == "volatility" or self.slippage_model == "custom":
            # Get volatility if not provided
            if volatility is None:
                volatility = self._get_instrument_volatility(instrument)
            
            # Base slippage as a fraction of volatility
            base_slippage = volatility * self.config["slippage"]["volatility_factor"]
            
            # Adjust for order size (market impact)
            size_factor = 1.0 + (size / 10.0) * self.config["slippage"]["market_impact_factor"]
            
            # Adjust for order type
            if order_type == "market":
                type_factor = self.config["order_types"]["market_slippage_factor"]
            elif order_type == "stop":
                type_factor = self.config["order_types"]["stop_slippage_factor"]
            elif order_type == "limit":
                # Limit orders typically have less slippage
                type_factor = 0.5
            else:
                type_factor = 1.0
            
            # Calculate slippage
            slippage = base_slippage * size_factor * type_factor
            
            # Add asymmetric factor based on market direction
            if market_direction:
                asymmetric_factor = self.config["slippage"]["asymmetric_factor"]
                if (direction == "buy" and market_direction == "up") or (direction == "sell" and market_direction == "down"):
                    # Unfavorable slippage when trading with the market
                    slippage *= (1.0 + asymmetric_factor)
                elif (direction == "buy" and market_direction == "down") or (direction == "sell" and market_direction == "up"):
                    # Potentially favorable slippage when trading against the market
                    if random.random() < 0.3:  # 30% chance of favorable slippage
                        slippage = -slippage * 0.5
            
            # Add random noise
            slippage *= random.uniform(0.8, 1.2)
            
            # Apply maximum slippage limit
            if slippage > 0:  # Only limit unfavorable slippage
                slippage = min(slippage, self.config["slippage"]["max_slippage_pips"])
            
            return slippage
        
        # Default to zero slippage if all else fails
        return 0.0
    
    def _simulate_latency(self) -> Tuple[float, bool]:
        """
        Simulate network latency for order execution.
        
        Returns:
            Tuple of (latency_ms, timeout)
        """
        if self.latency_model == "none":
            return 0.0, False
        
        elif self.latency_model == "fixed":
            # Use fixed latency from configuration
            latency = self.config["latency"]["fixed_ms"]
            
            # Check for timeout
            timeout = random.random() < self.config["latency"]["timeout_probability"]
            
            return latency, timeout
        
        elif self.latency_model == "random":
            # Random latency between min and max
            latency = random.uniform(
                self.config["latency"]["min_ms"],
                self.config["latency"]["max_ms"]
            )
            
            # Check for timeout
            timeout = random.random() < self.config["latency"]["timeout_probability"]
            
            return latency, timeout
        
        elif self.latency_model == "realistic":
            # Base latency
            base_latency = random.uniform(
                self.config["latency"]["min_ms"],
                self.config["latency"]["max_ms"]
            )
            
            # Add jitter
            jitter = random.uniform(
                -self.config["latency"]["jitter_ms"],
                self.config["latency"]["jitter_ms"]
            )
            
            latency = max(1.0, base_latency + jitter)
            
            # Occasionally add a spike
            if random.random() < 0.05:  # 5% chance of latency spike
                latency *= random.uniform(2.0, 5.0)
            
            # Check for timeout
            timeout = random.random() < self.config["latency"]["timeout_probability"]
            
            return latency, timeout
        
        # Default to zero latency if all else fails
        return 0.0, False
    
    def _check_requote(self, 
                      instrument: str, 
                      order_type: str, 
                      volatility: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if an order should be requoted.
        
        Args:
            instrument: Instrument symbol
            order_type: Order type ("market", "limit", "stop")
            volatility: Volatility in pips (default: None, calculated if needed)
            
        Returns:
            Tuple of (requote, price_change_pips)
        """
        if not self.requotes_enabled:
            return False, 0.0
        
        # Get volatility if not provided
        if volatility is None:
            volatility = self._get_instrument_volatility(instrument)
        
        # Base requote probability
        requote_probability = self.config["requotes"]["probability"]
        
        # Adjust for volatility
        volatility_factor = volatility / 50.0  # Normalize to typical volatility
        requote_probability *= (1.0 + (volatility_factor - 1.0) * self.config["requotes"]["volatility_factor"])
        
        # Adjust for order type
        if order_type == "market":
            # Market orders are more likely to be requoted
            requote_probability *= 1.2
        elif order_type == "limit":
            # Limit orders are less likely to be requoted
            requote_probability *= 0.5
        
        # Check if requote occurs
        requote = random.random() < requote_probability
        
        # Calculate price change for requote
        if requote:
            # Base price change as a fraction of volatility
            price_change = volatility * self.config["requotes"]["price_change_factor"] * random.uniform(0.5, 1.5)
            
            # Randomly determine direction of price change
            if random.random() < 0.5:
                price_change = -price_change
            
            return True, price_change
        
        return False, 0.0
    
    def _check_partial_fill(self, 
                           instrument: str, 
                           order_type: str, 
                           size: float) -> Tuple[bool, float]:
        """
        Check if an order should be partially filled.
        
        Args:
            instrument: Instrument symbol
            order_type: Order type ("market", "limit", "stop")
            size: Order size in lots
            
        Returns:
            Tuple of (partial_fill, fill_ratio)
        """
        if not self.partial_fills_enabled:
            return False, 1.0
        
        # Only apply partial fills to orders above the size threshold
        if size < self.config["partial_fills"]["size_threshold"]:
            return False, 1.0
        
        # Base partial fill probability
        partial_fill_probability = self.config["partial_fills"]["probability"]
        
        # Adjust for order size
        size_factor = size / self.config["partial_fills"]["size_threshold"]
        partial_fill_probability *= min(2.0, size_factor)
        
        # Adjust for order type
        if order_type == "market":
            # Market orders are less likely to be partially filled
            partial_fill_probability *= 0.8
        elif order_type == "limit":
            # Limit orders are more likely to be partially filled
            partial_fill_probability *= 1.2
        
        # Check if partial fill occurs
        partial_fill = random.random() < partial_fill_probability
        
        # Calculate fill ratio
        if partial_fill:
            fill_ratio = random.uniform(
                self.config["partial_fills"]["min_fill_ratio"],
                self.config["partial_fills"]["max_fill_ratio"]
            )
            
            return True, fill_ratio
        
        return False, 1.0
    
    def simulate_fill(self, 
                     order: Dict, 
                     market_data: Dict,
                     timestamp: Optional[datetime] = None) -> Dict:
        """
        Simulate order execution with realistic market conditions.
        
        Args:
            order: Order details (instrument, type, direction, price, size)
            market_data: Market data (current_price, volatility, market_direction)
            timestamp: Timestamp for the simulation (default: current time)
            
        Returns:
            Dictionary with fill details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract order details
        instrument = order.get("instrument", "")
        order_type = order.get("type", "market").lower()
        direction = order.get("direction", "buy").lower()
        requested_price = order.get("price")
        size = order.get("size", 1.0)
        
        # Extract market data
        current_price = market_data.get("current_price")
        volatility = market_data.get("volatility", self._get_instrument_volatility(instrument, timestamp))
        market_direction = market_data.get("market_direction")
        
        # Initialize fill result
        fill_result = {
            "order_id": order.get("order_id", ""),
            "instrument": instrument,
            "type": order_type,
            "direction": direction,
            "requested_price": requested_price,
            "requested_size": size,
            "timestamp": timestamp.isoformat(),
            "status": "filled",
            "fill_price": None,
            "fill_size": size,
            "slippage_pips": 0.0,
            "spread_pips": 0.0,
            "latency_ms": 0.0,
            "requoted": False,
            "partially_filled": False,
            "timed_out": False,
            "rejection_reason": None
        }
        
        # Update statistics
        self.stats["orders_submitted"] += 1
        
        # Initialize instrument stats if not exists
        if instrument not in self.stats["instrument_stats"]:
            self.stats["instrument_stats"][instrument] = {
                "orders_submitted": 0,
                "orders_filled": 0,
                "orders_partially_filled": 0,
                "orders_rejected": 0,
                "orders_requoted": 0,
                "total_slippage_pips": 0,
                "avg_slippage_pips": 0,
                "max_slippage_pips": 0,
                "avg_spread_pips": 0
            }
        
        self.stats["instrument_stats"][instrument]["orders_submitted"] += 1
        
        # Simulate latency
        latency, timeout = self._simulate_latency()
        fill_result["latency_ms"] = latency
        
        # Check for timeout
        if timeout:
            fill_result["status"] = "rejected"
            fill_result["timed_out"] = True
            fill_result["rejection_reason"] = "Execution timeout"
            
            self.stats["orders_timed_out"] += 1
            return fill_result
        
        # Get current spread
        spread = self._get_instrument_spread(instrument, timestamp, volatility)
        fill_result["spread_pips"] = spread
        
        # Update spread statistics
        self.stats["avg_spread_pips"] = (self.stats["avg_spread_pips"] * (self.stats["orders_submitted"] - 1) + spread) / self.stats["orders_submitted"]
        self.stats["instrument_stats"][instrument]["avg_spread_pips"] = (
            self.stats["instrument_stats"][instrument]["avg_spread_pips"] * 
            (self.stats["instrument_stats"][instrument]["orders_submitted"] - 1) + spread
        ) / self.stats["instrument_stats"][instrument]["orders_submitted"]
        
        # Check for requote
        requote, price_change = self._check_requote(instrument, order_type, volatility)
        
        if requote:
            fill_result["status"] = "requoted"
            fill_result["requoted"] = True
            
            # Adjust price for requote
            if requested_price is not None:
                fill_result["fill_price"] = requested_price + price_change
            else:
                # For market orders, adjust from current price
                if direction == "buy":
                    fill_result["fill_price"] = current_price + spread/2 + price_change
                else:
                    fill_result["fill_price"] = current_price - spread/2 + price_change
            
            self.stats["orders_requoted"] += 1
            self.stats["instrument_stats"][instrument]["orders_requoted"] += 1
            
            return fill_result
        
        # Check for partial fill
        partial_fill, fill_ratio = self._check_partial_fill(instrument, order_type, size)
        
        if partial_fill:
            fill_result["partially_filled"] = True
            fill_result["fill_size"] = size * fill_ratio
            
            self.stats["orders_partially_filled"] += 1
            self.stats["instrument_stats"][instrument]["orders_partially_filled"] += 1
        
        # Calculate slippage
        slippage = self._calculate_slippage(
            instrument, order_type, direction, size, volatility, market_direction
        )
        
        fill_result["slippage_pips"] = slippage
        
        # Update slippage statistics
        self.stats["total_slippage_pips"] += abs(slippage)
        self.stats["avg_slippage_pips"] = self.stats["total_slippage_pips"] / self.stats["orders_submitted"]
        self.stats["max_slippage_pips"] = max(self.stats["max_slippage_pips"], abs(slippage))
        
        self.stats["instrument_stats"][instrument]["total_slippage_pips"] += abs(slippage)
        self.stats["instrument_stats"][instrument]["avg_slippage_pips"] = (
            self.stats["instrument_stats"][instrument]["total_slippage_pips"] / 
            self.stats["instrument_stats"][instrument]["orders_submitted"]
        )
        self.stats["instrument_stats"][instrument]["max_slippage_pips"] = max(
            self.stats["instrument_stats"][instrument]["max_slippage_pips"], 
            abs(slippage)
        )
        
        # Calculate fill price
        if order_type == "market":
            # Market order
            if direction == "buy":
                # Buy at ask price (current price + half spread) plus slippage
                fill_result["fill_price"] = current_price + spread/2 + slippage/10000
            else:
                # Sell at bid price (current price - half spread) minus slippage
                fill_result["fill_price"] = current_price - spread/2 - slippage/10000
        
        elif order_type == "limit":
            # Limit order
            if requested_price is None:
                fill_result["status"] = "rejected"
                fill_result["rejection_reason"] = "Missing limit price"
                
                self.stats["orders_rejected"] += 1
                self.stats["instrument_stats"][instrument]["orders_rejected"] += 1
                
                return fill_result
            
            if direction == "buy":
                # Buy limit: only fills if current price <= limit price
                if current_price + spread/2 <= requested_price:
                    # Fill at limit price or better
                    fill_result["fill_price"] = min(requested_price, current_price + spread/2 + slippage/10000)
                else:
                    # Not filled
                    fill_result["status"] = "rejected"
                    fill_result["rejection_reason"] = "Price moved away from limit"
                    
                    self.stats["orders_rejected"] += 1
                    self.stats["instrument_stats"][instrument]["orders_rejected"] += 1
                    
                    return fill_result
            else:
                # Sell limit: only fills if current price >= limit price
                if current_price - spread/2 >= requested_price:
                    # Fill at limit price or better
                    fill_result["fill_price"] = max(requested_price, current_price - spread/2 - slippage/10000)
                else:
                    # Not filled
                    fill_result["status"] = "rejected"
                    fill_result["rejection_reason"] = "Price moved away from limit"
                    
                    self.stats["orders_rejected"] += 1
                    self.stats["instrument_stats"][instrument]["orders_rejected"] += 1
                    
                    return fill_result
        
        elif order_type == "stop":
            # Stop order
            if requested_price is None:
                fill_result["status"] = "rejected"
                fill_result["rejection_reason"] = "Missing stop price"
                
                self.stats["orders_rejected"] += 1
                self.stats["instrument_stats"][instrument]["orders_rejected"] += 1
                
                return fill_result
            
            if direction == "buy":
                # Buy stop: only fills if current price >= stop price
                if current_price >= requested_price:
                    # Fill at stop price or worse (with slippage)
                    fill_result["fill_price"] = requested_price + slippage/10000
                else:
                    # Not filled
                    fill_result["status"] = "rejected"
                    fill_result["rejection_reason"] = "Price did not reach stop level"
                    
                    self.stats["orders_rejected"] += 1
                    self.stats["instrument_stats"][instrument]["orders_rejected"] += 1
                    
                    return fill_result
            else:
                # Sell stop: only fills if current price <= stop price
                if current_price <= requested_price:
                    # Fill at stop price or worse (with slippage)
                    fill_result["fill_price"] = requested_price - slippage/10000
                else:
                    # Not filled
                    fill_result["status"] = "rejected"
                    fill_result["rejection_reason"] = "Price did not reach stop level"
                    
                    self.stats["orders_rejected"] += 1
                    self.stats["instrument_stats"][instrument]["orders_rejected"] += 1
                    
                    return fill_result
        
        # Order filled successfully
        self.stats["orders_filled"] += 1
        self.stats["instrument_stats"][instrument]["orders_filled"] += 1
        
        # Update latency statistics
        self.stats["avg_latency_ms"] = (self.stats["avg_latency_ms"] * (self.stats["orders_filled"] - 1) + latency) / self.stats["orders_filled"]
        
        return fill_result
    
    def get_statistics(self) -> Dict:
        """
        Get execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return self.stats
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_partially_filled": 0,
            "orders_rejected": 0,
            "orders_requoted": 0,
            "orders_timed_out": 0,
            "total_slippage_pips": 0,
            "avg_slippage_pips": 0,
            "max_slippage_pips": 0,
            "avg_spread_pips": 0,
            "avg_latency_ms": 0,
            "instrument_stats": {}
        }
        
        logger.info("Reset execution statistics")
    
    def save_statistics(self, filename: str) -> None:
        """
        Save execution statistics to a JSON file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved execution statistics to {filename}")

def main():
    """Test the execution simulator with sample orders."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the execution simulator')
    parser.add_argument('--slippage-model', type=str, default='volatility', 
                        choices=['none', 'fixed', 'volatility', 'custom'],
                        help='Slippage model type')
    parser.add_argument('--spread-model', type=str, default='dynamic', 
                        choices=['fixed', 'dynamic', 'historical', 'custom'],
                        help='Spread model type')
    parser.add_argument('--partial-fills', action='store_true', 
                        help='Enable partial fills simulation')
    parser.add_argument('--requotes', action='store_true', 
                        help='Enable requotes simulation')
    parser.add_argument('--latency-model', type=str, default='random', 
                        choices=['none', 'fixed', 'random', 'realistic'],
                        help='Latency model type')
    parser.add_argument('--orders', type=int, default=100, 
                        help='Number of orders to simulate')
    parser.add_argument('--output', type=str, default='execution_stats.json', 
                        help='Output filename for statistics')
    
    args = parser.parse_args()
    
    # Initialize execution simulator
    simulator = ExecutionSimulator(
        slippage_model=args.slippage_model,
        spread_model=args.spread_model,
        partial_fills_enabled=args.partial_fills,
        requotes_enabled=args.requotes,
        latency_model=args.latency_model
    )
    
    # Sample instruments
    instruments = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    
    # Sample order types
    order_types = ["market", "limit", "stop"]
    
    # Sample directions
    directions = ["buy", "sell"]
    
    # Sample market directions
    market_directions = ["up", "down", None]
    
    # Simulate orders
    for i in range(args.orders):
        # Generate random order
        instrument = random.choice(instruments)
        order_type = random.choice(order_types)
        direction = random.choice(directions)
        size = random.uniform(0.1, 10.0)
        
        # Generate random market data
        current_price = 1.0 + random.uniform(-0.1, 0.1)
        volatility = random.uniform(30.0, 100.0)
        market_direction = random.choice(market_directions)
        
        # Create order
        order = {
            "order_id": f"ORDER_{i}",
            "instrument": instrument,
            "type": order_type,
            "direction": direction,
            "size": size
        }
        
        # Add price for limit and stop orders
        if order_type == "limit":
            if direction == "buy":
                order["price"] = current_price * random.uniform(0.95, 0.99)  # Buy limit below current price
            else:
                order["price"] = current_price * random.uniform(1.01, 1.05)  # Sell limit above current price
        elif order_type == "stop":
            if direction == "buy":
                order["price"] = current_price * random.uniform(1.01, 1.05)  # Buy stop above current price
            else:
                order["price"] = current_price * random.uniform(0.95, 0.99)  # Sell stop below current price
        
        # Create market data
        market_data = {
            "current_price": current_price,
            "volatility": volatility,
            "market_direction": market_direction
        }
        
        # Simulate fill
        fill_result = simulator.simulate_fill(order, market_data)
        
        # Print result
        print(f"Order {i+1}/{args.orders}: {instrument} {direction} {order_type} - Status: {fill_result['status']}")
        if fill_result['status'] == 'filled':
            print(f"  Fill Price: {fill_result['fill_price']:.5f}, Size: {fill_result['fill_size']:.2f}, Slippage: {fill_result['slippage_pips']:.2f} pips")
        elif fill_result['status'] == 'requoted':
            print(f"  Requoted to: {fill_result['fill_price']:.5f}")
        elif fill_result['status'] == 'rejected':
            print(f"  Rejected: {fill_result['rejection_reason']}")
    
    # Print statistics
    stats = simulator.get_statistics()
    print("\nExecution Statistics:")
    print(f"Orders Submitted: {stats['orders_submitted']}")
    print(f"Orders Filled: {stats['orders_filled']} ({stats['orders_filled'] / stats['orders_submitted'] * 100:.2f}%)")
    print(f"Orders Partially Filled: {stats['orders_partially_filled']} ({stats['orders_partially_filled'] / stats['orders_submitted'] * 100:.2f}%)")
    print(f"Orders Rejected: {stats['orders_rejected']} ({stats['orders_rejected'] / stats['orders_submitted'] * 100:.2f}%)")
    print(f"Orders Requoted: {stats['orders_requoted']} ({stats['orders_requoted'] / stats['orders_submitted'] * 100:.2f}%)")
    print(f"Orders Timed Out: {stats['orders_timed_out']} ({stats['orders_timed_out'] / stats['orders_submitted'] * 100:.2f}%)")
    print(f"Average Slippage: {stats['avg_slippage_pips']:.2f} pips")
    print(f"Maximum Slippage: {stats['max_slippage_pips']:.2f} pips")
    print(f"Average Spread: {stats['avg_spread_pips']:.2f} pips")
    print(f"Average Latency: {stats['avg_latency_ms']:.2f} ms")
    
    # Save statistics
    simulator.save_statistics(args.output)
    print(f"\nStatistics saved to {args.output}")

if __name__ == "__main__":
    main()